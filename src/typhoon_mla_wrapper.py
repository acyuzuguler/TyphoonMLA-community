

import torch
import numpy as np 
import math 

from typhoon_mla import TyphoonMLA

from src.helpers import convert_absorb_to_naive, merge_kv_cache

class TyphoonMLATest:
    def __init__(self, bsz, seqlens, n_heads, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, v_head_dim, is_stage1_absorb, run_in_single_stage, softmax_scale, data_layout, device, dtype) -> None:
        self.bsz = bsz 
        self.seqlens = seqlens
        self.n_heads = n_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim
        self.device = device 
        self.dtype = dtype 
        self.data_layout = data_layout
        self.softmax_scale = softmax_scale

        self.is_stage1_absorb = is_stage1_absorb

        self.seqlen_stage1 = seqlens[0]
        self.seqlen_stage2 = seqlens[1]

        self.run_in_single_stage = run_in_single_stage

        self.typhoon_mla = TyphoonMLA(is_stage1_absorb, run_in_single_stage, device, dtype)
        self.typhoon_mla.plan(bsz, seqlens, n_heads, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, v_head_dim)

    def generate_input_data(self):
        q_nope = torch.rand(size=[self.bsz, self.n_heads, self.qk_nope_head_dim], device=self.device, dtype=self.dtype)
        q_rope = torch.rand(size=[self.bsz, self.n_heads, self.qk_rope_head_dim], device=self.device, dtype=self.dtype)
        q = torch.cat([q_nope, q_rope], dim=-1)

        wkv_b = 1e-2*torch.randn(size=[self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), self.kv_lora_rank], dtype=self.dtype, device=self.device)
        wkv_b1, wkv_b2 = wkv_b.view(self.n_heads, (self.qk_nope_head_dim + self.v_head_dim), self.kv_lora_rank).split([self.qk_nope_head_dim, self.v_head_dim], dim=1)

        kv_cache_stage1, pe_cache_stage1, k_cache_stage1, v_cache_stage1 = None, None, None, None
        if self.seqlen_stage1 > 0:
            kv_cache_stage1 = torch.randn(size=(self.seqlens[0], 1, self.kv_lora_rank), dtype=self.dtype, device=self.device)
            pe_cache_stage1 = torch.randn(size=(self.seqlens[0], 1, self.qk_rope_head_dim), dtype=self.dtype, device=self.device)  
            if not self.is_stage1_absorb:
                k_cache_stage1, v_cache_stage1 = convert_absorb_to_naive(kv_cache_stage1, pe_cache_stage1, wkv_b1, wkv_b2)

        kv_cache_stage2, pe_cache_stage2 = None, None
        if self.seqlen_stage2 > 0:
            kv_cache_stage2 = torch.randn((sum(self.seqlens[1:]), 1, self.kv_lora_rank), dtype=self.dtype, device=self.device)
            pe_cache_stage2 = torch.randn((sum(self.seqlens[1:]), 1, self.qk_rope_head_dim), dtype=self.dtype, device=self.device)

        if self.run_in_single_stage:
            kv_cache_stage1, pe_cache_stage1 = merge_kv_cache(kv_cache_stage1, pe_cache_stage1, kv_cache_stage2, pe_cache_stage2, self.bsz)
            kv_cache_stage2, pe_cache_stage2 = None, None
            k_cache_stage1, v_cache_stage1 = None, None

        return q, kv_cache_stage1, pe_cache_stage1, kv_cache_stage2, pe_cache_stage2, k_cache_stage1, v_cache_stage1, wkv_b1, wkv_b2
        
    def convert_to_dense(self, kv_cache_stage1, pe_cache_stage1, kv_cache_stage2, pe_cache_stage2):
        # if run in single stage, kv_cache_stage1 and pe_cache_stage1 are already dense
        if self.run_in_single_stage:
            return kv_cache_stage1, pe_cache_stage1

        dense_kv_cache = torch.zeros(size=[self.bsz, (self.seqlen_stage1+self.seqlen_stage2), 1, self.kv_lora_rank], dtype=self.dtype, device=self.device)
        dense_pe_cache = torch.zeros(size=[self.bsz, (self.seqlen_stage1+self.seqlen_stage2), 1, self.qk_rope_head_dim], dtype=self.dtype, device=self.device)

        for b in range(self.bsz):
            if self.seqlen_stage1 > 0:
                dense_kv_cache[b, :self.seqlen_stage1, :, :] = kv_cache_stage1
                dense_pe_cache[b, :self.seqlen_stage1, :, :] = pe_cache_stage1
            if self.seqlen_stage2 > 0:
                dense_kv_cache[b, self.seqlen_stage1:, :, :] = kv_cache_stage2[b*self.seqlen_stage2:(b+1)*self.seqlen_stage2, :, :]
                dense_pe_cache[b, self.seqlen_stage1:, :, :] = pe_cache_stage2[b*self.seqlen_stage2:(b+1)*self.seqlen_stage2, :, :]

        dense_kv_cache = dense_kv_cache.view(self.bsz*(self.seqlen_stage1+self.seqlen_stage2), 1, self.kv_lora_rank)
        dense_pe_cache = dense_pe_cache.view(self.bsz*(self.seqlen_stage1+self.seqlen_stage2), 1, self.qk_rope_head_dim)
        return dense_kv_cache, dense_pe_cache

    def baseline(self, q, dense_k_cache, dense_v_cache):
        dense_k_cache = dense_k_cache.view(self.bsz, -1, self.n_heads, self.qk_head_dim)
        dense_v_cache = dense_v_cache.view(self.bsz, -1, self.n_heads, self.v_head_dim)
        return torch.nn.functional.scaled_dot_product_attention(q.unsqueeze(2), dense_k_cache.transpose(1,2), dense_v_cache.transpose(1,2)).squeeze(2)

    def perf(self, warm_up=25, n_repeat=100):
        for i in range(warm_up):
            q, kv_cache_stage1, pe_cache_stage1, kv_cache_stage2, pe_cache_stage2, k_cache_stage1, v_cache_stage1, wkv_b1, wkv_b2 = self.generate_input_data()
            self.run(q, kv_cache_stage1, pe_cache_stage1, kv_cache_stage2, pe_cache_stage2, k_cache_stage1, v_cache_stage1, wkv_b1, wkv_b2)

        start = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
        end = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]

        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device=self.device)

        elapsed  = []
        for i in range(n_repeat):
            q, kv_cache_stage1, pe_cache_stage1, kv_cache_stage2, pe_cache_stage2, k_cache_stage1, v_cache_stage1, wkv_b1, wkv_b2 = self.generate_input_data()

            cache.zero_()

            start[i].record()
            self.run(q, kv_cache_stage1, pe_cache_stage1, kv_cache_stage2, pe_cache_stage2, k_cache_stage1, v_cache_stage1, wkv_b1, wkv_b2)
            end[i].record()      
            
        torch.cuda.synchronize()
        elapsed = [start[i].elapsed_time(end[i]) for i in range(n_repeat)]

        return np.median(elapsed).item()
    
    def run(self, q, kv_cache_stage1, pe_cache_stage1, kv_cache_stage2, pe_cache_stage2, k_cache_stage1, v_cache_stage1, wkv_b1, wkv_b2):
        return self.typhoon_mla.run(q, kv_cache_stage1, pe_cache_stage1, kv_cache_stage2, pe_cache_stage2, k_cache_stage1, v_cache_stage1, wkv_b1, wkv_b2)

def func_test():
    dtype = torch.float16
    device = torch.device("cuda")
    torch.manual_seed(0)

    bsz = 8
    seqlen_stage1 = 32
    seqlen_stage2 = 64
    seqlens = [seqlen_stage1] + [seqlen_stage2] * bsz

    n_heads = 128
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    v_head_dim = 128
    softmax_scale = 1 / math.sqrt(qk_head_dim)

    is_stage1_absorb = False
    run_in_single_stage = True 

    mla_flashinfer_test = TyphoonMLATest(bsz, seqlens, n_heads, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, v_head_dim, is_stage1_absorb, run_in_single_stage, softmax_scale, data_layout="NHD", device=device, dtype=dtype)
    q, kv_cache_stage1, pe_cache_stage1, kv_cache_stage2, pe_cache_stage2, k_cache_stage1, v_cache_stage1, wkv_b1, wkv_b2 = mla_flashinfer_test.generate_input_data()
    out = mla_flashinfer_test.run(q, kv_cache_stage1, pe_cache_stage1, kv_cache_stage2, pe_cache_stage2, k_cache_stage1, v_cache_stage1, wkv_b1, wkv_b2)

    dense_kv_cache, dense_pe_cache = mla_flashinfer_test.convert_to_dense(kv_cache_stage1, pe_cache_stage1, kv_cache_stage2, pe_cache_stage2)
    dense_k_cache, dense_v_cache = convert_absorb_to_naive(dense_kv_cache, dense_pe_cache, wkv_b1, wkv_b2)

    gt = mla_flashinfer_test.baseline(q, dense_k_cache, dense_v_cache)

    print(out[0,0,:])
    print(gt[0,0,:])
    
    assert torch.allclose(out, gt, atol=0.125, rtol=0), " err: {:.2e}".format(torch.max(torch.abs(out-gt)))
    print("Test OK.")    

def benchmark():
    dtype = torch.float16
    device = torch.device("cuda")
    torch.manual_seed(0)

    threshold = 128

    seqlen_stage1 = 4096
    seqlen_stage2 = 128
    
    n_heads = 128
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    v_head_dim = 128
    softmax_scale = 1 / math.sqrt(qk_head_dim)

    bszs = [1,2,4,8,16,32,64,128,256,512,1024]
    elapseds = []
    for bsz in bszs:
        seqlens = [seqlen_stage1] + [seqlen_stage2] * bsz

        is_stage1_absorb = bsz < threshold 
        run_in_single_stage = bsz < threshold 

        mla_flashinfer_test = TyphoonMLATest(bsz, seqlens, n_heads, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, v_head_dim, is_stage1_absorb, run_in_single_stage, softmax_scale, data_layout="NHD", device=device, dtype=dtype)
        m_elapsed = mla_flashinfer_test.perf()
        elapseds.append(m_elapsed)
        print("bsz: {:<5}\telapsed: {:.2f} ms".format(bsz, m_elapsed))

    print(", ".join(["{:.3f}".format(float(e)) for e in elapseds]))

if __name__=="__main__":
    func_test()
    benchmark()
