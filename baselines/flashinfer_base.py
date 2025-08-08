
import torch
import flashinfer
import numpy as np 
import math 

import matplotlib.pyplot as plt 

class MLAAbsorbFlashInferTest():
    def __init__(self, bsz, seqlens, n_heads, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, v_head_dim, with_projections, softmax_scale, data_layout, device, dtype) -> None:
        self.bsz = bsz 
        self.seqlens = seqlens
        self.n_heads = n_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim
        self.softmax_scale = softmax_scale
        self.device = device 
        self.dtype = dtype 
        self.data_layout = data_layout
        self.with_projections = with_projections

        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
        self.mla_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
            workspace_buffer,
            backend="auto"
        )

        q_indptr = torch.tensor([b for b in range(bsz+1)], dtype=torch.int32, device=device)
        kv_lens = torch.tensor(seqlens, dtype=torch.int32, device=self.device)
        kv_indptr = torch.tensor([0] + np.cumsum(seqlens).tolist(), dtype=torch.int32, device=self.device)  
        kv_indices = torch.tensor([b for b in range(sum(seqlens))], dtype=torch.int32, device=self.device)

        self.mla_wrapper.plan(
            q_indptr,
            kv_indptr,
            kv_indices,
            kv_lens,
            n_heads,
            kv_lora_rank,
            qk_rope_head_dim,
            page_size=1,
            causal=False,  # causal
            sm_scale=softmax_scale,
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
        )

    def generate_input_data(self):
        if self.with_projections:
            q_nope = torch.randn((self.bsz, self.n_heads, self.qk_nope_head_dim), dtype=self.dtype, device=self.device) 
        else:
            q_nope = torch.randn((self.bsz, self.n_heads, self.kv_lora_rank), dtype=self.dtype, device=self.device) 
        q_rope = torch.randn((self.bsz, self.n_heads, self.qk_rope_head_dim), dtype=self.dtype, device=self.device) 
        kv = torch.randn(size=[self.bsz, self.kv_lora_rank], dtype=self.dtype, device=self.device)
        pe = torch.randn(size=[self.bsz, self.qk_rope_head_dim], dtype=self.dtype, device=self.device)
        kv_cache = torch.empty((sum(self.seqlens), 1, self.kv_lora_rank), dtype=self.dtype, device=self.device)
        pe_cache = torch.empty((sum(self.seqlens), 1, self.qk_rope_head_dim), dtype=self.dtype, device=self.device)
        if self.with_projections:
            wkv_b = 1e-2*torch.randn(size=[self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), self.kv_lora_rank], dtype=self.dtype, device=self.device)
        else:
            wkv_b = None
        wkv_b1, wkv_b2 = wkv_b.view(self.n_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank).split([self.qk_nope_head_dim, self.v_head_dim], dim=1)
        return q_nope, q_rope, kv, pe, kv_cache, pe_cache, wkv_b1, wkv_b2

    def perf(self, warm_up=25, n_repeat=100):
        for i in range(warm_up):
            q_nope, q_rope, kv, pe, kv_cache, pe_cache, wkv_b1, wkv_b2 = self.generate_input_data()
            self.run(q_nope, q_rope, kv, pe, kv_cache, pe_cache, wkv_b1, wkv_b2)

        start = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
        end = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]

        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device=self.device)

        elapsed  = []
        for i in range(n_repeat):
            q_nope, q_rope, kv, pe, kv_cache, pe_cache, wkv_b1, wkv_b2 = self.generate_input_data()

            cache.zero_()

            start[i].record()
            self.run(q_nope, q_rope, kv, pe, kv_cache, pe_cache, wkv_b1, wkv_b2)
            end[i].record()      

        torch.cuda.synchronize()
        elapsed = [start[i].elapsed_time(end[i]) for i in range(n_repeat)]

        return np.median(elapsed)
    
    def run(self, q_nope, q_rope, kv, pe, kv_cache, pe_cache, wkv_b1, wkv_b2):
        if self.with_projections:
            q_nope = torch.einsum("bhd,hdc->bhc", q_nope, wkv_b1)
        
        out = self.mla_wrapper.run(q_nope, q_rope, kv_cache, pe_cache, return_lse=False)
    
        if self.with_projections:
            out = torch.einsum("bhc,hdc->bhd", out, wkv_b2)   

        return out


def benchmark():
    n_heads = 128
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim # 192
    v_head_dim = 128
    softmax_scale = 1 / math.sqrt(qk_head_dim)

    with_projections = True

    elapsed = []
    bszs = [1,2,4,8,16,32,64,128,256,512,1024]
    for bsz in bszs:
        seqlens = [4096+128]*bsz
        mla_flashinfer_test = MLAAbsorbFlashInferTest(bsz, seqlens, n_heads, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, v_head_dim, with_projections, softmax_scale, data_layout="NHD", device=device, dtype=dtype)
        elapsed.append(mla_flashinfer_test.perf())

        print("bsz: {:<5} elapsed: {:.2f} ms".format(bsz, elapsed[-1]))

    print(", ".join(["{:.3f}".format(float(e)) for e in elapsed]))

    plt.plot(bszs, elapsed)
    plt.savefig("flashinfer.png")

if __name__=="__main__":
    dtype = torch.float16
    device = torch.device("cuda")
    torch.manual_seed(0)

    benchmark()
