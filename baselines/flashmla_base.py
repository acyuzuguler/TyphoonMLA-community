
import numpy as np
import torch

from flash_mla import flash_mla_with_kvcache, get_mla_metadata


class FlashMLATest:
    def __init__(self, bsz, seqlens, n_heads, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, v_head_dim, with_projections, device, dtype):
        self.bsz = bsz 
        self.seqlens = seqlens
        self.q_seqlen = 1
        self.n_heads = n_heads
        self.n_kv_heads = 1
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim
        self.with_projections = with_projections
        self.device = device 
        self.dtype = dtype 
        self.max_seqlen = 32*1024 
        assert np.max(seqlens) <= self.max_seqlen

    def generate_input_data(self):
        if self.with_projections:
            q_nope = torch.randn(size=[self.bsz, 1, self.n_heads, self.qk_nope_head_dim], dtype=self.dtype, device=self.device)
        else:
            q_nope = torch.randn(size=[self.bsz, 1, self.n_heads, self.kv_lora_rank], dtype=self.dtype, device=self.device)
        q_rope = torch.randn(size=[self.bsz, 1, self.n_heads, self.qk_rope_head_dim], dtype=self.dtype, device=self.device)
        queries = torch.concatenate([q_nope, q_rope], dim=-1)

        cache_seqlens = torch.full((self.bsz,), 0, dtype=torch.int32, device=self.device)
        for i in range(self.bsz):
            cache_seqlens[i] = self.seqlens[i]
                
        block_size = 128
        num_blocks = self.max_seqlen // block_size
        block_table = torch.arange(self.bsz*num_blocks, dtype=torch.int32, device=self.device).view(self.bsz, num_blocks)
        blocked_k = torch.randn(size=[self.bsz*num_blocks, block_size, self.n_kv_heads, self.kv_lora_rank+self.qk_rope_head_dim], dtype=self.dtype, device=self.device)
        for i in range(self.bsz):
            blocked_k.view(self.bsz, self.max_seqlen, self.n_kv_heads, self.kv_lora_rank+self.qk_rope_head_dim)[i, cache_seqlens[i].item():] = (
                float("nan")
            )

        tile_scheduler_metadata, num_splits = get_mla_metadata(
            cache_seqlens, self.n_heads // self.n_kv_heads, self.n_kv_heads
        )

        if self.with_projections:
            wkv_b = 1e-2*torch.randn(size=[self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), self.kv_lora_rank], dtype=self.dtype, device=self.device)
        else:
            wkv_b = None

        wkv_b1, wkv_b2 = wkv_b.view(self.n_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank).split([self.qk_nope_head_dim, self.v_head_dim], dim=1)
        return queries, blocked_k, block_table, cache_seqlens, tile_scheduler_metadata, num_splits, wkv_b1, wkv_b2

    def calc_num_ops(self):
        n_ops = sum(self.seqlens) * self.n_heads * (self.kv_lora_rank + self.qk_rope_head_dim) 
        n_ops += sum(self.seqlens) * self.n_heads * self.kv_lora_rank
        n_ops = n_ops * 2
        return n_ops # in flops (not mac)

    def calc_memsize(self):
        memsize = sum(self.seqlens) * self.n_kv_heads * (self.kv_lora_rank+self.qk_rope_head_dim) # kv-cache & pe-cache
        memsize += self.bsz * self.n_heads * (self.kv_lora_rank+self.qk_rope_head_dim) # queries
        memsize += self.bsz * self.n_heads * self.kv_lora_rank # output
        memsize = memsize * (torch.finfo(self.dtype).bits // 8)
        return memsize # in bytes
    
    def calc_cube_throughput(self, elapsed): # elapsed is in ms
        n_ops = self.calc_num_ops() # flops
        return n_ops / (elapsed * 1e-3) # return in flops/s

    def calc_hbm_throughput(self, elapsed): # elapsed is in ms
        memsize = self.calc_memsize() # bytes
        return memsize / (elapsed * 1e-3) # return in bytes/s

    def run(self, queries, blocked_k, block_table, cache_seqlens, tile_scheduler_metadata, num_splits, wkv_b1, wkv_b2):
        if self.with_projections:
            q_nope, q_rope = queries.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            q_nope = torch.einsum("bhd,hdc->bhc", q_nope.squeeze(1), wkv_b1).unsqueeze(1)
            queries = torch.concatenate([q_nope, q_rope], dim=-1)

        out_flash, lse_flash = flash_mla_with_kvcache(
                queries,
                blocked_k,
                block_table,
                cache_seqlens,
                self.kv_lora_rank,
                tile_scheduler_metadata,
                num_splits,
                causal=False,
            )

        if self.with_projections:
            out_flash = torch.einsum("bhc,hdc->bhd", out_flash.squeeze(1), wkv_b2).unsqueeze(1)

        return out_flash, lse_flash

    def perf(self, warm_up=25, n_repeat=100):
        for i in range(warm_up):
            queries, blocked_k, block_table, cache_seqlens, tile_scheduler_metadata, num_splits, wkv_b1, wkv_b2 = self.generate_input_data()
            self.run(queries, blocked_k, block_table, cache_seqlens, tile_scheduler_metadata, num_splits, wkv_b1, wkv_b2)

        start = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
        end = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]

        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device=self.device)

        elapsed  = []
        for i in range(n_repeat):
            queries, blocked_k, block_table, cache_seqlens, tile_scheduler_metadata, num_splits, wkv_b1, wkv_b2 = self.generate_input_data()

            cache.zero_()

            start[i].record()
            self.run(queries, blocked_k, block_table, cache_seqlens, tile_scheduler_metadata, num_splits, wkv_b1, wkv_b2)
            end[i].record()      
                          
        torch.cuda.synchronize()
        elapsed = [start[i].elapsed_time(end[i]) for i in range(n_repeat)]
        m_elapsed = np.median(elapsed)

        cube_throughput = self.calc_cube_throughput(m_elapsed)
        hbm_throughput = self.calc_hbm_throughput(m_elapsed)

        return m_elapsed, cube_throughput, hbm_throughput

def benchmark():
    dtype = torch.float16
    device = torch.device("cuda")
    torch.manual_seed(0)

    
    max_seqlen = 8192
    n_heads = 128
    n_kv_heads = 1
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    with_projections = True 

    bszs = [1,2,4,8,16,32,64,128,256,512,1024]
    elapseds = []
    for bsz in bszs:
        seqlens = [4096+128] * bsz

        flash_mla_test = FlashMLATest(bsz, seqlens, n_heads, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, v_head_dim, with_projections, device, dtype)
        m_elapsed, cube_throughput, hbm_throughput = flash_mla_test.perf()
        elapseds.append(m_elapsed)
        print("bsz: {:<5}\t elapsed: {:.2f} ms\t cube: {:.2f} TOPS/s\t hbm: {:.2f}TB/s".format(bsz, m_elapsed, cube_throughput*1e-12, hbm_throughput*1e-12))

    print(", ".join(["{:.3f}".format(float(e)) for e in elapseds]))

if __name__=="__main__":
    benchmark()
