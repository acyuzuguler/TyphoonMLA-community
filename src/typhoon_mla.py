
import numpy as np
import math
import torch 
import flashinfer

from typing import List

class TyphoonMLA:
    def __init__(self, is_stage1_absorb, run_in_single_stage, device, dtype):
        self.is_stage1_absorb = is_stage1_absorb
        self.run_in_single_stage = run_in_single_stage
        self.device = device
        self.dtype = dtype

        if self.run_in_single_stage: # execute in single stage
            self.single_stage_absorb_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
                float_workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=self.device),
                backend="auto"
            )
        else: # execute in two stages
            # stage 1
            if self.is_stage1_absorb:
                self.stage1_absorb_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
                    float_workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=self.device),
                    backend="auto"
                )
            else:
                self.stage1_naive_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                    float_workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device), 
                    kv_layout="NHD", 
                    backend="fa2"
                ) 

            # stage 2 is always absorb
            self.stage2_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
                float_workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=self.device),
                backend="auto"
            )

    def plan(self, bsz: int, seqlens: List[int], n_heads: int, qk_nope_head_dim: int, qk_rope_head_dim: int, kv_lora_rank: int, v_head_dim: int):
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        softmax_scale = 1 / math.sqrt(qk_head_dim)

        seqlen_stage1 = seqlens[0]
        seqlen_stage2 = seqlens[1]
        
        # for small batch sizes, running it in a single stage is faster because of launch overhead
        if self.run_in_single_stage:
            q_indptr = torch.tensor([b for b in range(bsz+1)], dtype=torch.int32, device=self.device)
            kv_lens = torch.tensor([seqlen_stage1 + seqlen_stage2 for b in range(bsz)], dtype=torch.int32, device=self.device)
            kv_indptr = torch.tensor([b*(seqlen_stage1 + seqlen_stage2) for b in range(bsz+1)], dtype=torch.int32, device=self.device)  
            kv_indices = torch.tensor([b for b in range(bsz*(seqlen_stage1 + seqlen_stage2))], dtype=torch.int32, device=self.device)

            self.single_stage_absorb_wrapper.plan(
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
        else:
            # stage 1
            if seqlen_stage1 > 0:
                if self.is_stage1_absorb:
                    q_indptr = torch.tensor([0, bsz], dtype=torch.int32, device=self.device)
                    kv_lens = torch.tensor([seqlen_stage1,], dtype=torch.int32, device=self.device)
                    kv_indptr = torch.tensor([0, seqlen_stage1], dtype=torch.int32, device=self.device)
                    kv_indices = torch.tensor(list(range(bsz*seqlen_stage1)), dtype=torch.int32, device=self.device)

                    self.stage1_absorb_wrapper.plan(
                        q_indptr,
                        kv_indptr,
                        kv_indices,
                        kv_lens,
                        n_heads,
                        kv_lora_rank,
                        qk_rope_head_dim,
                        page_size=1,
                        causal=False,
                        sm_scale=softmax_scale,
                        q_data_type=self.dtype,
                        kv_data_type=self.dtype,
                    )
                else:
                    qo_ind = [0, bsz]
                    kv_ind = [0, seqlen_stage1]

                    qo_indptr = torch.tensor(
                        qo_ind, dtype=torch.int32, device=self.device
                    )
                    kv_indptr = torch.tensor(
                        kv_ind, dtype=torch.int32, device=self.device
                    )

                    self.stage1_naive_wrapper.plan(
                        qo_indptr=qo_indptr,
                        kv_indptr=kv_indptr,
                        num_qo_heads=n_heads,
                        num_kv_heads=n_heads,
                        head_dim_qk=qk_head_dim,
                        head_dim_vo=v_head_dim,
                        causal=False,
                        q_data_type=self.dtype,
                        sm_scale=softmax_scale
                    )

            # stage 2
            if seqlen_stage2 > 0:
                q_indptr = torch.tensor([b for b in range(bsz+1)], dtype=torch.int32, device=self.device)
                kv_lens = torch.tensor(seqlens[1:], dtype=torch.int32, device=self.device)
                kv_indptr = torch.tensor([0] + np.cumsum(seqlens[1:]).tolist(), dtype=torch.int32, device=self.device)  
                kv_indices = torch.tensor([b for b in range(sum(seqlens[1:]))], dtype=torch.int32, device=self.device)

                self.stage2_wrapper.plan(
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

    def run(self, q, kv_cache_stage1, pe_cache_stage1, kv_cache_stage2, pe_cache_stage2, k_cache_stage1, v_cache_stage1, wkv_b1, wkv_b2):   
        if self.run_in_single_stage:
            return self._run_full_absorb(q, kv_cache_stage1, pe_cache_stage1, wkv_b1, wkv_b2)
        else:
            return self._run_in_2stage(q, kv_cache_stage1, pe_cache_stage1, kv_cache_stage2, pe_cache_stage2, k_cache_stage1, v_cache_stage1, wkv_b1, wkv_b2)

    def _run_full_absorb(self, q, kv_cache_stage1, pe_cache_stage1, wkv_b1, wkv_b2):
        bsz, n_heads, qk_head_dim = q.shape
        bszxseqlen_stage1, _, kv_lora_rank = kv_cache_stage1.shape
        bszxseqlen_stage1, _, qk_rope_head_dim = pe_cache_stage1.shape
        qk_nope_head_dim = qk_head_dim - qk_rope_head_dim

        q_nope, q_rope = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
        q_nope = torch.einsum("bhd,hdc->bhc", q_nope, wkv_b1)

        out = self.single_stage_absorb_wrapper.run(q_nope, q_rope, kv_cache_stage1, pe_cache_stage1, return_lse=False)

        out = torch.einsum("bhc,hdc->bhd", out, wkv_b2)
        return out 

    def _run_in_2stage(self, q, kv_cache_stage1, pe_cache_stage1, kv_cache_stage2, pe_cache_stage2, k_cache_stage1, v_cache_stage1, wkv_b1, wkv_b2):
        bsz, n_heads, qk_head_dim = q.shape
        seqlen_stage1, _, kv_lora_rank = kv_cache_stage1.shape
        seqlen_stage1, _, qk_rope_head_dim = pe_cache_stage1.shape
        seqlen_stage2_sum, _, kv_lora_rank = kv_cache_stage2.shape
        seqlen_stage2_sum, _, qk_rope_head_dim = pe_cache_stage2.shape
        qk_nope_head_dim = qk_head_dim - qk_rope_head_dim

        # stage 1
        if seqlen_stage1 > 0:
            if self.is_stage1_absorb:
                q_nope, q_rope = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
                q_nope = torch.einsum("bhd,hdc->bhc", q_nope, wkv_b1)
                out1, lse1 = self.stage1_absorb_wrapper.run(q_nope, q_rope, kv_cache_stage1, pe_cache_stage1, return_lse=True)
                out1 = torch.einsum("bhc,hdc->bhd", out1, wkv_b2)
            else:
                out1, lse1 = self.stage1_naive_wrapper.run(q, k_cache_stage1, v_cache_stage1, return_lse=True)

        # stage 2
        if seqlen_stage2_sum > 0:
            if seqlen_stage1 == 0 or not self.is_stage1_absorb:
                q_nope, q_rope = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
                q_nope = torch.einsum("bhd,hdc->bhc", q_nope, wkv_b1)

            out2, lse2 = self.stage2_wrapper.run(q_nope, q_rope, kv_cache_stage2, pe_cache_stage2, return_lse=True)
            out2 = torch.einsum("bhc,hdc->bhd", out2, wkv_b2)
            if seqlen_stage1 == 0:
                return out2

            out = self.combine_lse(out1, lse1, out2, lse2)
            return out
        else:
            return out1

    def combine_lse(self, out1: torch.Tensor, lse1: torch.Tensor, out2: torch.Tensor, lse2: torch.Tensor):
        """
        Combines two attention results using their LSEs.

        Out1/2 shape: [batch, seq_len, qheads, hdim]
        lse1/2 shape: [batch, seq_len, qheads]
        """
        max_lse = torch.maximum(lse1, lse2)

        adjust_factor1 = (lse1 - max_lse).exp()
        adjust_factor2 = (lse2 - max_lse).exp()

        new_denominator = adjust_factor1 + adjust_factor2

        aggregated = (
            out1 * adjust_factor1.unsqueeze(-1) + out2 * adjust_factor2.unsqueeze(-1)
        ) / new_denominator.unsqueeze(-1)

        return aggregated.to(out1.dtype)
