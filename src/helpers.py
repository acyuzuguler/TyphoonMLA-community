
import torch
import numpy as np
import math
import logging 
import sys 

def profile(size, func, device, dtype=torch.float16):
    warm_up = 25
    n_repeat = 100

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device=device)

    elapsed  = []
    for i in range(warm_up+n_repeat):
        input_tensors = []
        for s in size:
            input_tensors.append(torch.rand(s, device=device, dtype=dtype))

        cache.zero_()

        start.record()
        func(*input_tensors)
        end.record()

        torch.cuda.synchronize()
        if i > warm_up:
            elapsed.append(start.elapsed_time(end))

    return np.median(elapsed)


def convert_absorb_to_naive(kv_cache, pe_cache, wkv_b1, wkv_b2):
    _, _, kv_lora_rank = kv_cache.shape
    _, _, qk_rope_head_dim = pe_cache.shape 
    n_heads, qk_nope_head_dim, kv_lora_rank = wkv_b1.shape
    n_heads, v_head_dim, kv_lora_rank = wkv_b2.shape

    wkv_b1 = wkv_b1.reshape(n_heads*qk_nope_head_dim, kv_lora_rank)
    wkv_b2 = wkv_b2.reshape(n_heads*v_head_dim, kv_lora_rank)

    k_nope = torch.matmul(kv_cache, wkv_b1.T)
    k_nope = k_nope.view(-1, n_heads, qk_nope_head_dim)

    pe_cache = pe_cache.expand(-1, n_heads, -1)
    dense_k_cache = torch.cat([k_nope, pe_cache], dim=-1)

    dense_v_cache = torch.matmul(kv_cache, wkv_b2.T)
    dense_v_cache = dense_v_cache.view(-1, n_heads, v_head_dim)

    return dense_k_cache, dense_v_cache

def merge_kv_cache(kv_cache_stage1, pe_cache_stage1, kv_cache_stage2, pe_cache_stage2, bsz):
    kv_cache_stage1_expanded = kv_cache_stage1.repeat(bsz, 1, 1)
    pe_cache_stage1_expanded = pe_cache_stage1.repeat(bsz, 1, 1)
    kv_cache_single_stage = torch.cat([kv_cache_stage1_expanded, kv_cache_stage2], dim=0)
    pe_cache_single_stage = torch.cat([pe_cache_stage1_expanded, pe_cache_stage2], dim=0)
    return kv_cache_single_stage, pe_cache_single_stage

def init_logger(level, name='log.out'):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=name,
        level=level,
        filemode="w",
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s'
    )
    stdout_handler = logging.StreamHandler(
        stream=sys.stdout,
    )
    logging.getLogger().addHandler(stdout_handler)