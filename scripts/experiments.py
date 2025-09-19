

import math
import torch 
import json 
import numpy as np 
import argparse
import sys

import logging
from src.helpers import init_logger

np.random.seed(0)

def form_batch(active_batch, system_prompt_len):
    seqlens_tree = [system_prompt_len] 
    for ind in active_batch:
        prompt_len = active_batch[ind]["prompt_len"]
        decode_pos = active_batch[ind]["pos"]
        seqlens_tree += [prompt_len + decode_pos]

    seqlens_dense = [seqlens_tree[0] + s for s in seqlens_tree[1:]]

    return seqlens_tree, seqlens_dense



def calc_num_ops(seqlens_dense, n_heads, kv_lora_rank, qk_rope_head_dim):
    n_ops = sum(seqlens_dense) * n_heads * (kv_lora_rank + qk_rope_head_dim) 
    n_ops += sum(seqlens_dense) * n_heads * kv_lora_rank
    n_ops = n_ops * 2
    return n_ops # in flops (not mac)


system_prompts ={
    "promptA": "anthropic/claude-4-opus",
    "promptB": "openai/o3",
    "promptC": "grok/personas"
}

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bsz", type=int, default=1024, help="Batch size")
    parser.add_argument("--system_prompt", type=str, default="promptA", help="System prompt")
    parser.add_argument("--model", type=str, choices=["deepseekv3", "kimik2"], default="deepseekv3", help="Model to benchmark")
    parser.add_argument("--dataset", type=str, choices=["mmlu", "gsm8k", "simpleqa"], default="mmlu", help="Dataset to benchmark on")
    parser.add_argument("--kernel", type=str, choices=["typhoonmla", "flashinfer", "flashmla"], default="typhoonmla", help="Kernel to use for the benchmark")
    args = parser.parse_args()

    logname = 'logs/' + "_".join([str(v) for v in [args.kernel, args.model, args.dataset, args.system_prompt, args.bsz]]) + '.out'
    init_logger(logging.INFO, name=logname)

    logging.info("python {}".format(" ".join(sys.argv)))

    if args.kernel == "typhoonmla":
        from typhoon_mla_wrapper import TyphoonMLATest
    elif args.kernel == "flashinfer":        
        from baselines.flashinfer_base import MLAAbsorbFlashInferTest
    elif args.kernel == "flashmla":
        from baselines.flashmla_base import FlashMLATest

    dtype = torch.float16
    device = torch.device("cuda")
    torch.manual_seed(0)

    THRESHOLD = 128
    bsz = 1024


    with open("data/archs/" + args.model + ".json", "r") as f:
        arch = json.load(f)

    n_heads = arch["num_attention_heads"]
    qk_nope_head_dim = arch["qk_nope_head_dim"]
    qk_rope_head_dim = arch["qk_rope_head_dim"]
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    kv_lora_rank = arch["kv_lora_rank"]
    v_head_dim = arch["v_head_dim"]
    softmax_scale = 1 / math.sqrt(qk_head_dim)

    with open("data/datasets/system_prompt_len.json", "r") as f:
        system_prompt_lens = json.load(f)
    system_prompt_len = system_prompt_lens[system_prompts[args.system_prompt]]

    dataset_path = "data/datasets/" + args.model + "_" + args.dataset + ".json"
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    n_samples = len(dataset["prompt_tokens"])

    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    indices = indices.tolist()

    next_ind = 0
    active_batch = {ind: {"pos": 0, "prompt_len": dataset["prompt_tokens"][ind], "decode_stop_len": dataset["generated_tokens"][ind]} for ind in indices[next_ind:bsz]}
    next_ind += len(active_batch)

    n_completed = 0
    total_n_ops = 0
    total_time = 0
    it = 0
    while len(active_batch) > 0:
        seqlens_tree, seqlens_dense = form_batch(active_batch, system_prompt_len)

        if args.kernel == "typhoonmla":
            is_stage1_absorb = bsz < THRESHOLD 
            run_in_single_stage = bsz < THRESHOLD 

            mla_flashinfer_test = TyphoonMLATest(bsz, seqlens_tree, n_heads, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, v_head_dim, is_stage1_absorb, run_in_single_stage, softmax_scale, data_layout="NHD", device=device, dtype=dtype)
            m_elapsed = mla_flashinfer_test.perf(warm_up=1, n_repeat=1)
        elif args.kernel == "flashinfer":
            mla_flashinfer_test = MLAAbsorbFlashInferTest(bsz, seqlens_dense, n_heads, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, v_head_dim, with_projections=True, softmax_scale=softmax_scale, data_layout="NHD", device=device, dtype=dtype)
            m_elapsed = mla_flashinfer_test.perf(warm_up=1, n_repeat=1)
        elif args.kernel == "flashmla":
            flash_mla_test = FlashMLATest(bsz, seqlens_dense, n_heads, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, v_head_dim, with_projections=True, device=device, dtype=dtype)
            m_elapsed, _, _ = flash_mla_test.perf(warm_up=1, n_repeat=1)
        else:
            raise NotImplementedError("Kernel {} is not implemented.".format(args.kernel))

        total_time += m_elapsed

        n_ops = calc_num_ops(seqlens_dense, n_heads, kv_lora_rank, qk_rope_head_dim)
        total_n_ops += n_ops

        logging.info("it: {}\t n_ops: {:.3f} TFLOPs\t elapsed: {:.3f} ms\t throughput: {:.3f} TFLOPs/s".format(it, n_ops*1e-12, m_elapsed, (total_n_ops*1e-12)/(total_time*1e-3)))

        for ind in list(active_batch.keys()):
            if active_batch[ind]["pos"] >= active_batch[ind]["decode_stop_len"]:
                del active_batch[ind]
                logging.info("{} finished".format(ind))
                n_completed += 1
                continue

            active_batch[ind]["pos"] += 1

        while len(active_batch) < bsz:
            if next_ind >= n_samples:
                break
            
            active_batch[indices[next_ind]] = {"pos": 0, "prompt_len": dataset["prompt_tokens"][indices[next_ind]], "decode_stop_len": dataset["generated_tokens"][indices[next_ind]]}
            logging.info("adding {} to active batch.".format(indices[next_ind]))
            next_ind += 1

        if len(active_batch) < bsz:
            break

        it += 1

    logging.info("num. of completed samples: {}\t total num ops: {:.3f} TFLOPs\t total time: {:.3f} ms\t throughput: {:.3f} TFLOPs/s".format(n_completed, total_n_ops*1e-12, total_time, (total_n_ops*1e-12)/(total_time*1e-3)))