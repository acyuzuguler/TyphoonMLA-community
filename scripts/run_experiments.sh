
conda activate mla

bsz=1024
system_prompts=("promptA" "promptB" "promptC")
models=("deepseekv3" "kimik2")
datasets=("mmlu" "simpleqa")

for system_prompt in "${system_prompts[@]}"; do
  for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
      echo "Running experiments with system prompt: $system_prompt, model: $model, dataset: $dataset"
      python scripts/experiments.py --bsz=$bsz --system_prompt $system_prompt --model $model --dataset $dataset --kernel flashinfer
    done
  done
done

conda activate flashmla

bsz=1024
system_prompts=("promptA" "promptB" "promptC")
models=("deepseekv3" "kimik2")
datasets=("mmlu" "simpleqa")

for system_prompt in "${system_prompts[@]}"; do
  for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
      echo "Running experiments with system prompt: $system_prompt, model: $model, dataset: $dataset"
      python scripts/experiments.py --bsz=$bsz --system_prompt $system_prompt --model $model --dataset $dataset --kernel flashmla
    done
  done
done

echo "Experiments completed at time: $(date +%Y-%m-%d_%H:%M:%S)"