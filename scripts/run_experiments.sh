
CONDA_DIR=$(conda info --base)
source $CONDA_DIR/etc/profile.d/conda.sh
conda activate mla

bsz=1024
system_prompts=("promptA" "promptB" "promptC")
models=("deepseekv3" "kimik2")
datasets=("mmlu" "gsm8k" "simpleqa")

for system_prompt in "${system_prompts[@]}"; do
  for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
      echo "Running experiments with system prompt: $system_prompt, model: $model, dataset: $dataset"
      python scripts/experiments.py --bsz=$bsz --system_prompt $system_prompt --model $model --dataset $dataset --kernel treemla
    done
  done
done

for system_prompt in "${system_prompts[@]}"; do
  for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
      echo "Running experiments with system prompt: $system_prompt, model: $model, dataset: $dataset"
      python scripts/experiments.py --bsz=$bsz --system_prompt $system_prompt --model $model --dataset $dataset --kernel flashinfer
    done
  done
done

for system_prompt in "${system_prompts[@]}"; do
  for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
      echo "Running experiments with system prompt: $system_prompt, model: $model, dataset: $dataset"
      python scripts/experiments.py --bsz=$bsz --system_prompt $system_prompt --model $model --dataset $dataset --kernel flashmla
    done
  done
done

echo "Experiments completed at time: $(date +%Y-%m-%d_%H:%M:%S)"