#!/bin/bash

CONDA_DIR=$(conda info --base)
source $CONDA_DIR/etc/profile.d/conda.sh
conda activate mla

# run all experiments. results are stored at logs/
bash scripts/run_experiments.sh

# plot results. output image: plots/main_exp.png
python plots/plot.py