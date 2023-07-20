#!/bin/bash -l
#COBALT -q full-node
#COBALT -t 720
#COBALT -n 1
#COBALT -A MultiActiveAI

module load conda/2023-01-11
conda activate

python src/rl_main.py --supervision --gpu=0 --n_gpus=8 --wandb