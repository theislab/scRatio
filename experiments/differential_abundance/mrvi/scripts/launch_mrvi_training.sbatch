#!/bin/bash

#SBATCH --job-name=train_mrvi_sweep
#SBATCH --output=./logs/train_mrvi_sweep.out
#SBATCH --error=./logs/train_mrvi_sweep.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH --nice=0
#SBATCH --time=0-10:00:00
#SBATCH --exclude=supergpu02,supergpu03

## hydra error logging
export HYDRA_FULL_ERROR=1;

python ../train_mrvi.py --multirun +hparams_search=default +launcher=slurm