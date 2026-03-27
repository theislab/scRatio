#!/bin/bash

#SBATCH --job-name=train_clf_mi
#SBATCH --output=./logs/train_clf_mi.out
#SBATCH --error=./logs/train_clf_mi.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH --nice=0
#SBATCH --time=0-10:00:00

## hydra error logging
export HYDRA_FULL_ERROR=1;

cd ..

python train_clf_mi.py --multirun +hparams_search=mi +launcher=slurm
