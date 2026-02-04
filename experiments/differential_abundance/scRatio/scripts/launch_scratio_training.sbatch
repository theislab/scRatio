#!/bin/bash

#SBATCH --job-name=train_scratio_sweep
#SBATCH --output=./logs/train_scratio_sweep.out
#SBATCH --error=./logs/train_scratio_sweep.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH --nice=0
#SBATCH --time=0-10:00:00
#SBATCH --exclude=supergpu02,supergpu03,gpusrv45,gpusrv36

## hydra error logging
export HYDRA_FULL_ERROR=1;

cd ..

python train_scRatio.py --multirun +hparams_search=default +launcher=slurm