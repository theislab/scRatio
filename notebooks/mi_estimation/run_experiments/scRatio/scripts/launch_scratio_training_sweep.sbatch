#!/bin/bash

#SBATCH --job-name=train_scratio_sweep_uncond
#SBATCH --output=./logs/train_scratio_sweep_uncond.out
#SBATCH --error=./logs/train_scratio_sweep_uncond.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH --nice=0
#SBATCH --time=0-10:00:00

## hydra error logging
export HYDRA_FULL_ERROR=1;

# cd ..

export PYTHONPATH="../../../../runs/differential_abundance/:$PYTHONPATH"

python train_uncond.py --multirun paths=sweeps +hparams_search=default +launcher=slurm

# python train_scRatio.py --multirun paths=sweeps +hparams_search=default +launcher=slurm