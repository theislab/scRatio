#!/bin/bash
#SBATCH --job-name=train_scFM
#SBATCH --output=/home/icb/egor.antipov/scFM_density_estimation/runs/sbatches/outputs/train_scFM_output_%A.txt
#SBATCH --error=/home/icb/egor.antipov/scFM_density_estimation/runs/sbatches/outputs/train_scFM_error_%A.txt
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=400Gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
##SBATCH --constraint=h100_80gb
##SBATCH --constraint=a100_40gb
##SBATCH --constraint=v100_32gb
#SBATCH --nice=0

export HYDRA_FULL_ERROR=1

cd /home/icb/egor.antipov/scFM_density_estimation/

source ../miniconda3/bin/activate
conda activate scaling-transformers

python ./runs/scripts/train.py \
    model.hidden_dims=[1024,1024,1024,1024] \
    datamodule.num_pcs=50 \
    datamodule.conditions=["donor","cytokine"] \
    datamodule.num_conditions=2 \
    datamodule.condition_dims=[12,91] \
    datamodule.train_batch_size=4096 \
    datamodule.val_batch_size=30000 \
    datamodule.test_batch_size=60000 \
    datamodule.val_split=0.02 \
    datamodule.test_split=0.01 \
    training.save_checkpoint=false
