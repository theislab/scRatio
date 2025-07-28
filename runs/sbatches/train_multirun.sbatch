#!/bin/bash
#SBATCH --job-name=train_scFM
#SBATCH --output=/home/icb/egor.antipov/scFM_density_estimation/runs/sbatches/outputs/train_scFM_output_%A.txt
#SBATCH --error=/home/icb/egor.antipov/scFM_density_estimation/runs/sbatches/outputs/train_scFM_error_%A.txt
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=300Gb
#SBATCH --cpus-per-task=20
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nice=0

export HYDRA_FULL_ERROR=1

cd /home/icb/egor.antipov/scFM_density_estimation/

source ../miniconda3/bin/activate
conda activate scaling-transformers

python ./runs/scripts/train.py \
    --multirun \
    model.hidden_dims=[1024,1024,1024,1024] \
    datamodule.num_pcs=60 \
    datamodule.conditions=["donor","cytokine"] \
    datamodule.index=1 \
    datamodule.probability=0 \
    datamodule.batch_size=8192,4096,2048,1024,512,256 \
    +launcher=gpu_normal
