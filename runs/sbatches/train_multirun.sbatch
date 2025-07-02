#!/bin/bash
#SBATCH --job-name=train_scFM
#SBATCH --output=/home/icb/egor.antipov/scFM_density_estimation/runs/sbatches/outputs/train_scFM_output_%A.txt
#SBATCH --error=/home/icb/egor.antipov/scFM_density_estimation/runs/sbatches/outputs/train_scFM_error_%A.txt
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=100Gb
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nice=0

export HYDRA_FULL_ERROR=1

cd /home/icb/egor.antipov/scFM_density_estimation/

source ../miniconda3/bin/activate
conda activate scaling-transformers

python ./runs/scripts/train.py \
    --multirun \
    model.hidden_dims=[512,512,512],[512,512,512,512],[1024,1024,1024],[1024,1024,1024,1024],[2048,2048,2048],[2048,2048,2048,2048] \
    datamodule.num_pcs=20,40,60,80,100 \
    +launcher=gpu_normal
