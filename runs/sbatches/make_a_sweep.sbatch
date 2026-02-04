#!/bin/bash
#SBATCH --job-name=tables_scFM
#SBATCH --output=/home/icb/egor.antipov/scFM_density_estimation/runs/sbatches/outputs/sweep_output_%A.txt
#SBATCH --error=/home/icb/egor.antipov/scFM_density_estimation/runs/sbatches/outputs/sweep_error_%A.txt
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=30Gb
#SBATCH --cpus-per-task=10
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nice=0

export HYDRA_FULL_ERROR=1

cd /home/icb/egor.antipov/scFM_density_estimation/

source ../miniconda3/bin/activate
conda activate scaling-transformers

python ./runs/scripts/make_a_sweep.py \
    --multirun \
    +num_dims=2,5,10,20,30,50 \
    +lr=0.001,0.0001 \
    +cond_latent_dim=32,64,128 \
    +start_from_scratch=false \
    +launcher=super_gpu_normal