#!/bin/bash
#SBATCH --job-name=tables_scFM
#SBATCH --output=/home/icb/egor.antipov/scFM_density_estimation/runs/sbatches/outputs/tables_output_%A.txt
#SBATCH --error=/home/icb/egor.antipov/scFM_density_estimation/runs/sbatches/outputs/tables_error_%A.txt
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

python ./runs/scripts/make_comparison_icml_add.py \
    --multirun \
    +loc=1,2,4 \
    +num_dims=2,5,10,20,30 \
    +start_from_scratch=true \
    +launcher=super_gpu_normal

# python ./runs/scripts/make_comparison_icml_with_checkpoints.py \
#     +loc=8 \
#     +num_dims=10 \
#     +start_from_scratch=false


##SBATCH --time=24:00:00
##SBATCH --mem=400Gb
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=10
##SBATCH --partition=gpu_p
##SBATCH --qos=gpu_normal
##SBATCH --constraint=h100_80gb
##SBATCH --nice=0

##SBATCH --time=1:00:00
##SBATCH --mem=30Gb
##SBATCH --cpus-per-task=10
##SBATCH --partition=cpu_p
##SBATCH --qos=cpu_normal
##SBATCH --nice=0