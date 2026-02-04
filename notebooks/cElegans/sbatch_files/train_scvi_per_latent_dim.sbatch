#!/bin/bash

#SBATCH --job-name=train_scvi_per_latent_dim
#SBATCH --output=../logs/train_scvi_per_latent_dim.log
#SBATCH --error=../logs/train_scvi_per_latent_dim.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=20G
#SBATCH --time=0-10:00:00


ENV_NAME="sc_flow_density"
echo "Using environment: $ENV_NAME"

WDIR=/home/icb/lorenzo.consoli/repos/scFM_density_estimation/notebooks/cElegans/scripts
cd "$WDIR" || exit 1
echo "Current Working Directory: $WDIR"

export HYDRA_FULL_ERROR=1

conda run --no-capture-output -n "$ENV_NAME" python train_scvi.py \
    --multirun \
    +sweep_latent_dim=default_grid \
    +launcher=slurm
