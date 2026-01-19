#!/bin/bash

#SBATCH --job-name=train_cfm
#SBATCH --output=../logs/train_cfm.log
#SBATCH --error=../logs/train_cfm.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=0-02:00:00


ENV_NAME="sc_flow_density"
echo "Using environment: $ENV_NAME"

WDIR=/home/icb/lorenzo.consoli/repos/scFM_density_estimation/notebooks/cElegans/scripts
cd "$WDIR" || exit 1
echo "Current Working Directory: $WDIR"

export HYDRA_FULL_ERROR=1

conda run --no-capture-output -n "$ENV_NAME" python train_cfm.py
