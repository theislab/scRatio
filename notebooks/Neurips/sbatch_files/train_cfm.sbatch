#!/bin/bash

#SBATCH --job-name=train_cfm
#SBATCH --output=../logs/train_cfm_%j.log   # %j adds the SLURM job ID
#SBATCH --error=../logs/train_cfm_%j.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=0-02:00:00

# Accept the first argument as SCVI model directory
scvi_model_dir="$1"

if [ -z "$scvi_model_dir" ]; then
    echo "ERROR: No SCVI model directory provided!"
    exit 1
fi

# Conda environment
ENV_NAME="sc_flow_density"
echo "Using environment: $ENV_NAME"

WDIR=../scripts
cd "$WDIR" || exit 1
echo "Current Working Directory: $WDIR"

export HYDRA_FULL_ERROR=1

echo $scvi_model_dir

# Run Python script with config override
conda run --no-capture-output -n "$ENV_NAME" \
    python train_cfm.py paths.scvi_model_dir="$scvi_model_dir"
