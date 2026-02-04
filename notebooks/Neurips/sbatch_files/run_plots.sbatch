#!/bin/bash

#SBATCH --job-name=run_plots
#SBATCH --output=../logs/run_plots_%j.log   # %j adds the SLURM job ID
#SBATCH --error=../logs/run_plots_%j.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=0-12:00:00

# Accept the first argument as SCVI model directory
cfm_model_dir="$1"

if [ -z "$cfm_model_dir" ]; then
    echo "ERROR: No CFM model directory provided!"
    exit 1
fi

# Conda environment
ENV_NAME="sc_flow_density"
echo "Using environment: $ENV_NAME"

WDIR=/home/icb/lorenzo.consoli/repos/scFM_density_estimation/notebooks/Neurips/scripts
cd "$WDIR" || exit 1
echo "Current Working Directory: $WDIR"

export HYDRA_FULL_ERROR=1

echo $cfm_model_dir

# Run Python script with config override
conda run --no-capture-output -n "$ENV_NAME" \
    python generate_plots.py paths.cfm_model_dir="$cfm_model_dir"
