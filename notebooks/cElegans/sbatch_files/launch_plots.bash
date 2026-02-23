#!/bin/bash

# Directory containing subdirectories
PARENT_DIR="../../../results/neurips/cfm_runs"

# Iterate over each subdirectory
for dir in "$PARENT_DIR"/*; do
    # Make sure it is a directory
    [ -d "$dir" ] || continue

    # Get the directory name (for tmux session)
    dir_name=$(basename "$dir")

    # Start a detached tmux session running the script with full path
    sbatch ../sbatch_files/run_plots.sbatch $dir;
done
