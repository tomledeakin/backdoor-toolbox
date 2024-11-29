#!/bin/bash

# Directory containing the generated SLURM scripts
script_dir="umap_scripts"

# Loop through each script in the directory and submit it
for script in "$script_dir"/*.sh; do
    echo "Submitting job for $script..."
    sbatch "$script"
done

echo "All jobs submitted."

