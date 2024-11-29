#!/bin/bash

# Submit all scripts in the gtsrb_scripts directory
for script in gtsrb_scripts/*.sh; do
    echo "Submitting $script..."
    sbatch "$script"
done

echo "All GTSRB scripts submitted."

