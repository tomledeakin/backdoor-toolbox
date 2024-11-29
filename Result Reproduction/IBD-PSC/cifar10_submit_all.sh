#!/bin/bash

# Submit all scripts in the cifar10_scripts directory
for script in cifar10_scripts/*.sh; do
    echo "Submitting $script..."
    sbatch "$script"
done

echo "All scripts submitted."

