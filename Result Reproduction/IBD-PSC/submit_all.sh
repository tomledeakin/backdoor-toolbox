#!/bin/bash

# Submit all scripts in the "scripts" directory
for script in scripts/*.sh; do
    sbatch "$script"
done

echo "All jobs submitted."


