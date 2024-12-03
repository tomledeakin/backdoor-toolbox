#!/bin/bash
#SBATCH --job-name=layer_umap_visualize         # Job name
#SBATCH --output=layer_umap_visualize_output.log # Standard output log
#SBATCH --error=layer_umap_visualize_error.log   # Standard error log
#SBATCH --partition=gpu                         # Use GPU partition
#SBATCH --gres=gpu:a100:1                       # Request 1 A100 GPU
#SBATCH --time=48:00:00                         # Time limit (48 hours)
#SBATCH --mail-user=tomledeakin@gmail.com       # Email for notifications
#SBATCH --mail-type=END,FAIL                    # Notify on job completion or failure

# Navigate to the project directory
cd "$HOME/BackdoorBox Research/backdoor-toolbox" || { echo "Directory not found"; exit 1; }

# Activate the Python virtual environment
source "my_env/bin/activate"

# Define arrays for datasets, attack types, and poison rates
datasets=("cifar10" "gtsrb")                              # Add more datasets if needed
attacks=("badnet" "blend" "trojan" "adaptive_blend" "adaptive_patch")  # List of attack types
poison_rates=(0.05 0.1)                         # Poison rates to loop through

# Iterate over each combination of dataset, attack type, and poison rate
for dataset in "${datasets[@]}"; do
    for attack in "${attacks[@]}"; do
        for poison_rate in "${poison_rates[@]}"; do
            echo "============================================"
            echo "Dataset: $dataset, Attack: $attack, Poison Rate: $poison_rate"
            echo "============================================"

            # Visualize the model's latent space (Step 4)
            echo "Visualizing the model's latent space..."
            python test_model.py -dataset="$dataset" -poison_type="$attack" -poison_rate="$poison_rate"
            if [ $? -ne 0 ]; then
                echo "Error in visualizing latent space for Dataset: $dataset, Attack: $attack, Poison Rate: $poison_rate"
                continue
            fi

            echo "Visualization completed for Dataset: $dataset, Attack: $attack, Poison Rate: $poison_rate"
            echo ""
        done
    done
done

echo "All visualization tasks completed successfully."
