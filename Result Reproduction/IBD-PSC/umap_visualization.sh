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
datasets=("gtsrb")                              # Add more datasets if needed
attacks=("badnet" "blend" "trojan" "adaptive_blend" "adaptive_patch")  # List of attack types
poison_rates=(0.05 0.1)                         # Poison rates to loop through

# Iterate over each combination of dataset, attack type, and poison rate
for dataset in "${datasets[@]}"; do
    for attack in "${attacks[@]}"; do
        for poison_rate in "${poison_rates[@]}"; do
            echo "============================================"
            echo "Dataset: $dataset, Attack: $attack, Poison Rate: $poison_rate"
            echo "============================================"

            # Step 1: Create a poisoned training set
            echo "Creating poisoned training set..."
            python create_poisoned_set.py -dataset="$dataset" -poison_type="$attack" -poison_rate="$poison_rate"
            if [ $? -ne 0 ]; then
                echo "Error in creating poisoned training set for Dataset: $dataset, Attack: $attack, Poison Rate: $poison_rate"
                continue
            fi

            # Step 2: Train on the poisoned training set
            echo "Training on poisoned training set..."
            python train_on_poisoned_set.py -dataset="$dataset" -poison_type="$attack" -poison_rate="$poison_rate"
            if [ $? -ne 0 ]; then
                echo "Error in training model for Dataset: $dataset, Attack: $attack, Poison Rate: $poison_rate"
                continue
            fi

            # Step 3: Test the backdoor model
            echo "Testing the backdoor model..."
            python test_model.py -dataset="$dataset" -poison_type="$attack" -poison_rate="$poison_rate"
            if [ $? -ne 0 ]; then
                echo "Error in testing model for Dataset: $dataset, Attack: $attack, Poison Rate: $poison_rate"
                continue
            fi

            # Step 4: Visualize the model's latent space
            echo "Visualizing the model's latent space..."
            python resnet18_layer_visualize.py -dataset="$dataset" -poison_type="$attack" -poison_rate="$poison_rate" -n_neighbors=5
            if [ $? -ne 0 ]; then
                echo "Error in visualizing latent space for Dataset: $dataset, Attack: $attack, Poison Rate: $poison_rate"
                continue
            fi

            echo "Visualization completed for Dataset: $dataset, Attack: $attack, Poison Rate: $poison_rate"
            echo ""
        done
    done
done

echo "All tasks completed successfully."
