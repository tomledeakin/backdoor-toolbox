#!/bin/bash
#SBATCH --job-name=backdoor_experiments                               # Job name
#SBATCH --output=main_output.log                                      # General output file for the job
#SBATCH --error=main_error.log                                        # General error file for the job
#SBATCH --partition=gpu                                               # Use GPU partition
#SBATCH --gres=gpu:a100:1                                             # Request 1 A100 GPU
#SBATCH --time=06:00:00                                               # Maximum runtime

# Move to the directory containing the environment
cd "$HOME/BackdoorBox Research/backdoor-toolbox"

# Activate the virtual environment
source "my_env/bin/activate"

# Create directories to store results if they do not exist
mkdir -p ibd_psc_output ibd_psc_error ibd_psc_visualizations

# Define datasets, attack types, and visualization methods
datasets=("cifar10" "gtsrb" "imagenet")
attacks=("badnet" "blend" "trojan" "clean_label" "dynamic" "SIG" "ISSBA" "WaNet" "refool" "TaCT" "adaptive_blend" "adaptive_patch" "adaptive_k_way" "badnet_all_to_all" "SleeperAgent")
visualization_methods=("pca" "tsne" "oracle")

# Loop over each dataset and attack type
for dataset in "${datasets[@]}"; do
    for attack in "${attacks[@]}"; do
        # Define unique output and error filenames for each experiment
        output_file="ibd_psc_output/${dataset}_${attack}_output.log"
        error_file="ibd_psc_error/${dataset}_${attack}_error.log"

        echo "Running experiment for dataset: $dataset with attack: $attack"

        # Step 1: Create poisoned training set
        echo "Step 1: Creating poisoned training set with ${attack} attack on ${dataset}..."
        python create_poisoned_set.py -dataset=$dataset -poison_type=$attack -poison_rate=0.01 >> "$output_file" 2>> "$error_file"

        # Step 2: Train the model on the poisoned dataset
        echo "Step 2: Training the model on the poisoned dataset..."
        python train_on_poisoned_set.py -dataset=$dataset -poison_type=$attack -poison_rate=0.01 >> "$output_file" 2>> "$error_file"

        # Step 3: Test the backdoor model
        echo "Step 3: Testing the backdoor model..."
        python test_model.py -dataset=$dataset -poison_type=$attack -poison_rate=0.01 >> "$output_file" 2>> "$error_file"

        # Step 4: Visualize the model's latent space with each visualization method
        for method in "${visualization_methods[@]}"; do
            visualization_file="ibd_psc_visualizations/${dataset}_${attack}_${method}.png"
            echo "Step 4: Visualizing the model's latent space with ${method}..."
            python visualize.py -method=$method -dataset=$dataset -poison_type=$attack -poison_rate=0.01 >> "$output_file" 2>> "$error_file"
            
            # Move the visualization image to the storage directory
            mv "${method}_${dataset}_${attack}.png" "$visualization_file" || echo "Visualization (${method}) not found for ${dataset}_${attack}"
        done

        # Step 5: Apply IBD_PSC defense method
        echo "Step 5: Applying IBD_PSC defense method..."
        python other_defense.py -defense=IBD_PSC -dataset=$dataset -poison_type=$attack -poison_rate=0.01 >> "$output_file" 2>> "$error_file"

        echo "Experiment for $dataset with $attack completed."

        # Clear the output and error files for the next experiment
        cat "$output_file" > "$output_file.bak" && > "$output_file"
        cat "$error_file" > "$error_file.bak" && > "$error_file"
    done
done

