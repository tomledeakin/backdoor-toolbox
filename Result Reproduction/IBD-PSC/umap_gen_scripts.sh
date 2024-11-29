#!/bin/bash

# Define datasets and attack types
datasets=("cifar10" "gtsrb")
attacks=("badnet" "blend" "trojan" "clean_label" "dynamic" "SIG" "ISSBA" "WaNet" "refool" "TaCT" "adaptive_blend" "adaptive_patch" "adaptive_k_way" "badnet_all_to_all" "SleeperAgent")

# Create the umap_scripts directory if it doesn't exist
mkdir -p umap_scripts

# Loop over each dataset and attack type
for dataset in "${datasets[@]}"; do
    for attack in "${attacks[@]}"; do
        # Define script filename inside umap_scripts directory
        script_file="umap_scripts/${dataset}_${attack}.sh"

        # Create the script file with output and error logs in the same directory
        cat <<EOT > "$script_file"
#!/bin/bash
#SBATCH --job-name=${dataset}_${attack}                         # Job name
#SBATCH --output=umap_scripts/${dataset}_${attack}_output.log   # Output file
#SBATCH --error=umap_scripts/${dataset}_${attack}_error.log     # Error file
#SBATCH --partition=gpu                                         # Use GPU partition
#SBATCH --gres=gpu:a100:1                                       # Request 1 A100 GPU
#SBATCH --time=06:00:00                                         # Max runtime

# Move to the directory containing the environment
cd "\$HOME/BackdoorBox Research/backdoor-toolbox"
source "my_env/bin/activate"

# Step 1: Create poisoned training set
echo "Creating poisoned training set for ${attack} on ${dataset}..."
python create_poisoned_set.py -dataset=$dataset -poison_type=$attack -poison_rate=0.01

# Step 2: Train the model on the poisoned dataset
echo "Training the model on the poisoned dataset..."
python train_on_poisoned_set.py -dataset=$dataset -poison_type=$attack -poison_rate=0.01

# Step 3: Test the backdoor model
echo "Testing the backdoor model..."
python test_model.py -dataset=$dataset -poison_type=$attack -poison_rate=0.01

# Step 4: Visualize the model's latent space with various methods
echo "Visualizing the model's latent space..."
methods=("umap")
for method in "\${methods[@]}"; do
    echo "Using method: \$method"
    python umap_visualize.py -method=\$method -dataset=$dataset -poison_type=$attack -poison_rate=0.01
done

echo "Experiment for $dataset with $attack completed."
EOT

        # Make the script executable
        chmod +x "$script_file"
    done
done

echo "Script generation completed."

