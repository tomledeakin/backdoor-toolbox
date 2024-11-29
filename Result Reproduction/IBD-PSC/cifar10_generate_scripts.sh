#!/bin/bash

# Define attack types for cifar10 dataset
attacks=("badnet" "blend" "trojan" "clean_label" "dynamic" "SIG" "ISSBA" "WaNet" "refool" "TaCT" "adaptive_blend" "adaptive_patch" "adaptive_k_way" "badnet_all_to_all" "SleeperAgent")

# Create the cifar10_scripts directory if it doesn't exist
mkdir -p cifar10_scripts

# Loop over each attack type for the cifar10 dataset
for attack in "${attacks[@]}"; do
    # Define script filename
    script_file="cifar10_scripts/cifar10_${attack}.sh"

    # Create the script file with output and error logs in the same directory
    cat <<EOT > "$script_file"
#!/bin/bash
#SBATCH --job-name=cifar10_${attack}                         # Job name
#SBATCH --output=cifar10_scripts/cifar10_${attack}_output.log # Output file
#SBATCH --error=cifar10_scripts/cifar10_${attack}_error.log   # Error file
#SBATCH --partition=gpu                                      # Use GPU partition
#SBATCH --gres=gpu:a100:1                                    # Request 1 A100 GPU
#SBATCH --time=06:00:00                                      # Max runtime

# Move to the directory containing the environment
cd "$HOME/BackdoorBox Research/backdoor-toolbox"
source "my_env/bin/activate"

# Step 1: Create poisoned training set
echo "Creating poisoned training set for ${attack} on cifar10..."
python create_poisoned_set.py -dataset=cifar10 -poison_type=$attack -poison_rate=0.01

# Step 2: Train the model on the poisoned dataset
echo "Training the model on the poisoned dataset..."
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=$attack -poison_rate=0.01

# Step 3: Test the backdoor model
echo "Testing the backdoor model..."
python test_model.py -dataset=cifar10 -poison_type=$attack -poison_rate=0.01

# Step 4: Visualize the model's latent space with PCA, TSNE, and Oracle
echo "Visualizing the model's latent space..."
python visualize.py -method=pca -dataset=cifar10 -poison_type=$attack -poison_rate=0.01
python visualize.py -method=tsne -dataset=cifar10 -poison_type=$attack -poison_rate=0.01
python visualize.py -method=oracle -dataset=cifar10 -poison_type=$attack -poison_rate=0.01

# Step 5: Apply IBD_PSC defense method
echo "Applying IBD_PSC defense method..."
python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=$attack -poison_rate=0.01

echo "Experiment for cifar10 with $attack completed."
EOT

    # Make the script executable
    chmod +x "$script_file"
done

echo "Script generation completed."

