#!/bin/bash
#SBATCH --job-name=badnet_vs_ibd_psc_experiment_no_cleanser   # Job name
#SBATCH --output=%x_%j_output.log                             # Output file
#SBATCH --error=%x_%j_error.log                               # Error file
#SBATCH --time=06:00:00                                       # Max runtime (adjust as needed)
#SBATCH --partition=gpu                                       # Use GPU partition
#SBATCH --gres=gpu:a100:1                                     # Request 1 A100 GPU

# Move to the directory containing the environment
cd "$HOME/BackdoorBox Research/backdoor-toolbox"

# Activate the virtual environment
source "my_env/bin/activate"

# Record start time
start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "Experiment started at: $start_time"

# Step 1: Create poisoned training set with badnet attack
echo "Step 1: Creating poisoned training set with badnet attack..."
srun "my_env/bin/python" "create_poisoned_set.py" -dataset=cifar10 -poison_type=badnet -poison_rate=0.01

# Step 2: Train the model on the poisoned dataset
echo "Step 2: Training the model on the poisoned dataset..."
srun "my_env/bin/python" "train_on_poisoned_set.py" -dataset=cifar10 -poison_type=badnet -poison_rate=0.01

# Step 3: Test the backdoor model
echo "Step 3: Testing the backdoor model..."
srun "my_env/bin/python" "test_model.py" -dataset=cifar10 -poison_type=badnet -poison_rate=0.01

# Step 4: Visualize the model's latent space with PCA
echo "Step 4: Visualizing the model's latent space with PCA..."
srun "my_env/bin/python" "visualize.py" -method=pca -dataset=cifar10 -poison_type=badnet -poison_rate=0.01

# Step 5: Apply IBD_PSC defense method
echo "Step 5: Applying IBD_PSC defense method..."
srun "my_env/bin/python" "other_defense.py" -defense=IBD_PSC -dataset=cifar10 -poison_type=badnet -poison_rate=0.01

# Record end time
end_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "Experiment ended at: $end_time"

