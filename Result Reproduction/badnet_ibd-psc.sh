#!/bin/bash
#SBATCH --job-name=full_process_job            # Job name
#SBATCH --output=full_process_output.log       # File to save standard output
#SBATCH --error=full_process_error.log         # File to save standard error
#SBATCH --time=04:00:00                        # Maximum runtime (4 hours, adjust if needed)
#SBATCH --partition=gpu                        # Use the GPU partition
#SBATCH --gres=gpu:a100:1                      # Request 1 A100 GPU

# Move to the directory containing the environment
cd "$HOME/BackdoorBox Research/backdoor-toolbox"

# Activate the virtual environment
source "my_env/bin/activate"

# Step 1: Create clean dataset split for CIFAR10
echo "Step 1: Creating the clean dataset split for CIFAR10..."
srun "my_env/bin/python" "../backdoor-toolbox/create_clean_set.py" -dataset=cifar10

# Step 2: Create poisoned training set with BadNet attack
echo "Step 2: Creating poisoned training set with BadNet attack..."
srun "my_env/bin/python" "../backdoor-toolbox/create_poisoned_set.py" -dataset=cifar10 -poison_type=badnet -poison_rate=0.01

# Step 3: Train the model on the poisoned dataset
echo "Step 3: Training model on poisoned dataset..."
srun "my_env/bin/python" "../backdoor-toolbox/train_on_poisoned_set.py" -dataset=cifar10 -poison_type=badnet -poison_rate=0.01

# Step 4: Test the model on the poisoned test set before applying defense
echo "Step 4: Testing the model on the poisoned test set before applying defense..."
srun "my_env/bin/python" "../backdoor-toolbox/test_model.py" -dataset=cifar10 -poison_type=badnet -poison_rate=0.01

# Step 5: Apply IBD_PSC defense method
echo "Step 5: Applying IBD_PSC defense method..."
srun "my_env/bin/python" "../backdoor-toolbox/other_defense.py" -defense=IBD_PSC -dataset=cifar10 -poison_type=badnet -poison_rate=0.01

# Step 6: Visualize the model after defense
echo "Step 6: Visualizing the model after defense..."
srun "my_env/bin/python" "../backdoor-toolbox/visualize.py" -method=tsne -dataset=cifar10 -poison_type=badnet -poison_rate=0.01

