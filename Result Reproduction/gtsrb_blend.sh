#!/bin/bash
#SBATCH --job-name=gtsrb_blend                        # Job name
#SBATCH --output=gtsrb_blend_output.log               # File to save standard output
#SBATCH --error=gtsrb_blend_error.log                 # File to save standard error
#SBATCH --partition=gpu                               # Use the GPU partition
#SBATCH --gres=gpu:a100:1                             # Request 1 A100 GPU
#SBATCH --time=04:00:00                               # Maximum runtime (4 hours, adjust if needed)

# Move to the directory containing the environment
cd "$HOME/BackdoorBox Research/backdoor-toolbox"

# Activate the virtual environment
source "my_env/bin/activate"

# Step 1: Create clean dataset split for GTSRB
echo "Step 1: Creating the clean dataset split for GTSRB..."
srun "my_env/bin/python" "../backdoor-toolbox/create_clean_set.py" -dataset=gtsrb

# Step 2: Create poisoned training set with Blend attack
echo "Step 2: Creating poisoned training set with Blend attack..."
srun "my_env/bin/python" "../backdoor-toolbox/create_poisoned_set.py" -dataset=gtsrb -poison_type=blend -poison_rate=0.01

# Step 3: Train the model on the poisoned dataset
echo "Step 3: Training model on poisoned dataset..."
srun "my_env/bin/python" "../backdoor-toolbox/train_on_poisoned_set.py" -dataset=gtsrb -poison_type=blend -poison_rate=0.01

# Step 4: Test the model on the poisoned test set before applying defense
echo "Step 4: Testing model on poisoned test set before applying defense..."
srun "my_env/bin/python" "../backdoor-toolbox/test_model.py" -dataset=gtsrb -poison_type=blend -poison_rate=0.01

# Step 5: Apply IBD_PSC defense method
echo "Step 5: Applying IBD_PSC defense method..."
srun "my_env/bin/python" "../backdoor-toolbox/other_defense.py" -defense=IBD_PSC -dataset=gtsrb -poison_type=blend -poison_rate=0.01

# Step 6: Visualize the model after defense
echo "Step 6: Visualizing the model after defense..."
srun "my_env/bin/python" "../backdoor-toolbox/visualize.py" -method=tsne -dataset=gtsrb -poison_type=blend -poison_rate=0.01

