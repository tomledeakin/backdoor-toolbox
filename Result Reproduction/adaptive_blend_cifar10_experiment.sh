#!/bin/bash
#SBATCH --job-name=adaptive_blend_experiment                   # Job name
#SBATCH --output=%x_%j_output.log                              # Output file
#SBATCH --error=%x_%j_error.log                                # Error file
#SBATCH --time=06:00:00                                       # Max runtime (adjust as needed)
#SBATCH --partition=gpu                                       # Use GPU partition
#SBATCH --gres=gpu:a100:1                                     # Request 1 A100 GPU

# Move to the directory containing the environment
cd "$HOME/BackdoorBox Research/backdoor-toolbox"

# Activate the virtual environment
source "my_env/bin/activate"

# Record the start time
start_time=$(date +%Y-%m-%d_%H-%M-%S)
echo "Start time: $start_time"

# Step 1: Create poisoned training set
echo "Step 1: Creating poisoned training set..."
srun python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.003 -cover_rate=0.003 -alpha 0.15

# Step 2: Train on the poisoned training set
echo "Step 2: Training on the poisoned training set..."
srun python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.003 -cover_rate=0.003 -alpha 0.15 -test_alpha 0.2

# Step 3: Test the backdoor model
echo "Step 3: Testing the backdoor model..."
srun python test_model.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.003 -cover_rate=0.003 -alpha 0.15 -test_alpha 0.2

# Step 4: Visualize using PCA
echo "Step 4: Visualizing with PCA..."
srun python visualize.py -method=pca -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.003 -cover_rate=0.003 -alpha 0.15 -test_alpha 0.2

# Step 5: Cleanse with SCAn
echo "Step 5: Cleansing with SCAn..."
srun python cleanser.py -cleanser=SCAn -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.003 -cover_rate=0.003 -alpha 0.15 -test_alpha 0.2

# Step 6: Retrain on cleansed set with SCAn
echo "Step 6: Retraining on cleansed set with SCAn..."
srun python train_on_cleansed_set.py -cleanser=SCAn -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.003 -cover_rate=0.003 -alpha 0.15 -test_alpha 0.2

# Step 7: Apply defense ABL
echo "Step 7: Applying defense ABL..."
srun python other_defense.py -defense=ABL -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.003 -cover_rate=0.003 -alpha 0.15 -test_alpha 0.2

# Record the end time
end_time=$(date +%Y-%m-%d_%H-%M-%S)
echo "End time: $end_time"

