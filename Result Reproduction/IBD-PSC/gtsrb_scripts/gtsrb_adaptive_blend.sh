#!/bin/bash
#SBATCH --job-name=gtsrb_adaptive_blend                         # Job name
#SBATCH --output=gtsrb_scripts/gtsrb_adaptive_blend_output.log  # Output file
#SBATCH --error=gtsrb_scripts/gtsrb_adaptive_blend_error.log    # Error file
#SBATCH --partition=gpu                                    # Use GPU partition
#SBATCH --gres=gpu:a100:1                                  # Request 1 A100 GPU
#SBATCH --time=06:00:00                                    # Max runtime

# Move to the directory containing the environment
cd "/home/s222576762/BackdoorBox Research/backdoor-toolbox"
source "my_env/bin/activate"

# Step 1: Create poisoned training set
echo "Creating poisoned training set for adaptive_blend on GTSRB..."
python create_poisoned_set.py -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.01

# Step 2: Train the model on the poisoned dataset
echo "Training the model on the poisoned dataset..."
python train_on_poisoned_set.py -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.01

# Step 3: Test the backdoor model
echo "Testing the backdoor model..."
python test_model.py -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.01

# Step 4: Visualize the model's latent space with PCA, TSNE, and Oracle
echo "Visualizing the model's latent space..."
python visualize.py -method=pca -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.01
python visualize.py -method=tsne -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.01
python visualize.py -method=oracle -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.01

# Step 5: Apply IBD_PSC defense method
echo "Applying IBD_PSC defense method..."
python other_defense.py -defense=IBD_PSC -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.01

echo "Experiment for GTSRB with adaptive_blend completed."
