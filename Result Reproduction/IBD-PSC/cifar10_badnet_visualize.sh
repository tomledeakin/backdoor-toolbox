#!/bin/bash
#SBATCH --job-name=cifar10_badnet_visualize  # Job name
#SBATCH --output=cifar10_badnet_logs/cifar10_badnet_output.log  # Output file
#SBATCH --error=cifar10_badnet_logs/cifar10_badnet_error.log  # Error file
#SBATCH --partition=gpu  # Use GPU partition
#SBATCH --gres=gpu:a100:1  # Request 1 A100 GPU
#SBATCH --time=06:00:00  # Max runtime

# Move to the directory containing the environment
cd "$HOME/BackdoorBox Research/backdoor-toolbox"
source "my_env/bin/activate"

# Step: Visualize the model's latent space across layers
echo "Visualizing the model's latent space across layers..."

# Run the visualization script without the -method argument
python layer_visualize.py -dataset cifar10 -poison_type badnet -poison_rate=0.01

echo "Layer-wise visualization for cifar10 with badnet completed."

