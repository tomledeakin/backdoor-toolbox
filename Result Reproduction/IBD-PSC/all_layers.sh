#!/bin/bash
#SBATCH --job-name=hpc             # Job name
#SBATCH --output=output.log                      # Standard output log
#SBATCH --error=error.log                        # Standard error log
#SBATCH --partition=gpu                          # Use GPU partition
#SBATCH --gres=gpu:a100:1                             # Request 1 GPU
#SBATCH --time=48:00:00                          # Time limit (48 hours)
#SBATCH --mail-user=tomledeakin@gmail.com        # Email for notifications
#SBATCH --mail-type=ALL                     # Notify on job completion or failure

# Navigate to the project directory
cd "$HOME/BackdoorBox Research/backdoor-toolbox" || { echo "Directory not found"; exit 1; }

# Activate the Python virtual environment
source "my_env/bin/activate"
cd "$HOME/BackdoorBox Research/backdoor-toolbox"

# cifar10 - badnet
echo "cifar10 - badnet | START"
python all_layers_resnet18_layer_visualize.py -dataset=cifar10 -poison_type=blend -poison_rate=0.1 -data_ratio=0.4
echo "cifar10 - badnet | COMPLETE"

# cifar10 - adaptive_patch
echo "cifar10 - adaptive_patch | START"
python all_layers_resnet18_layer_visualize.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05 -data_ratio=0.4
echo "cifar10 - adaptive_patch | COMPLETE"

echo "All tasks completed successfully."





