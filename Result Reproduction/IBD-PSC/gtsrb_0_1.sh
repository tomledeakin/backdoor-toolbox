#!/bin/bash
#SBATCH --job-name=gtsrb_0_1                    # Job name
#SBATCH --output=gtsrb_0_1_output.log           # Standard output log
#SBATCH --error=gtsrb_0_1_error.log             # Standard error log
#SBATCH --partition=gpu                         # Use GPU partition
#SBATCH --gres=gpu:a100:1                       # Request 1 A100 GPU
#SBATCH --time=48:00:00                         # Time limit (48 hours)
#SBATCH --mail-user=tomledeakin@gmail.com       # Email for notifications
#SBATCH --mail-type=END,FAIL                    # Notify on job completion or failure


# Navigate to the project directory
cd "$HOME/BackdoorBox Research/backdoor-toolbox" || { echo "Directory not found"; exit 1; }

# Activate the Python virtual environment
source "my_env/bin/activate"


# cifar10 - badnet
echo "cifar10 - badnet | START"
# python create_poisoned_set.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
# python train_on_poisoned_set.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
# python test_model.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
python other_defense.py -defense=TED -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
# python connectivity_resnet18_layer_visualize.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.1 -n_neighbors=50
echo "cifar10 - badnet | COMPLETE"

# # cifar10 - adaptive_patch
# echo "cifar10 - adaptive_patch | START"
# python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# python test_model.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# python connectivity_resnet18_layer_visualize.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05 -n_neighbors=50
# echo "cifar10 - adaptive_patch | COMPLETE"



echo "All tasks completed successfully."





