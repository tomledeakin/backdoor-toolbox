#!/bin/bash
#SBATCH --job-name=cifar_0_1                    # Job name
#SBATCH --output=cifar_0_1_output.log           # Standard output log
#SBATCH --error=cifar_0_1_error.log             # Standard error log
#SBATCH --partition=gpu                         # Use GPU partition
#SBATCH --gres=gpu:a100:1                       # Request 1 A100 GPU
#SBATCH --time=48:00:00                         # Time limit (48 hours)
#SBATCH --mail-user=tomledeakin@gmail.com       # Email for notifications
#SBATCH --mail-type=END,FAIL                    # Notify on job completion or failure

# Navigate to the project directory
cd "$HOME/BackdoorBox Research/backdoor-toolbox" || { echo "Directory not found"; exit 1; }

# Activate the Python virtual environment
source "my_env/bin/activate"

# # gtsrb - badnet
# echo "gtsrb - badnet | START"
# # python create_poisoned_set.py -dataset=gtsrb -poison_type=badnet -poison_rate=0.1
# # python train_on_poisoned_set.py -dataset=gtsrb -poison_type=badnet -poison_rate=0.1
# # python test_model.py -dataset=gtsrb -poison_type=badnet -poison_rate=0.1
# # python resnet18_layer_visualize.py -dataset=gtsrb -poison_type=badnet -poison_rate=0.1 -n_neighbors=50
# python other_defense.py -defense=IBD_PSC -dataset=gtsrb -poison_type=badnet -poison_rate=0.1
# echo "gtsrb - badnet | COMPLETE"

# # gtsrb - blend
# echo "gtsrb - blend | START"
# # python create_poisoned_set.py -dataset=gtsrb -poison_type=blend -poison_rate=0.1
# # python train_on_poisoned_set.py -dataset=gtsrb -poison_type=blend -poison_rate=0.1
# # python test_model.py -dataset=gtsrb -poison_type=blend -poison_rate=0.1
# # python resnet18_layer_visualize.py -dataset=gtsrb -poison_type=blend -poison_rate=0.1 -n_neighbors=50
# python other_defense.py -defense=IBD_PSC -dataset=gtsrb -poison_type=blend -poison_rate=0.1
# echo "gtsrb - blend | COMPLETE"

# # gtsrb - adaptive_blend
# echo "gtsrb - adaptive_blend | START"
# # python create_poisoned_set.py -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# # python train_on_poisoned_set.py -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# # python test_model.py -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# # python resnet18_layer_visualize.py -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05 -n_neighbors=50
# python other_defense.py -defense=IBD_PSC -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# echo "gtsrb - adaptive_blend | COMPLETE"

# # gtsrb - adaptive_patch
# echo "gtsrb - adaptive_patch | START"
# # python create_poisoned_set.py -dataset=gtsrb -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# # python train_on_poisoned_set.py -dataset=gtsrb -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# # python test_model.py -dataset=gtsrb -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# # python resnet18_layer_visualize.py -dataset=gtsrb -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05 -n_neighbors=50
# python other_defense.py -defense=IBD_PSC -dataset=gtsrb -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# echo "gtsrb - adaptive_patch | COMPLETE"

# cifar10 - badnet
echo "cifar10 - badnet | START"
python create_poisoned_set.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
python test_model.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
# python resnet18_layer_visualize.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
echo "cifar10 - badnet | COMPLETE"

# cifar10 - blend
echo "cifar10 - blend | START"
python create_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.1
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.1
python test_model.py -dataset=cifar10 -poison_type=blend -poison_rate=0.1
# python resnet18_layer_visualize.py -dataset=cifar10 -poison_type=blend -poison_rate=0.1
python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=blend -poison_rate=0.1
echo "cifar10 - blend | COMPLETE"

# cifar10 - adaptive_blend
echo "cifar10 - adaptive_blend | START"
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
python test_model.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# python resnet18_layer_visualize.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
echo "cifar10 - adaptive_blend | COMPLETE"

# cifar10 - adaptive_patch
echo "cifar10 - adaptive_patch | START"
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
python test_model.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# python resnet18_layer_visualize.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
echo "cifar10 - adaptive_patch | COMPLETE"






# echo "cifar10 - badnet | START | 0.09"
# python create_poisoned_set.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.09
# python train_on_poisoned_set.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.09
# python test_model.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.09
# # python resnet18_layer_visualize.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.09
# python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=badnet -poison_rate=0.09
# echo "cifar10 - badnet | COMPLETE | 0.09"

# # cifar10 - badnet
# echo "cifar10 - badnet | START | 0.1"
# python create_poisoned_set.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
# python train_on_poisoned_set.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
# python test_model.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
# # python resnet18_layer_visualize.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
# python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
# echo "cifar10 - badnet | COMPLETE | 0.1"

# echo "cifar10 - badnet | START | 0.13"
# python create_poisoned_set.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.13
# python train_on_poisoned_set.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.13
# python test_model.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.13
# # python resnet18_layer_visualize.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.13
# python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=badnet -poison_rate=0.13
# echo "cifar10 - badnet | COMPLETE | 0.13"

# echo "cifar10 - badnet | START | 0.15"
# python create_poisoned_set.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.15
# python train_on_poisoned_set.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.15
# python test_model.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.15
# # python resnet18_layer_visualize.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.15
# python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=badnet -poison_rate=0.15
# echo "cifar10 - badnet | COMPLETE | 0.15"

echo "All tasks completed successfully."



