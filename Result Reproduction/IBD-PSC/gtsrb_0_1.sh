#!/bin/bash
#SBATCH --job-name=hpc             # Job name
#SBATCH --output=output.log                      # Standard output log
#SBATCH --error=error.log                        # Standard error log
#SBATCH --partition=gpu                          # Use GPU partition
#SBATCH --gres=gpu:a100:2                             # Request 1 GPU
#SBATCH --time=48:00:00                          # Time limit (48 hours)
#SBATCH --mail-user=tomledeakin@gmail.com        # Email for notifications
#SBATCH --mail-type=ALL                     # Notify on job completion or failure


# Navigate to the project directory
cd "$HOME/BackdoorBox Research/backdoor-toolbox" || { echo "Directory not found"; exit 1; }

# Activate the Python virtual environment
source "my_env/bin/activate"
cd "$HOME/BackdoorBox Research/backdoor-toolbox"

echo "cifar10 - badnet | START"
# python create_poisoned_set.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
# python train_on_poisoned_set.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
# python test_model.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
# python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
python other_defense.py -defense=TED -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
echo "cifar10 - badnet | COMPLETE"

echo "cifar10 - blend | START"
# python create_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.1
# python train_on_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.1
# python test_model.py -dataset=cifar10 -poison_type=blend -poison_rate=0.1
# python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=blend -poison_rate=0.1
python other_defense.py -defense=TED -dataset=cifar10 -poison_type=blend -poison_rate=0.1
echo "cifar10 - blend | COMPLETE"

echo "cifar10 - adaptive_patch | START"
# python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# python test_model.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
python other_defense.py -defense=TED -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
echo "cifar10 - adaptive_patch | COMPLETE"

echo "cifar10 - adaptive_blend | START"
# python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# python test_model.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
python other_defense.py -defense=TED -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
echo "cifar10 - adaptive_blend | COMPLETE"

# echo "cifar10 - WaNet | START"
# python create_poisoned_set.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
# python train_on_poisoned_set.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
# python test_model.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
# python all_layers_resnet18_layer_visualize.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05 -data_ratio=0.4
# # python other_defense.py -defense=TED -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
# echo "cifar10 - WaNet | COMPLETE"

# echo "cifar10 - ISSBA | START"
# python create_poisoned_set.py -dataset=cifar10 -poison_type=ISSBA -poison_rate=0.1
# python train_on_poisoned_set.py -dataset=cifar10 -poison_type=ISSBA -poison_rate=0.1
# python test_model.py -dataset=cifar10 -poison_type=ISSBA -poison_rate=0.1
# python all_layers_resnet18_layer_visualize.py -dataset=cifar10 -poison_type=ISSBA -poison_rate=0.1 -data_ratio=0.4
# # python other_defense.py -defense=TED -dataset=cifar10 -poison_type=blend -poison_rate=0.1
# echo "cifar10 - ISSBA | COMPLETE"

# echo "cifar10 - trojan | START"
# python create_poisoned_set.py -dataset=cifar10 -poison_type=trojan -poison_rate=0.1
# python train_on_poisoned_set.py -dataset=cifar10 -poison_type=trojan -poison_rate=0.1
# python test_model.py -dataset=cifar10 -poison_type=trojan -poison_rate=0.1
# python all_layers_resnet18_layer_visualize.py -dataset=cifar10 -poison_type=trojan -poison_rate=0.1 -data_ratio=0.4
# # python other_defense.py -defense=TED -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# echo "cifar10 - trojan | COMPLETE"

# echo "cifar10 - SIG | START"
# python create_poisoned_set.py -dataset=cifar10 -poison_type=SIG -poison_rate=0.1
# python train_on_poisoned_set.py -dataset=cifar10 -poison_type=SIG -poison_rate=0.1
# python test_model.py -dataset=cifar10 -poison_type=SIG -poison_rate=0.1
# python all_layers_resnet18_layer_visualize.py -dataset=cifar10 -poison_type=SIG -poison_rate=0.1 -data_ratio=0.4
# # python other_defense.py -defense=TED -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# echo "cifar10 - SIG | COMPLETE"

echo "All tasks completed successfully."







