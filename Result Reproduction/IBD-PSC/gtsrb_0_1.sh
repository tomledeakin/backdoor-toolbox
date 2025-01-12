#!/bin/bash
#SBATCH --job-name=hpc             # Job name
#SBATCH --output=output.log                      # Standard output log
#SBATCH --error=error.log                        # Standard error log
#SBATCH --partition=gpu                          # Use GPU partition
#SBATCH --gres=gpu:a100:1                            # Request 1 GPU
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
# # python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
# python other_defense.py -defense=TED -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
# echo "TED | cifar10 - badnet | COMPLETE"
python other_defense.py -defense=FOLD -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
echo "FOLD | cifar10 - badnet | COMPLETE"
echo "cifar10 - badnet | COMPLETE"

echo "cifar10 - blend | START"
# python create_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.1
# python train_on_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.1
# python test_model.py -dataset=cifar10 -poison_type=blend -poison_rate=0.1
# # python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=blend -poison_rate=0.1
# python other_defense.py -defense=TED -dataset=cifar10 -poison_type=blend -poison_rate=0.1
# echo "TED | cifar10 - blend | COMPLETE"
python other_defense.py -defense=FOLD -dataset=cifar10 -poison_type=blend -poison_rate=0.1
echo "FOLD | cifar10 - blend | COMPLETE"
echo "cifar10 - blend | COMPLETE"

echo "cifar10 - adaptive_patch | START"
# python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# python test_model.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# python other_defense.py -defense=TED -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# echo "TED | cifar10 - adaptive_patch | COMPLETE"
python other_defense.py -defense=FOLD -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
echo "FOLD | cifar10 - adaptive_patch | COMPLETE"
echo "cifar10 - adaptive_patch | COMPLETE"

echo "cifar10 - adaptive_blend | START"
# python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# python test_model.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# python other_defense.py -defense=TED -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# echo "TED | cifar10 - adaptive_blend | COMPLETE"
python other_defense.py -defense=FOLD -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
echo "FOLD | cifar10 - adaptive_blend | COMPLETE"
echo "cifar10 - adaptive_blend | COMPLETE"

echo "cifar10 - WaNet | START"
# python create_poisoned_set.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
# python train_on_poisoned_set.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
# python test_model.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
# python all_layers_resnet18_layer_visualize.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05 -data_ratio=0.4
# python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
# python other_defense.py -defense=TED -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
# echo "TED | cifar10 - WaNet | COMPLETE"
python other_defense.py -defense=FOLD -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
echo "FOLD | cifar10 - WaNet | COMPLETE"
echo "cifar10 - WaNet | COMPLETE"

echo "cifar10 - trojan | START"
# python create_poisoned_set.py -dataset=cifar10 -poison_type=trojan -poison_rate=0.1
# python train_on_poisoned_set.py -dataset=cifar10 -poison_type=trojan -poison_rate=0.1
# python test_model.py -dataset=cifar10 -poison_type=trojan -poison_rate=0.1
# python all_layers_resnet18_layer_visualize.py -dataset=cifar10 -poison_type=trojan -poison_rate=0.1 -data_ratio=0.4
# python other_defense.py -defense=TED -dataset=cifar10 -poison_type=trojan -poison_rate=0.1 
# echo "TED | cifar10 - trojan | COMPLETE"
python other_defense.py -defense=FOLD -dataset=cifar10 -poison_type=trojan -poison_rate=0.1
echo "FOLD | cifar10 - trojan | COMPLETE"
echo "cifar10 - trojan | COMPLETE"

echo "cifar10 - dynamic | START"
# python create_poisoned_set.py -dataset=cifar10 -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
# python train_on_poisoned_set.py -dataset=cifar10 -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
# python test_model.py -dataset=cifar10 -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
# python all_layers_resnet18_layer_visualize.py -dataset=cifar10 -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05 -data_ratio=0.4
# python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
# python other_defense.py -defense=TED -dataset=cifar10 -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
# echo "TED | cifar10 - dynamic | COMPLETE"
python other_defense.py -defense=FOLD -dataset=cifar10 -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
echo "FOLD | cifar10 - dynamic | COMPLETE"
echo "cifar10 - dynamic | COMPLETE"


echo "cifar10 - TaCT | START"
# python create_poisoned_set.py -dataset=cifar10 -poison_type=TaCT -poison_rate=0.1 -cover_rate=0.05
# python train_on_poisoned_set.py -dataset=cifar10 -poison_type=TaCT -poison_rate=0.1 -cover_rate=0.05
# python test_model.py -dataset=cifar10 -poison_type=TaCT -poison_rate=0.1 -cover_rate=0.05
# python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=TaCT -poison_rate=0.1 -cover_rate=0.05
# python all_layers_resnet18_layer_visualize.py -dataset=cifar10 -poison_type=TaCT -poison_rate=0.1 -cover_rate=0.05 -data_ratio=0.4
# python other_defense.py -defense=TED -dataset=cifar10 -poison_type=TaCT -poison_rate=0.1 -cover_rate=0.05
# echo "TED | cifar10 - TaCT | COMPLETE"
python other_defense.py -defense=FOLD -dataset=cifar10 -poison_type=TaCT -poison_rate=0.1 -cover_rate=0.05
echo "FOLD | cifar10 - TaCT | COMPLETE"
echo "cifar10 - TaCT | COMPLETE"


# echo "gtsrb - badnet | START"
# # python create_poisoned_set.py -dataset=gtsrb -poison_type=badnet -poison_rate=0.1
# # python train_on_poisoned_set.py -dataset=gtsrb -poison_type=badnet -poison_rate=0.1
# # python test_model.py -dataset=gtsrb -poison_type=badnet -poison_rate=0.1
# # # python other_defense.py -defense=IBD_PSC -dataset=gtsrb -poison_type=badnet -poison_rate=0.1
# # python other_defense.py -defense=TED -dataset=gtsrb -poison_type=badnet -poison_rate=0.1
# # echo "TED | gtsrb - badnet | COMPLETE"
# python other_defense.py -defense=FOLD -dataset=gtsrb -poison_type=badnet -poison_rate=0.1
# echo "FOLD | gtsrb - badnet | COMPLETE"
# echo "gtsrb - badnet | COMPLETE"

# echo "gtsrb - blend | START"
# # python create_poisoned_set.py -dataset=gtsrb -poison_type=blend -poison_rate=0.1
# # python train_on_poisoned_set.py -dataset=gtsrb -poison_type=blend -poison_rate=0.1
# # python test_model.py -dataset=gtsrb -poison_type=blend -poison_rate=0.1
# # # python other_defense.py -defense=IBD_PSC -dataset=gtsrb -poison_type=blend -poison_rate=0.1
# # python other_defense.py -defense=TED -dataset=gtsrb -poison_type=blend -poison_rate=0.1
# # echo "TED | gtsrb - blend | COMPLETE"
# python other_defense.py -defense=FOLD -dataset=gtsrb -poison_type=blend -poison_rate=0.1
# echo "FOLD | gtsrb - blend | COMPLETE"
# echo "gtsrb - blend | COMPLETE"

# echo "gtsrb - adaptive_patch | START"
# # python create_poisoned_set.py -dataset=gtsrb -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# # python train_on_poisoned_set.py -dataset=gtsrb -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# # python test_model.py -dataset=gtsrb -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# # python other_defense.py -defense=IBD_PSC -dataset=gtsrb -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# # python other_defense.py -defense=TED -dataset=gtsrb -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# # echo "TED | gtsrb - adaptive_patch | COMPLETE"
# python other_defense.py -defense=FOLD -dataset=gtsrb -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# echo "FOLD | gtsrb - adaptive_patch | COMPLETE"
# echo "gtsrb - adaptive_patch | COMPLETE"

# echo "gtsrb - adaptive_blend | START"
# # python create_poisoned_set.py -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# # python train_on_poisoned_set.py -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# # python test_model.py -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# # python other_defense.py -defense=IBD_PSC -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# # python other_defense.py -defense=TED -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# # echo "TED | gtsrb - adaptive_blend | COMPLETE"
# python other_defense.py -defense=FOLD -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# echo "FOLD | gtsrb - adaptive_blend | COMPLETE"
# echo "gtsrb - adaptive_blend | COMPLETE"

# echo "gtsrb - WaNet | START"
# # python create_poisoned_set.py -dataset=gtsrb -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
# # python train_on_poisoned_set.py -dataset=gtsrb -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
# # python test_model.py -dataset=gtsrb -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
# # python all_layers_resnet18_layer_visualize.py -dataset=gtsrb -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05 -data_ratio=0.4
# # python other_defense.py -defense=IBD_PSC -dataset=gtsrb -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
# # python other_defense.py -defense=TED -dataset=gtsrb -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
# # echo "TED | gtsrb - WaNet | COMPLETE"
# python other_defense.py -defense=FOLD -dataset=gtsrb -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
# echo "FOLD | gtsrb - WaNet | COMPLETE"
# echo "gtsrb - WaNet | COMPLETE"

# echo "gtsrb - trojan | START"
# # python create_poisoned_set.py -dataset=gtsrb -poison_type=trojan -poison_rate=0.1
# # python train_on_poisoned_set.py -dataset=gtsrb -poison_type=trojan -poison_rate=0.1
# # python test_model.py -dataset=gtsrb -poison_type=trojan -poison_rate=0.1
# # python all_layers_resnet18_layer_visualize.py -dataset=gtsrb -poison_type=trojan -poison_rate=0.1 -data_ratio=0.4
# # python other_defense.py -defense=TED -dataset=gtsrb -poison_type=trojan -poison_rate=0.1 
# # echo "TED | gtsrb - trojan | COMPLETE"
# python other_defense.py -defense=FOLD -dataset=gtsrb -poison_type=trojan -poison_rate=0.1
# echo "FOLD | gtsrb - trojan | COMPLETE"
# echo "gtsrb - trojan | COMPLETE"

# echo "gtsrb - dynamic | START"
# # python create_poisoned_set.py -dataset=gtsrb -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
# # python train_on_poisoned_set.py -dataset=gtsrb -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
# # python test_model.py -dataset=gtsrb -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
# # python all_layers_resnet18_layer_visualize.py -dataset=gtsrb -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05 -data_ratio=0.4
# # python other_defense.py -defense=IBD_PSC -dataset=gtsrb -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
# # python other_defense.py -defense=TED -dataset=gtsrb -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
# # echo "TED | gtsrb - dynamic | COMPLETE"
# python other_defense.py -defense=FOLD -dataset=gtsrb -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
# echo "FOLD | gtsrb - dynamic | COMPLETE"
# echo "gtsrb - dynamic | COMPLETE"




echo "All tasks completed successfully."







