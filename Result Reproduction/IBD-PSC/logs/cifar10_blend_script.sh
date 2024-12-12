#!/bin/bash
#SBATCH --job-name=cifar10_blend
#SBATCH --output=logs/cifar10_blend_output.log
#SBATCH --error=logs/cifar10_blend_error.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=48:00:00
#SBATCH --mail-user=tomledeakin@gmail.com
#SBATCH --mail-type=END,FAIL

cd "/home/s222576762/BackdoorBox Research/backdoor-toolbox" || { echo 'Directory not found'; exit 1; }
source "my_env/bin/activate"

python create_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.1
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.1
python test_model.py -dataset=cifar10 -poison_type=blend -poison_rate=0.1
# python resnet18_layer_visualize.py -dataset=cifar10 -poison_type=blend -poison_rate=0.1
python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=blend -poison_rate=0.1
echo "cifar10 - blend | COMPLETE"
