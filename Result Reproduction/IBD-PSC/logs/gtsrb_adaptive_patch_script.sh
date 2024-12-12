#!/bin/bash
#SBATCH --job-name=gtsrb_adaptive_patch
#SBATCH --output=logs/gtsrb_adaptive_patch_output.log
#SBATCH --error=logs/gtsrb_adaptive_patch_error.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=48:00:00
#SBATCH --mail-user=tomledeakin@gmail.com
#SBATCH --mail-type=END,FAIL

cd "/home/s222576762/BackdoorBox Research/backdoor-toolbox" || { echo 'Directory not found'; exit 1; }
source "my_env/bin/activate"

python create_poisoned_set.py -dataset=gtsrb -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
python train_on_poisoned_set.py -dataset=gtsrb -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
python test_model.py -dataset=gtsrb -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# python resnet18_layer_visualize.py -dataset=gtsrb -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05 -n_neighbors=50
python other_defense.py -defense=IBD_PSC -dataset=gtsrb -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
echo "gtsrb - adaptive_patch | COMPLETE"
