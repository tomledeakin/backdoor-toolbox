#!/bin/bash
#SBATCH --job-name=gtsrb
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1   # Yêu cầu 2 GPU A100 (nhưng node radagast có 4 GPU, bạn có thể sử dụng đủ bộ nhớ của GPU0)
#SBATCH --mem=200G          # Cấp đủ RAM hệ thống nếu cần
#SBATCH --time=24:00:00
#SBATCH --mail-user=your_email@domain.com
#SBATCH --mail-type=ALL

# Navigate to the project directory
cd "$HOME/BackdoorBox Research/backdoor-toolbox" || { echo "Directory not found"; exit 1; }

# Activate the Python virtual environment
source "my_env/bin/activate"
cd "$HOME/BackdoorBox Research/backdoor-toolbox"
export PYTHONUNBUFFERED=1

# python train_SSDT.py --dataset cifar10 --attack_mode SSDT --n_iters 200
# python train_SSDT.py --dataset gtsrb --attack_mode SSDT --n_iters 200

# python create_poisoned_set.py -dataset=cifar10 -poison_type=clean_label -poison_rate=0.01
#python train_on_poisoned_set.py -dataset=cifar10 -poison_type=clean_label -poison_rate=0.01 -resume_from_meta_info

#python create_poisoned_set.py -dataset=tinyimagenet200 -poison_type=badnet -poison_rate=0.1
python train_on_poisoned_set.py -dataset=tinyimagenet200 -poison_type=badnet -poison_rate=0.1

#echo "cifar10 - badnet | START"
## python create_poisoned_set.py -dataset=cifar10 -poison_type=badnet -poison_rate='0.1'
## python train_on_poisoned_set.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
## python test_model.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
## python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
#python other_defense.py -defense=TED -dataset=cifar10 -poison_type=badnet -poison_rate=0.1 -validation_per_class=2 -num_test_samples=50
#echo "TED | cifar10 - badnet | COMPLETE"
#python other_defense.py -defense=TEDPLUS -dataset=cifar10 -poison_type=badnet -poison_rate=0.1 -validation_per_class=2 -num_test_samples=50 -num_neighbors=3
#echo "TEDPLUS | cifar10 - badnet | COMPLETE"
#echo "cifar10 - badnet | COMPLETE"
#
#echo "cifar10 - blend | START"
## python create_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.1
## python train_on_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.1
## python test_model.py -dataset=cifar10 -poison_type=blend -poison_rate=0.1
## python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=blend -poison_rate=0.1
#python other_defense.py -defense=TED -dataset=cifar10 -poison_type=blend -poison_rate=0.1 -validation_per_class=2 -num_test_samples=50
#echo "TED | cifar10 - blend | COMPLETE"
#python other_defense.py -defense=TEDPLUS -dataset=cifar10 -poison_type=blend -poison_rate=0.1 -validation_per_class=2 -num_test_samples=50 -num_neighbors=3
#echo "TEDPLUS | cifar10 - blend | COMPLETE"
#echo "cifar10 - blend | COMPLETE"
#
#echo "cifar10 - adaptive_patch | START"
## python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
## python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
## python test_model.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
## python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=10
#python other_defense.py -defense=TED -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=2 -num_test_samples=50
#echo "TED | cifar10 - adaptive_patch | COMPLETE"
#python other_defense.py -defense=TEDPLUS -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=2 -num_test_samples=50 -num_neighbors=3
#echo "TEDPLUS | cifar10 - adaptive_patch | COMPLETE"
#echo "cifar10 - adaptive_patch | COMPLETE"

#echo "cifar10 - adaptive_blend | START"
## python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
## python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
## python test_model.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
## python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=10
#python other_defense.py -defense=TED -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=5 -num_test_samples=50
#echo "TED | cifar10 - adaptive_blend | COMPLETE"
##python other_defense.py -defense=TEDPLUS -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=2 -num_test_samples=50 -num_neighbors=3
##echo "TEDPLUS | cifar10 - adaptive_blend | COMPLETE"
#echo "cifar10 - adaptive_blend | COMPLETE"

#echo "cifar10 - WaNet | START"
## python create_poisoned_set.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
## python train_on_poisoned_set.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
## python test_model.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
## python all_layers_resnet18_layer_visualize.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05 -data_ratio=0.4
## python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
#python other_defense.py -defense=TED -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=2 -num_test_samples=50
#echo "TED | cifar10 - WaNet | COMPLETE"
#python other_defense.py -defense=TEDPLUS -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=2 -num_test_samples=50 -num_neighbors=3
#echo "TEDPLUS | cifar10 - WaNet | COMPLETE"
#echo "cifar10 - WaNet | COMPLETE"
#
#echo "cifar10 - trojan | START"
## python create_poisoned_set.py -dataset=cifar10 -poison_type=trojan -poison_rate=0.1
## python train_on_poisoned_set.py -dataset=cifar10 -poison_type=trojan -poison_rate=0.1
## python test_model.py -dataset=cifar10 -poison_type=trojan -poison_rate=0.1
## python all_layers_resnet18_layer_visualize.py -dataset=cifar10 -poison_type=trojan -poison_rate=0.1 -data_ratio=0.4
#python other_defense.py -defense=TED -dataset=cifar10 -poison_type=trojan -poison_rate=0.1 -validation_per_class=2 -num_test_samples=50
#echo "TED | cifar10 - trojan | COMPLETE"
#python other_defense.py -defense=TEDPLUS -dataset=cifar10 -poison_type=trojan -poison_rate=0.1 -validation_per_class=2 -num_test_samples=50 -num_neighbors=3
#echo "TEDPLUS | cifar10 - trojan | COMPLETE"
#echo "cifar10 - trojan | COMPLETE"
#
#echo "cifar10 - dynamic | START"
## python create_poisoned_set.py -dataset=cifar10 -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
## python train_on_poisoned_set.py -dataset=cifar10 -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
## python test_model.py -dataset=cifar10 -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
## python all_layers_resnet18_layer_visualize.py -dataset=cifar10 -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05 -data_ratio=0.4
## python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
#python other_defense.py -defense=TED -dataset=cifar10 -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=2 -num_test_samples=50
#echo "TED | cifar10 - dynamic | COMPLETE"
#python other_defense.py -defense=TEDPLUS -dataset=cifar10 -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=2 -num_test_samples=50 -num_neighbors=3
#echo "TEDPLUS | cifar10 - dynamic | COMPLETE"
#echo "cifar10 - dynamic | COMPLETE"
#
#
#echo "cifar10 - TaCT | START"
## python create_poisoned_set.py -dataset=cifar10 -poison_type=TaCT -poison_rate=0.01 -cover_rate=0.005
## python train_on_poisoned_set.py -dataset=cifar10 -poison_type=TaCT -poison_rate=0.01 -cover_rate=0.005
## python test_model.py -dataset=cifar10 -poison_type=TaCT -poison_rate=0.02 -cover_rate=0.01
## python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=TaCT -poison_rate=0.02 -cover_rate=0.01
## python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=TaCT -poison_rate=0.01
## python all_layers_resnet18_layer_visualize.py -dataset=cifar10 -poison_type=TaCT -poison_rate=0.02 -cover_rate=0.01 -data_ratio=0.3
#python other_defense.py -defense=TED -dataset=cifar10 -poison_type=TaCT -poison_rate=0.02 -cover_rate=0.01 -validation_per_class=100 -num_test_samples=50
#echo "TED | cifar10 - TaCT | COMPLETE"
#python other_defense.py -defense=TEDPLUS -dataset=cifar10 -poison_type=TaCT -poison_rate=0.02 -cover_rate=0.01 -validation_per_class=2 -num_test_samples=50 -num_neighbors=3
#echo "TEDPLUS | cifar10 - TaCT | COMPLETE"
#echo "cifar10 - TaCT | COMPLETE"
#
#echo "cifar10 - SSDT | START"
## python train_SSDT.py --dataset gtsrb --attack_mode SSDT --n_iters 200
## python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=SSDT
#
#python other_defense.py -defense=TED -dataset=cifar10 -poison_type=SSDT -validation_per_class=2 -num_test_samples=50
#echo "TED - complete"
#python other_defense.py -defense=TEDPLUS -dataset=cifar10 -poison_type=SSDT -validation_per_class=2 -num_test_samples=50 -num_neighbors=3
#echo "TEDPLUS - complete"
#echo "cifar10 - SSDT | COMPLETE"
#
#
#echo "gtsrb - badnet | START"
## python create_poisoned_set.py -dataset=gtsrb -poison_type=badnet -poison_rate=0.1
## python train_on_poisoned_set.py -dataset=gtsrb -poison_type=badnet -poison_rate=0.1
## python test_model.py -dataset=gtsrb -poison_type=badnet -poison_rate=0.1
## python other_defense.py -defense=IBD_PSC -dataset=gtsrb -poison_type=badnet -poison_rate=0.1
#python other_defense.py -defense=TED -dataset=gtsrb -poison_type=badnet -poison_rate=0.1 -validation_per_class=2 -num_test_samples=50
#echo "TED | gtsrb - badnet | COMPLETE"
#python other_defense.py -defense=TEDPLUS -dataset=gtsrb -poison_type=badnet -poison_rate=0.1 -validation_per_class=2 -num_test_samples=50 -num_neighbors=3
#echo "TEDPLUS | gtsrb - badnet | COMPLETE"
#echo "gtsrb - badnet | COMPLETE"
#
#echo "gtsrb - blend | START"
## python create_poisoned_set.py -dataset=gtsrb -poison_type=blend -poison_rate=0.1
## python train_on_poisoned_set.py -dataset=gtsrb -poison_type=blend -poison_rate=0.1
## python test_model.py -dataset=gtsrb -poison_type=blend -poison_rate=0.1
## python other_defense.py -defense=IBD_PSC -dataset=gtsrb -poison_type=blend -poison_rate=0.1
#python other_defense.py -defense=TED -dataset=gtsrb -poison_type=blend -poison_rate=0.1 -validation_per_class=2 -num_test_samples=50
#echo "TED | gtsrb - blend | COMPLETE"
#python other_defense.py -defense=TEDPLUS -dataset=gtsrb -poison_type=blend -poison_rate=0.1 -validation_per_class=2 -num_test_samples=50 -num_neighbors=3
#echo "TEDPLUS | gtsrb - blend | COMPLETE"
#echo "gtsrb - blend | COMPLETE"
#
#echo "gtsrb - adaptive_patch | START"
## python create_poisoned_set.py -dataset=gtsrb -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
## python train_on_poisoned_set.py -dataset=gtsrb -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
## python test_model.py -dataset=gtsrb -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
## python other_defense.py -defense=IBD_PSC -dataset=gtsrb -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
#python other_defense.py -defense=TED -dataset=gtsrb -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=2 -num_test_samples=50
#echo "TED | gtsrb - adaptive_patch | COMPLETE"
#python other_defense.py -defense=TEDPLUS -dataset=gtsrb -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=2 -num_test_samples=50 -num_neighbors=3
#echo "TEDPLUS | gtsrb - adaptive_patch | COMPLETE"
#echo "gtsrb - adaptive_patch | COMPLETE"
#
#echo "gtsrb - adaptive_blend | START"
## python create_poisoned_set.py -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
## python train_on_poisoned_set.py -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
## python test_model.py -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
## python other_defense.py -defense=IBD_PSC -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
#python other_defense.py -defense=TED -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=2 -num_test_samples=50
#echo "TED | gtsrb - adaptive_blend | COMPLETE"
#python other_defense.py -defense=TEDPLUS -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=2 -num_test_samples=50 -num_neighbors=1
#echo "TEDPLUS | gtsrb - adaptive_blend | COMPLETE"
#echo "gtsrb - adaptive_blend | COMPLETE"
#
#echo "gtsrb - WaNet | START"
## python create_poisoned_set.py -dataset=gtsrb -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
## python train_on_poisoned_set.py -dataset=gtsrb -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
## python test_model.py -dataset=gtsrb -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
## python all_layers_resnet18_layer_visualize.py -dataset=gtsrb -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05 -data_ratio=0.4
## python other_defense.py -defense=IBD_PSC -dataset=gtsrb -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
#python other_defense.py -defense=TED -dataset=gtsrb -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=2 -num_test_samples=50
#echo "TED | gtsrb - WaNet | COMPLETE"
#python other_defense.py -defense=TEDPLUS -dataset=gtsrb -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=2 -num_test_samples=50 -num_neighbors=3
#echo "TEDPLUS | gtsrb - WaNet | COMPLETE"
#echo "gtsrb - WaNet | COMPLETE"
#
#echo "gtsrb - trojan | START"
## python create_poisoned_set.py -dataset=gtsrb -poison_type=trojan -poison_rate=0.1
## python train_on_poisoned_set.py -dataset=gtsrb -poison_type=trojan -poison_rate=0.1
## python test_model.py -dataset=gtsrb -poison_type=trojan -poison_rate=0.1
## python all_layers_resnet18_layer_visualize.py -dataset=gtsrb -poison_type=trojan -poison_rate=0.1 -data_ratio=0.4
## python other_defense.py -defense=IBD_PSC -dataset=gtsrb -poison_type=trojan -poison_rate=0.1
#python other_defense.py -defense=TED -dataset=gtsrb -poison_type=trojan -poison_rate=0.1 -validation_per_class=2 -num_test_samples=50
#echo "TED | gtsrb - trojan | COMPLETE"
#python other_defense.py -defense=TEDPLUS -dataset=gtsrb -poison_type=trojan -poison_rate=0.1 -validation_per_class=2 -num_test_samples=50 -num_neighbors=3
#echo "TEDPLUS | gtsrb - trojan | COMPLETE"
#echo "gtsrb - trojan | COMPLETE"
#
#echo "gtsrb - dynamic | START"
## python create_poisoned_set.py -dataset=gtsrb -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
## python train_on_poisoned_set.py -dataset=gtsrb -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
## python test_model.py -dataset=gtsrb -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
## python all_layers_resnet18_layer_visualize.py -dataset=gtsrb -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05 -data_ratio=0.4
## python other_defense.py -defense=IBD_PSC -dataset=gtsrb -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
#python other_defense.py -defense=TED -dataset=gtsrb -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=2 -num_test_samples=50
#echo "TED | gtsrb - dynamic | COMPLETE"
#python other_defense.py -defense=TEDPLUS -dataset=gtsrb -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=2 -num_test_samples=50 -num_neighbors=3
#echo "TEDPLUS | gtsrb - dynamic | COMPLETE"
#echo "gtsrb - dynamic | COMPLETE"
#
#echo "gtsrb - TaCT | START"
## python test_model.py -dataset=gtsrb -poison_type=TaCT -poison_rate=0.02 -cover_rate=0.01
## python other_defense.py -defense=IBD_PSC -dataset=gtsrb -poison_type=TaCT -poison_rate=0.02 -cover_rate=0.01
#
#python other_defense.py -defense=TED -dataset=gtsrb -poison_type=TaCT -poison_rate=0.02 -cover_rate=0.01 -validation_per_class=2 -num_test_samples=50
#echo "TED | gtsrb - TaCT | COMPLETE"
#python other_defense.py -defense=TEDPLUS -dataset=gtsrb -poison_type=TaCT -poison_rate=0.02 -cover_rate=0.01 -validation_per_class=2 -num_test_samples=50 -num_neighbors=3
#echo "TEDPLUS | gtsrb - TaCT | COMPLETE"
#echo "gtsrb - TaCT | COMPLETE"
#
#
#echo "gtsrb - SSDT | START"
## # python train_SSDT.py --dataset gtsrb --attack_mode SSDT --n_iters 1000
## python other_defense.py -defense=IBD_PSC -dataset=gtsrb -poison_type=SSDT
#
#python other_defense.py -defense=TED -dataset=gtsrb -poison_type=SSDT -validation_per_class=2 -num_test_samples=50
#echo "TED - complete"
#python other_defense.py -defense=TEDPLUS -dataset=gtsrb -poison_type=SSDT -validation_per_class=2 -num_test_samples=50 -num_neighbors=3
#echo "TEDPLUS - complete"
#echo "gtsrb - SSDT | COMPLETE"
#
#echo "========================================="


# echo "cifar10 - badnet | START"
# # python create_poisoned_set.py -dataset=cifar10 -poison_type=badnet -poison_rate='0.1'
# # python train_on_poisoned_set.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
# # python test_model.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
# # # python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
# python other_defense.py -defense=TED -dataset=cifar10 -poison_type=badnet -poison_rate=0.1 -validation_per_class=10 -num_test_samples=50 
# echo "TED | cifar10 - badnet | COMPLETE"
# python other_defense.py -defense=TEDPLUS -dataset=cifar10 -poison_type=badnet -poison_rate=0.1 -validation_per_class=10 -num_test_samples=50 -num_neighbors=3
# echo "TEDPLUS | cifar10 - badnet | COMPLETE"
# echo "cifar10 - badnet | COMPLETE"

# echo "cifar10 - blend | START"
# # python create_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.1
# # python train_on_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.1
# # python test_model.py -dataset=cifar10 -poison_type=blend -poison_rate=0.1
# # # python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=blend -poison_rate=0.1
# python other_defense.py -defense=TED -dataset=cifar10 -poison_type=blend -poison_rate=0.1 -validation_per_class=10 -num_test_samples=50 
# echo "TED | cifar10 - blend | COMPLETE"
# python other_defense.py -defense=TEDPLUS -dataset=cifar10 -poison_type=blend -poison_rate=0.1 -validation_per_class=10 -num_test_samples=50 -num_neighbors=3
# echo "TEDPLUS | cifar10 - blend | COMPLETE"
# echo "cifar10 - blend | COMPLETE"

# echo "cifar10 - adaptive_patch | START"
# # python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# # python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# # python test_model.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# # python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# python other_defense.py -defense=TED -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=10 -num_test_samples=50 
# echo "TED | cifar10 - adaptive_patch | COMPLETE"
# python other_defense.py -defense=TEDPLUS -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=10 -num_test_samples=50 -num_neighbors=3
# echo "TEDPLUS | cifar10 - adaptive_patch | COMPLETE"
# echo "cifar10 - adaptive_patch | COMPLETE"

# echo "cifar10 - adaptive_blend | START"
# # python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# # python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# # python test_model.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# # python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# python other_defense.py -defense=TED -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=10 -num_test_samples=50 
# echo "TED | cifar10 - adaptive_blend | COMPLETE"
# python other_defense.py -defense=TEDPLUS -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=10 -num_test_samples=50 -num_neighbors=3
# echo "TEDPLUS | cifar10 - adaptive_blend | COMPLETE"
# echo "cifar10 - adaptive_blend | COMPLETE"

# echo "cifar10 - WaNet | START"
# # python create_poisoned_set.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
# # python train_on_poisoned_set.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
# # python test_model.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
# # python all_layers_resnet18_layer_visualize.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05 -data_ratio=0.4
# # python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
# python other_defense.py -defense=TED -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=10 -num_test_samples=50 
# echo "TED | cifar10 - WaNet | COMPLETE"
# python other_defense.py -defense=TEDPLUS -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=10 -num_test_samples=50 -num_neighbors=3
# echo "TEDPLUS | cifar10 - WaNet | COMPLETE"
# echo "cifar10 - WaNet | COMPLETE"

# echo "cifar10 - trojan | START"
# # python create_poisoned_set.py -dataset=cifar10 -poison_type=trojan -poison_rate=0.1
# # python train_on_poisoned_set.py -dataset=cifar10 -poison_type=trojan -poison_rate=0.1
# # python test_model.py -dataset=cifar10 -poison_type=trojan -poison_rate=0.1
# # python all_layers_resnet18_layer_visualize.py -dataset=cifar10 -poison_type=trojan -poison_rate=0.1 -data_ratio=0.4
# python other_defense.py -defense=TED -dataset=cifar10 -poison_type=trojan -poison_rate=0.1 -validation_per_class=10 -num_test_samples=50  
# echo "TED | cifar10 - trojan | COMPLETE"
# python other_defense.py -defense=TEDPLUS -dataset=cifar10 -poison_type=trojan -poison_rate=0.1 -validation_per_class=10 -num_test_samples=50 -num_neighbors=3
# echo "TEDPLUS | cifar10 - trojan | COMPLETE"
# echo "cifar10 - trojan | COMPLETE"

# echo "cifar10 - dynamic | START"
# # python create_poisoned_set.py -dataset=cifar10 -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
# # python train_on_poisoned_set.py -dataset=cifar10 -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
# # python test_model.py -dataset=cifar10 -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
# # python all_layers_resnet18_layer_visualize.py -dataset=cifar10 -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05 -data_ratio=0.4
# # python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
# python other_defense.py -defense=TED -dataset=cifar10 -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=10 -num_test_samples=50 
# echo "TED | cifar10 - dynamic | COMPLETE"
# python other_defense.py -defense=TEDPLUS -dataset=cifar10 -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=10 -num_test_samples=50 -num_neighbors=3
# echo "TEDPLUS | cifar10 - dynamic | COMPLETE"
# echo "cifar10 - dynamic | COMPLETE"


# echo "cifar10 - TaCT | START"
# # python create_poisoned_set.py -dataset=cifar10 -poison_type=TaCT -poison_rate=0.01 -cover_rate=0.005
# # python train_on_poisoned_set.py -dataset=cifar10 -poison_type=TaCT -poison_rate=0.01 -cover_rate=0.005
# # python test_model.py -dataset=cifar10 -poison_type=TaCT -poison_rate=0.02 -cover_rate=0.01
# # python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
# # python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=TaCT -poison_rate=0.02 -cover_rate=0.01
# # python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=TaCT -poison_rate=0.01
# # python all_layers_resnet18_layer_visualize.py -dataset=cifar10 -poison_type=TaCT -poison_rate=0.02 -cover_rate=0.01 -data_ratio=0.3
# python other_defense.py -defense=TED -dataset=cifar10 -poison_type=TaCT -poison_rate=0.02 -cover_rate=0.01 -validation_per_class=10 -num_test_samples=50 
# echo "TED | cifar10 - TaCT | COMPLETE"
# python other_defense.py -defense=TEDPLUS -dataset=cifar10 -poison_type=TaCT -poison_rate=0.02 -cover_rate=0.01 -validation_per_class=10 -num_test_samples=50 -num_neighbors=3
# echo "TEDPLUS | cifar10 - TaCT | COMPLETE"
# echo "cifar10 - TaCT | COMPLETE"

# echo "cifar10 - SSDT | START"
# # python train_SSDT.py --dataset gtsrb --attack_mode SSDT --n_iters 200
# python other_defense.py -defense=TED -dataset=cifar10 -poison_type=SSDT -validation_per_class=10 -num_test_samples=50 
# echo "TED - complete"
# python other_defense.py -defense=TEDPLUS -dataset=cifar10 -poison_type=SSDT -validation_per_class=10 -num_test_samples=50 -num_neighbors=3
# echo "TEDPLUS - complete"
# echo "cifar10 - SSDT | COMPLETE"


# echo "gtsrb - badnet | START"
# # python create_poisoned_set.py -dataset=gtsrb -poison_type=badnet -poison_rate=0.1
# # python train_on_poisoned_set.py -dataset=gtsrb -poison_type=badnet -poison_rate=0.1
# # python test_model.py -dataset=gtsrb -poison_type=badnet -poison_rate=0.1
# # # python other_defense.py -defense=IBD_PSC -dataset=gtsrb -poison_type=badnet -poison_rate=0.1
# python other_defense.py -defense=TED -dataset=gtsrb -poison_type=badnet -poison_rate=0.1 -validation_per_class=10 -num_test_samples=50 
# echo "TED | gtsrb - badnet | COMPLETE"
# python other_defense.py -defense=TEDPLUS -dataset=gtsrb -poison_type=badnet -poison_rate=0.1 -validation_per_class=10 -num_test_samples=50 -num_neighbors=3
# echo "TEDPLUS | gtsrb - badnet | COMPLETE"
# echo "gtsrb - badnet | COMPLETE"

# echo "gtsrb - blend | START"
# # python create_poisoned_set.py -dataset=gtsrb -poison_type=blend -poison_rate=0.1
# # python train_on_poisoned_set.py -dataset=gtsrb -poison_type=blend -poison_rate=0.1
# # python test_model.py -dataset=gtsrb -poison_type=blend -poison_rate=0.1
# # # python other_defense.py -defense=IBD_PSC -dataset=gtsrb -poison_type=blend -poison_rate=0.1
# python other_defense.py -defense=TED -dataset=gtsrb -poison_type=blend -poison_rate=0.1 -validation_per_class=10 -num_test_samples=50 
# echo "TED | gtsrb - blend | COMPLETE"
# python other_defense.py -defense=TEDPLUS -dataset=gtsrb -poison_type=blend -poison_rate=0.1 -validation_per_class=10 -num_test_samples=50 -num_neighbors=3
# echo "TEDPLUS | gtsrb - blend | COMPLETE"
# echo "gtsrb - blend | COMPLETE"

# echo "gtsrb - adaptive_patch | START"
# # python create_poisoned_set.py -dataset=gtsrb -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# # python train_on_poisoned_set.py -dataset=gtsrb -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# # python test_model.py -dataset=gtsrb -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# # python other_defense.py -defense=IBD_PSC -dataset=gtsrb -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# python other_defense.py -defense=TED -dataset=gtsrb -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=10 -num_test_samples=50 
# echo "TED | gtsrb - adaptive_patch | COMPLETE"
# python other_defense.py -defense=TEDPLUS -dataset=gtsrb -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=10 -num_test_samples=50 -num_neighbors=3
# echo "TEDPLUS | gtsrb - adaptive_patch | COMPLETE"
# echo "gtsrb - adaptive_patch | COMPLETE"

# echo "gtsrb - adaptive_blend | START"
# # python create_poisoned_set.py -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# # python train_on_poisoned_set.py -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# # python test_model.py -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# # python other_defense.py -defense=IBD_PSC -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# python other_defense.py -defense=TED -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=10 -num_test_samples=50 
# echo "TED | gtsrb - adaptive_blend | COMPLETE"
# python other_defense.py -defense=TEDPLUS -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=10 -num_test_samples=50 -num_neighbors=1
# echo "TEDPLUS | gtsrb - adaptive_blend | COMPLETE"
# echo "gtsrb - adaptive_blend | COMPLETE"

# echo "gtsrb - WaNet | START"
# # python create_poisoned_set.py -dataset=gtsrb -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
# # python train_on_poisoned_set.py -dataset=gtsrb -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
# # python test_model.py -dataset=gtsrb -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
# # python all_layers_resnet18_layer_visualize.py -dataset=gtsrb -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05 -data_ratio=0.4
# # python other_defense.py -defense=IBD_PSC -dataset=gtsrb -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
# python other_defense.py -defense=TED -dataset=gtsrb -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=10 -num_test_samples=50 
# echo "TED | gtsrb - WaNet | COMPLETE"
# python other_defense.py -defense=TEDPLUS -dataset=gtsrb -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=10 -num_test_samples=50 -num_neighbors=3
# echo "TEDPLUS | gtsrb - WaNet | COMPLETE"
# echo "gtsrb - WaNet | COMPLETE"

# echo "gtsrb - trojan | START"
# # python create_poisoned_set.py -dataset=gtsrb -poison_type=trojan -poison_rate=0.1
# # python train_on_poisoned_set.py -dataset=gtsrb -poison_type=trojan -poison_rate=0.1
# # python test_model.py -dataset=gtsrb -poison_type=trojan -poison_rate=0.1
# # python all_layers_resnet18_layer_visualize.py -dataset=gtsrb -poison_type=trojan -poison_rate=0.1 -data_ratio=0.4
# python other_defense.py -defense=TED -dataset=gtsrb -poison_type=trojan -poison_rate=0.1 -validation_per_class=10 -num_test_samples=50  
# echo "TED | gtsrb - trojan | COMPLETE"
# python other_defense.py -defense=TEDPLUS -dataset=gtsrb -poison_type=trojan -poison_rate=0.1 -validation_per_class=10 -num_test_samples=50 -num_neighbors=3
# echo "TEDPLUS | gtsrb - trojan | COMPLETE"
# echo "gtsrb - trojan | COMPLETE"

# echo "gtsrb - dynamic | START"
# # python create_poisoned_set.py -dataset=gtsrb -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
# # python train_on_poisoned_set.py -dataset=gtsrb -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
# # python test_model.py -dataset=gtsrb -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
# # python all_layers_resnet18_layer_visualize.py -dataset=gtsrb -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05 -data_ratio=0.4
# # python other_defense.py -defense=IBD_PSC -dataset=gtsrb -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
# python other_defense.py -defense=TED -dataset=gtsrb -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=10 -num_test_samples=50 
# echo "TED | gtsrb - dynamic | COMPLETE"
# python other_defense.py -defense=TEDPLUS -dataset=gtsrb -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=10 -num_test_samples=50 -num_neighbors=3
# echo "TEDPLUS | gtsrb - dynamic | COMPLETE"
# echo "gtsrb - dynamic | COMPLETE"

# echo "gtsrb - TaCT | START"
# python other_defense.py -defense=TED -dataset=gtsrb -poison_type=TaCT -poison_rate=0.02 -cover_rate=0.01 -validation_per_class=10 -num_test_samples=50 
# echo "TED | gtsrb - TaCT | COMPLETE"
# python other_defense.py -defense=TEDPLUS -dataset=gtsrb -poison_type=TaCT -poison_rate=0.02 -cover_rate=0.01 -validation_per_class=10 -num_test_samples=50 -num_neighbors=3
# echo "TEDPLUS | gtsrb - TaCT | COMPLETE"
# echo "gtsrb - TaCT | COMPLETE"


# echo "gtsrb - SSDT | START"
# # # python train_SSDT.py --dataset gtsrb --attack_mode SSDT --n_iters 1000
# python other_defense.py -defense=TED -dataset=gtsrb -poison_type=SSDT -validation_per_class=10 -num_test_samples=50 
# echo "TED - complete"
# python other_defense.py -defense=TEDPLUS -dataset=gtsrb -poison_type=SSDT -validation_per_class=10 -num_test_samples=50 -num_neighbors=3
# echo "TEDPLUS - complete"
# echo "gtsrb - SSDT | COMPLETE"


# echo "====================================="


# echo "cifar10 - badnet | START"
# # python create_poisoned_set.py -dataset=cifar10 -poison_type=badnet -poison_rate='0.1'
# # python train_on_poisoned_set.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
# # python test_model.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
# # # python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
# python other_defense.py -defense=TED -dataset=cifar10 -poison_type=badnet -poison_rate=0.1 -validation_per_class=5 -num_test_samples=50 
# echo "TED | cifar10 - badnet | COMPLETE"
# python other_defense.py -defense=TEDPLUS -dataset=cifar10 -poison_type=badnet -poison_rate=0.1 -validation_per_class=5 -num_test_samples=50 -num_neighbors=3
# echo "TEDPLUS | cifar10 - badnet | COMPLETE"
# echo "cifar10 - badnet | COMPLETE"

# echo "cifar10 - blend | START"
# # python create_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.1
# # python train_on_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.1
# # python test_model.py -dataset=cifar10 -poison_type=blend -poison_rate=0.1
# # # python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=blend -poison_rate=0.1
# python other_defense.py -defense=TED -dataset=cifar10 -poison_type=blend -poison_rate=0.1 -validation_per_class=5 -num_test_samples=50 
# echo "TED | cifar10 - blend | COMPLETE"
# python other_defense.py -defense=TEDPLUS -dataset=cifar10 -poison_type=blend -poison_rate=0.1 -validation_per_class=5 -num_test_samples=50 -num_neighbors=3
# echo "TEDPLUS | cifar10 - blend | COMPLETE"
# echo "cifar10 - blend | COMPLETE"

# echo "cifar10 - adaptive_patch | START"
# # python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# # python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# # python test_model.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# # python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# python other_defense.py -defense=TED -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=5 -num_test_samples=50 
# echo "TED | cifar10 - adaptive_patch | COMPLETE"
# python other_defense.py -defense=TEDPLUS -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=5 -num_test_samples=50 -num_neighbors=3
# echo "TEDPLUS | cifar10 - adaptive_patch | COMPLETE"
# echo "cifar10 - adaptive_patch | COMPLETE"

# echo "cifar10 - adaptive_blend | START"
# # python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# # python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# # python test_model.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# # python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# python other_defense.py -defense=TED -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=5 -num_test_samples=50 
# echo "TED | cifar10 - adaptive_blend | COMPLETE"
# python other_defense.py -defense=TEDPLUS -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=5 -num_test_samples=50 -num_neighbors=3
# echo "TEDPLUS | cifar10 - adaptive_blend | COMPLETE"
# echo "cifar10 - adaptive_blend | COMPLETE"

# echo "cifar10 - WaNet | START"
# # python create_poisoned_set.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
# # python train_on_poisoned_set.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
# # python test_model.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
# # python all_layers_resnet18_layer_visualize.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05 -data_ratio=0.4
# # python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
# python other_defense.py -defense=TED -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=5 -num_test_samples=50 
# echo "TED | cifar10 - WaNet | COMPLETE"
# python other_defense.py -defense=TEDPLUS -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=5 -num_test_samples=50 -num_neighbors=3
# echo "TEDPLUS | cifar10 - WaNet | COMPLETE"
# echo "cifar10 - WaNet | COMPLETE"

# echo "cifar10 - trojan | START"
# # python create_poisoned_set.py -dataset=cifar10 -poison_type=trojan -poison_rate=0.1
# # python train_on_poisoned_set.py -dataset=cifar10 -poison_type=trojan -poison_rate=0.1
# # python test_model.py -dataset=cifar10 -poison_type=trojan -poison_rate=0.1
# # python all_layers_resnet18_layer_visualize.py -dataset=cifar10 -poison_type=trojan -poison_rate=0.1 -data_ratio=0.4
# python other_defense.py -defense=TED -dataset=cifar10 -poison_type=trojan -poison_rate=0.1 -validation_per_class=5 -num_test_samples=50  
# echo "TED | cifar10 - trojan | COMPLETE"
# python other_defense.py -defense=TEDPLUS -dataset=cifar10 -poison_type=trojan -poison_rate=0.1 -validation_per_class=5 -num_test_samples=50 -num_neighbors=3
# echo "TEDPLUS | cifar10 - trojan | COMPLETE"
# echo "cifar10 - trojan | COMPLETE"

# echo "cifar10 - dynamic | START"
# # python create_poisoned_set.py -dataset=cifar10 -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
# # python train_on_poisoned_set.py -dataset=cifar10 -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
# # python test_model.py -dataset=cifar10 -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
# # python all_layers_resnet18_layer_visualize.py -dataset=cifar10 -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05 -data_ratio=0.4
# # python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
# python other_defense.py -defense=TED -dataset=cifar10 -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=5 -num_test_samples=50 
# echo "TED | cifar10 - dynamic | COMPLETE"
# python other_defense.py -defense=TEDPLUS -dataset=cifar10 -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=5 -num_test_samples=50 -num_neighbors=3
# echo "TEDPLUS | cifar10 - dynamic | COMPLETE"
# echo "cifar10 - dynamic | COMPLETE"


# echo "cifar10 - TaCT | START"
# # python create_poisoned_set.py -dataset=cifar10 -poison_type=TaCT -poison_rate=0.01 -cover_rate=0.005
# # python train_on_poisoned_set.py -dataset=cifar10 -poison_type=TaCT -poison_rate=0.01 -cover_rate=0.005
# # python test_model.py -dataset=cifar10 -poison_type=TaCT -poison_rate=0.02 -cover_rate=0.01
# # python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
# # python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=TaCT -poison_rate=0.02 -cover_rate=0.01
# # python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=TaCT -poison_rate=0.01
# # python all_layers_resnet18_layer_visualize.py -dataset=cifar10 -poison_type=TaCT -poison_rate=0.02 -cover_rate=0.01 -data_ratio=0.3
# python other_defense.py -defense=TED -dataset=cifar10 -poison_type=TaCT -poison_rate=0.02 -cover_rate=0.01 -validation_per_class=5 -num_test_samples=50 
# echo "TED | cifar10 - TaCT | COMPLETE"
# python other_defense.py -defense=TEDPLUS -dataset=cifar10 -poison_type=TaCT -poison_rate=0.02 -cover_rate=0.01 -validation_per_class=5 -num_test_samples=50 -num_neighbors=3
# echo "TEDPLUS | cifar10 - TaCT | COMPLETE"
# echo "cifar10 - TaCT | COMPLETE"

# echo "cifar10 - SSDT | START"
# # python train_SSDT.py --dataset gtsrb --attack_mode SSDT --n_iters 200
# python other_defense.py -defense=TED -dataset=cifar10 -poison_type=SSDT -validation_per_class=5 -num_test_samples=50 
# echo "TED - complete"
# python other_defense.py -defense=TEDPLUS -dataset=cifar10 -poison_type=SSDT -validation_per_class=5 -num_test_samples=50 -num_neighbors=3
# echo "TEDPLUS - complete"
# echo "cifar10 - SSDT | COMPLETE"


# echo "gtsrb - badnet | START"
# # python create_poisoned_set.py -dataset=gtsrb -poison_type=badnet -poison_rate=0.1
# # python train_on_poisoned_set.py -dataset=gtsrb -poison_type=badnet -poison_rate=0.1
# # python test_model.py -dataset=gtsrb -poison_type=badnet -poison_rate=0.1
# # # python other_defense.py -defense=IBD_PSC -dataset=gtsrb -poison_type=badnet -poison_rate=0.1
# python other_defense.py -defense=TED -dataset=gtsrb -poison_type=badnet -poison_rate=0.1 -validation_per_class=5 -num_test_samples=50 
# echo "TED | gtsrb - badnet | COMPLETE"
# python other_defense.py -defense=TEDPLUS -dataset=gtsrb -poison_type=badnet -poison_rate=0.1 -validation_per_class=5 -num_test_samples=50 -num_neighbors=3
# echo "TEDPLUS | gtsrb - badnet | COMPLETE"
# echo "gtsrb - badnet | COMPLETE"

# echo "gtsrb - blend | START"
# # python create_poisoned_set.py -dataset=gtsrb -poison_type=blend -poison_rate=0.1
# # python train_on_poisoned_set.py -dataset=gtsrb -poison_type=blend -poison_rate=0.1
# # python test_model.py -dataset=gtsrb -poison_type=blend -poison_rate=0.1
# # # python other_defense.py -defense=IBD_PSC -dataset=gtsrb -poison_type=blend -poison_rate=0.1
# python other_defense.py -defense=TED -dataset=gtsrb -poison_type=blend -poison_rate=0.1 -validation_per_class=5 -num_test_samples=50 
# echo "TED | gtsrb - blend | COMPLETE"
# python other_defense.py -defense=TEDPLUS -dataset=gtsrb -poison_type=blend -poison_rate=0.1 -validation_per_class=5 -num_test_samples=50 -num_neighbors=3
# echo "TEDPLUS | gtsrb - blend | COMPLETE"
# echo "gtsrb - blend | COMPLETE"

# echo "gtsrb - adaptive_patch | START"
# # python create_poisoned_set.py -dataset=gtsrb -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# # python train_on_poisoned_set.py -dataset=gtsrb -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# # python test_model.py -dataset=gtsrb -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# # python other_defense.py -defense=IBD_PSC -dataset=gtsrb -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05
# python other_defense.py -defense=TED -dataset=gtsrb -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=5 -num_test_samples=50 
# echo "TED | gtsrb - adaptive_patch | COMPLETE"
# python other_defense.py -defense=TEDPLUS -dataset=gtsrb -poison_type=adaptive_patch -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=5 -num_test_samples=50 -num_neighbors=3
# echo "TEDPLUS | gtsrb - adaptive_patch | COMPLETE"
# echo "gtsrb - adaptive_patch | COMPLETE"

# echo "gtsrb - adaptive_blend | START"
# # python create_poisoned_set.py -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# # python train_on_poisoned_set.py -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# # python test_model.py -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# # python other_defense.py -defense=IBD_PSC -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05
# python other_defense.py -defense=TED -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=5 -num_test_samples=50 
# echo "TED | gtsrb - adaptive_blend | COMPLETE"
# python other_defense.py -defense=TEDPLUS -dataset=gtsrb -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=5 -num_test_samples=50 -num_neighbors=1
# echo "TEDPLUS | gtsrb - adaptive_blend | COMPLETE"
# echo "gtsrb - adaptive_blend | COMPLETE"

# echo "gtsrb - WaNet | START"
# # python create_poisoned_set.py -dataset=gtsrb -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
# # python train_on_poisoned_set.py -dataset=gtsrb -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
# # python test_model.py -dataset=gtsrb -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
# # python all_layers_resnet18_layer_visualize.py -dataset=gtsrb -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05 -data_ratio=0.4
# # python other_defense.py -defense=IBD_PSC -dataset=gtsrb -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05
# python other_defense.py -defense=TED -dataset=gtsrb -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=5 -num_test_samples=50 
# echo "TED | gtsrb - WaNet | COMPLETE"
# python other_defense.py -defense=TEDPLUS -dataset=gtsrb -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=5 -num_test_samples=50 -num_neighbors=3
# echo "TEDPLUS | gtsrb - WaNet | COMPLETE"
# echo "gtsrb - WaNet | COMPLETE"

# echo "gtsrb - trojan | START"
# # python create_poisoned_set.py -dataset=gtsrb -poison_type=trojan -poison_rate=0.1
# # python train_on_poisoned_set.py -dataset=gtsrb -poison_type=trojan -poison_rate=0.1
# # python test_model.py -dataset=gtsrb -poison_type=trojan -poison_rate=0.1
# # python all_layers_resnet18_layer_visualize.py -dataset=gtsrb -poison_type=trojan -poison_rate=0.1 -data_ratio=0.4
# python other_defense.py -defense=TED -dataset=gtsrb -poison_type=trojan -poison_rate=0.1 -validation_per_class=5 -num_test_samples=50  
# echo "TED | gtsrb - trojan | COMPLETE"
# python other_defense.py -defense=TEDPLUS -dataset=gtsrb -poison_type=trojan -poison_rate=0.1 -validation_per_class=5 -num_test_samples=50 -num_neighbors=3
# echo "TEDPLUS | gtsrb - trojan | COMPLETE"
# echo "gtsrb - trojan | COMPLETE"

# echo "gtsrb - dynamic | START"
# # python create_poisoned_set.py -dataset=gtsrb -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
# # python train_on_poisoned_set.py -dataset=gtsrb -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
# # python test_model.py -dataset=gtsrb -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
# # python all_layers_resnet18_layer_visualize.py -dataset=gtsrb -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05 -data_ratio=0.4
# # python other_defense.py -defense=IBD_PSC -dataset=gtsrb -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05
# python other_defense.py -defense=TED -dataset=gtsrb -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=5 -num_test_samples=50 
# echo "TED | gtsrb - dynamic | COMPLETE"
# python other_defense.py -defense=TEDPLUS -dataset=gtsrb -poison_type=dynamic -poison_rate=0.1 -cover_rate=0.05 -validation_per_class=5 -num_test_samples=50 -num_neighbors=3
# echo "TEDPLUS | gtsrb - dynamic | COMPLETE"
# echo "gtsrb - dynamic | COMPLETE"

# echo "gtsrb - TaCT | START"
# python other_defense.py -defense=TED -dataset=gtsrb -poison_type=TaCT -poison_rate=0.02 -cover_rate=0.01 -validation_per_class=5 -num_test_samples=50 
# echo "TED | gtsrb - TaCT | COMPLETE"
# python other_defense.py -defense=TEDPLUS -dataset=gtsrb -poison_type=TaCT -poison_rate=0.02 -cover_rate=0.01 -validation_per_class=5 -num_test_samples=50 -num_neighbors=3
# echo "TEDPLUS | gtsrb - TaCT | COMPLETE"
# echo "gtsrb - TaCT | COMPLETE"

# echo "gtsrb - SSDT | START"
# # # python train_SSDT.py --dataset gtsrb --attack_mode SSDT --n_iters 1000
# python other_defense.py -defense=TED -dataset=gtsrb -poison_type=SSDT -validation_per_class=5 -num_test_samples=50 
# echo "TED - complete"
# python other_defense.py -defense=TEDPLUS -dataset=gtsrb -poison_type=SSDT -validation_per_class=5 -num_test_samples=50 -num_neighbors=3
# echo "TEDPLUS - complete"
# echo "gtsrb - SSDT | COMPLETE"


echo "All tasks completed successfully."








