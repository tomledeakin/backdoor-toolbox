#!/bin/bash

# Create a logs directory if it doesn't exist
mkdir -p logs

# Declare datasets and poison types
datasets=("gtsrb" "cifar10")
poison_types=("badnet" "blend" "adaptive_blend" "adaptive_patch")
poison_rate="0.1"
cover_rate="0.05"  # Only for adaptive_blend and adaptive_patch

# Loop through each dataset and poison type
for dataset in "${datasets[@]}"; do
    for poison_type in "${poison_types[@]}"; do
        # Create a unique script name
        script_name="logs/${dataset}_${poison_type}_script.sh"

        # Create the script for the specific dataset and poison type
        echo "#!/bin/bash" > "$script_name"
        echo "#SBATCH --job-name=${dataset}_${poison_type}" >> "$script_name"
        echo "#SBATCH --output=logs/${dataset}_${poison_type}_output.log" >> "$script_name"
        echo "#SBATCH --error=logs/${dataset}_${poison_type}_error.log" >> "$script_name"
        echo "#SBATCH --partition=gpu" >> "$script_name"
        echo "#SBATCH --gres=gpu:a100:1" >> "$script_name"
        echo "#SBATCH --time=48:00:00" >> "$script_name"
        echo "#SBATCH --mail-user=tomledeakin@gmail.com" >> "$script_name"
        echo "#SBATCH --mail-type=END,FAIL" >> "$script_name"
        echo "" >> "$script_name"
        echo "cd \"$HOME/BackdoorBox Research/backdoor-toolbox\" || { echo 'Directory not found'; exit 1; }" >> "$script_name"
        echo "source \"my_env/bin/activate\"" >> "$script_name"
        echo "" >> "$script_name"

        # Add the commands based on poison type
        if [[ "$poison_type" == "adaptive_blend" || "$poison_type" == "adaptive_patch" ]]; then
            echo "python create_poisoned_set.py -dataset=$dataset -poison_type=$poison_type -poison_rate=$poison_rate -cover_rate=$cover_rate" >> "$script_name"
            echo "python train_on_poisoned_set.py -dataset=$dataset -poison_type=$poison_type -poison_rate=$poison_rate -cover_rate=$cover_rate" >> "$script_name"
            echo "python test_model.py -dataset=$dataset -poison_type=$poison_type -poison_rate=$poison_rate -cover_rate=$cover_rate" >> "$script_name"
            echo "# python resnet18_layer_visualize.py -dataset=$dataset -poison_type=$poison_type -poison_rate=$poison_rate -cover_rate=$cover_rate -n_neighbors=50" >> "$script_name"
            echo "python other_defense.py -defense=IBD_PSC -dataset=$dataset -poison_type=$poison_type -poison_rate=$poison_rate -cover_rate=$cover_rate" >> "$script_name"
        else
            echo "python create_poisoned_set.py -dataset=$dataset -poison_type=$poison_type -poison_rate=$poison_rate" >> "$script_name"
            echo "python train_on_poisoned_set.py -dataset=$dataset -poison_type=$poison_type -poison_rate=$poison_rate" >> "$script_name"
            echo "python test_model.py -dataset=$dataset -poison_type=$poison_type -poison_rate=$poison_rate" >> "$script_name"
            echo "# python resnet18_layer_visualize.py -dataset=$dataset -poison_type=$poison_type -poison_rate=$poison_rate" >> "$script_name"
            echo "python other_defense.py -defense=IBD_PSC -dataset=$dataset -poison_type=$poison_type -poison_rate=$poison_rate" >> "$script_name"
        fi

        echo "echo \"${dataset} - ${poison_type} | COMPLETE\"" >> "$script_name"

        # Make the script executable
        chmod +x "$script_name"
    done
done

# Submit all the generated scripts
for script in logs/*_script.sh; do
    sbatch "$script"
done

echo "All scripts have been generated and submitted successfully."

