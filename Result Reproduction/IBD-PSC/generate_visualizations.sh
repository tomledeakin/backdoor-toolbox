#!/bin/bash

# Define dataset and attack type
dataset="cifar10"
attack="adaptive_blend"

# Create the visualization_scripts directory if it doesn't exist
mkdir -p visualization_scripts

# Define script filename
script_file="visualization_scripts/${dataset}_${attack}_visualizations.sh"

# Create the script file with output and error logs in the same directory
cat <<EOT > "$script_file"
#!/bin/bash
#SBATCH --job-name=${dataset}_${attack}_viz              # Job name
#SBATCH --output=visualization_scripts/${dataset}_${attack}_viz_output.log  # Output file
#SBATCH --error=visualization_scripts/${dataset}_${attack}_viz_error.log    # Error file
#SBATCH --partition=gpu                                   # Use GPU partition
#SBATCH --gres=gpu:a100:1                                 # Request 1 A100 GPU
#SBATCH --time=04:00:00                                   # Max runtime

# Move to the directory containing the environment
cd "$HOME/BackdoorBox Research/backdoor-toolbox"
source "my_env/bin/activate"

# Step 4: Visualize the model's latent space for specified methods
echo "Visualizing the model's latent space for ${dataset} with ${attack}..."

# Core visualizations
python visualize.py -method=pairwise -dataset=$dataset -poison_type=$attack -poison_rate=0.01
python visualize.py -method=parallel_coordinates -dataset=$dataset -poison_type=$attack -poison_rate=0.01
python visualize.py -method=heatmap -dataset=$dataset -poison_type=$attack -poison_rate=0.01
python visualize.py -method=silhouette -dataset=$dataset -poison_type=$attack -poison_rate=0.01
python visualize.py -method=violin -dataset=$dataset -poison_type=$attack -poison_rate=0.01
python visualize.py -method=lof -dataset=$dataset -poison_type=$attack -poison_rate=0.01
python visualize.py -method=dbscan -dataset=$dataset -poison_type=$attack -poison_rate=0.01

# Additional dimensionality reduction and analysis methods
python visualize.py -method=pca -dataset=$dataset -poison_type=$attack -poison_rate=0.01
python visualize.py -method=tsne -dataset=$dataset -poison_type=$attack -poison_rate=0.01
python visualize.py -method=umap -dataset=$dataset -poison_type=$attack -poison_rate=0.01
python visualize.py -method=oracle -dataset=$dataset -poison_type=$attack -poison_rate=0.01
python visualize.py -method=mean_diff -dataset=$dataset -poison_type=$attack -poison_rate=0.01
python visualize.py -method=SS -dataset=$dataset -poison_type=$attack -poison_rate=0.01
python visualize.py -method=isomap -dataset=$dataset -poison_type=$attack -poison_rate=0.01
python visualize.py -method=lle -dataset=$dataset -poison_type=$attack -poison_rate=0.01
python visualize.py -method=kpca -dataset=$dataset -poison_type=$attack -poison_rate=0.01
python visualize.py -method=spectral -dataset=$dataset -poison_type=$attack -poison_rate=0.01

echo "Visualization for $dataset with $attack completed."
EOT

# Make the script executable
chmod +x "$script_file"

echo "Script generation for visualization completed."


