#!/bin/bash
#SBATCH --job-name=cifar10_adaptive_blend_viz              # Job name
#SBATCH --output=visualization_scripts/cifar10_adaptive_blend_viz_output.log  # Output file
#SBATCH --error=visualization_scripts/cifar10_adaptive_blend_viz_error.log    # Error file
#SBATCH --partition=gpu                                   # Use GPU partition
#SBATCH --gres=gpu:a100:1                                 # Request 1 A100 GPU
#SBATCH --time=04:00:00                                   # Max runtime

# Move to the directory containing the environment
cd "/home/s222576762/BackdoorBox Research/backdoor-toolbox"
source "my_env/bin/activate"

# Step 4: Visualize the model's latent space for specified methods
echo "Visualizing the model's latent space for cifar10 with adaptive_blend..."

# Core visualizations
python visualize.py -method=pairwise -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01
python visualize.py -method=parallel_coordinates -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01
python visualize.py -method=heatmap -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01
python visualize.py -method=silhouette -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01
python visualize.py -method=violin -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01
python visualize.py -method=lof -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01
python visualize.py -method=dbscan -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01

# Additional dimensionality reduction and analysis methods
python visualize.py -method=pca -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01
python visualize.py -method=tsne -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01
python visualize.py -method=umap -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01
python visualize.py -method=oracle -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01
python visualize.py -method=mean_diff -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01
python visualize.py -method=SS -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01
python visualize.py -method=isomap -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01
python visualize.py -method=lle -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01
python visualize.py -method=kpca -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01
python visualize.py -method=spectral -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01

echo "Visualization for cifar10 with adaptive_blend completed."
