import random
import numpy as np
import torch
import os
from torchvision import transforms
import argparse
from torch import nn
from utils import supervisor, tools, default_args
import config
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding
from matplotlib import pyplot as plt
from sklearn import svm
from umap import UMAP
import seaborn as sns
import pandas as pd

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-method', type=str, required=False, default='pca',
                    choices=['pca', 'tsne', 'umap', 'oracle', 'mean_diff', 'SS',
                             'isomap', 'lle', 'kpca', 'spectral'])
parser.add_argument('-dataset', type=str, required=False, default=default_args.parser_default['dataset'],
                    choices=default_args.parser_choices['dataset'])
parser.add_argument('-poison_type', type=str,  required=True,
                    choices=default_args.parser_choices['poison_type'])
parser.add_argument('-poison_rate', type=float,  required=False,
                    choices=default_args.parser_choices['poison_rate'],
                    default=default_args.parser_default['poison_rate'])
parser.add_argument('-cover_rate', type=float,  required=False,
                    choices=default_args.parser_choices['cover_rate'],
                    default=default_args.parser_default['cover_rate'])
parser.add_argument('-alpha', type=float,  required=False, default=default_args.parser_default['alpha'])
parser.add_argument('-test_alpha', type=float,  required=False, default=None)
parser.add_argument('-trigger', type=str,  required=False,
                    default=None)
parser.add_argument('-no_aug', default=False, action='store_true')
parser.add_argument('-model', type=str, required=False, default=None)
parser.add_argument('-model_path', required=False, default=None)
parser.add_argument('-no_normalize', default=False, action='store_true')
parser.add_argument('-devices', type=str, default='0')
parser.add_argument('-target_class', type=int, default=-1)
parser.add_argument('-seed', type=int, required=False, default=default_args.seed)

args = parser.parse_args()

# Set CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices
tools.setup_seed(args.seed)

# Determine the target class
if args.target_class == -1:
    target_class = config.target_class[args.dataset]
else:
    target_class = args.target_class

# Set trigger if not specified
if args.trigger is None:
    args.trigger = config.trigger_default[args.dataset][args.poison_type]

batch_size = 128
kwargs = {'num_workers': 4, 'pin_memory': True}

# Define visualizer classes for specific methods
class MeanDiffVisualizer:

    def fit_transform(self, clean, poison):
        clean_mean = clean.mean(dim=0)
        poison_mean = poison.mean(dim=0)
        mean_diff = poison_mean - clean_mean
        print("Mean L2 distance between poison and clean:", torch.norm(mean_diff, p=2).item())

        proj_clean_mean = torch.matmul(clean, mean_diff)
        proj_poison_mean = torch.matmul(poison, mean_diff)

        return proj_clean_mean, proj_poison_mean


class OracleVisualizer:

    def __init__(self):
        self.clf = svm.LinearSVC()

    def fit_transform(self, clean, poison):

        clean = clean.numpy()
        num_clean = len(clean)

        poison = poison.numpy()
        num_poison = len(poison)

        X = np.concatenate([clean, poison], axis=0)
        y = [0] * num_clean + [1] * num_poison

        self.clf.fit(X, y)
        print("SVM Accuracy:", self.clf.score(X, y))

        norm = np.linalg.norm(self.clf.coef_)
        self.clf.coef_ = self.clf.coef_ / norm
        self.clf.intercept_ = self.clf.intercept_ / norm

        projection = self.clf.decision_function(X)

        return projection[:num_clean], projection[num_clean:]


class SpectralVisualizer:

    def fit_transform(self, clean, poison):
        all_features = torch.cat((clean, poison), dim=0)
        all_features -= all_features.mean(dim=0)
        _, _, V = torch.svd(all_features, compute_uv=True, some=False)
        vec = V[:, 0]  # Principal singular vector
        vals = []
        for j in range(all_features.shape[0]):
            vals.append(torch.dot(all_features[j], vec).pow(2))
        vals = torch.tensor(vals)

        print(vals.shape)

        return vals[:clean.shape[0]], vals[clean.shape[0:]]

# Determine the number of classes based on the dataset
if args.dataset == 'cifar10':
    num_classes = 10
elif args.dataset == 'gtsrb':
    num_classes = 43
elif args.dataset == 'imagenette':
    num_classes = 10
else:
    raise NotImplementedError('<Unimplemented Dataset> %s' % args.dataset)

# Get data transformations
data_transform_aug, data_transform, trigger_transform, normalizer, denormalizer = supervisor.get_transforms(args)

# Get the model architecture
arch = supervisor.get_arch(args)

# Set up the poisoned dataset
poison_set_dir = supervisor.get_poison_set_dir(args)
if os.path.exists(os.path.join(poison_set_dir, 'data')):  # old version
    poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
elif os.path.exists(os.path.join(poison_set_dir, 'imgs')):  # new version
    poisoned_set_img_dir = os.path.join(poison_set_dir, 'imgs')
else:
    raise FileNotFoundError("Poisoned data directory not found.")

poisoned_set_label_path = os.path.join(poison_set_dir, 'labels')
poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')

poisoned_set = tools.IMG_Dataset(data_dir=poisoned_set_img_dir,
                                 label_path=poisoned_set_label_path, transforms=data_transform)

poisoned_set_loader = torch.utils.data.DataLoader(
    poisoned_set,
    batch_size=batch_size, shuffle=False, **kwargs)

poison_indices = torch.tensor(torch.load(poison_indices_path))

# Set up the test dataset
test_set_dir = 'clean_set/%s/test_split/' % args.dataset
test_set_img_dir = os.path.join(test_set_dir, 'data')
test_set_label_path = os.path.join(test_set_dir, 'labels')
test_set = tools.IMG_Dataset(data_dir=test_set_img_dir, label_path=test_set_label_path,
                             transforms=data_transform)
test_set_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size, shuffle=False, **kwargs
)

model_list = []
alias_list = []

# Load the model
if (hasattr(args, 'model_path') and args.model_path is not None) or (hasattr(args, 'model') and args.model is not None):
    path = supervisor.get_model_dir(args)
    model_list.append(path)
    alias_list.append('assigned')
else:
    args.no_aug = False
    path = supervisor.get_model_dir(args)
    model_list.append(path)
    alias_list.append(supervisor.get_model_name(args))

# Get the poison transform
poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                   target_class=target_class,
                                                   trigger_transform=data_transform,
                                                   is_normalized_input=True,
                                                   alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                   trigger_name=args.trigger, args=args)

if args.poison_type == 'TaCT':
    source_classes = [config.source_class]
else:
    source_classes = None

for vid, path in enumerate(model_list):

    ckpt = torch.load(path)

    # Load the model
    model = arch(num_classes=num_classes)
    model.load_state_dict(ckpt)
    model = nn.DataParallel(model)
    model = model.cuda()
    model.eval()

    # Start visualization
    print("Visualizing model '{}' on {}...".format(path, args.dataset))

    print('[test]')
    tools.test(model, test_set_loader, poison_test=True, poison_transform=poison_transform, num_classes=num_classes,
               source_classes=source_classes)

    targets = []
    features = []
    clean_features = []
    poisoned_features = []

    # Extract features and targets from the poisoned dataset
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(poisoned_set_loader):
            data, target = data.cuda(), target.cuda()
            targets.append(target)
            _, feature = model.forward(data, return_hidden=True)
            features.append(feature.cpu().detach())

    targets = torch.cat(targets, dim=0)
    targets = targets.cpu()
    features = torch.cat(features, dim=0)
    ids = torch.tensor(list(range(len(poisoned_set))))

    if len(poison_indices) == 0:
        # No poisoned data
        pass
    else:

        # Separate clean and poisoned data
        non_poison_indices = list(set(list(range(len(poisoned_set)))) - set(poison_indices.tolist()))

        clean_targets = targets[non_poison_indices]
        poisoned_targets = targets[poison_indices]

        print("Total Clean:", len(clean_targets))
        print("Total Poisoned:", len(poisoned_targets))

        clean_features = features[non_poison_indices]
        poisoned_features = features[poison_indices]

        clean_ids = ids[non_poison_indices]
        poisoned_ids = ids[poison_indices]

        # Combine all features and labels
        all_features = torch.cat([clean_features, poisoned_features], dim=0)
        all_targets = torch.cat([clean_targets, poisoned_targets], dim=0)
        all_labels = torch.cat([torch.zeros(len(clean_targets)), torch.ones(len(poisoned_targets))], dim=0)  # 0: Clean, 1: Poisoned

        # Convert to numpy arrays
        all_features_np = all_features.cpu().numpy()
        all_targets_np = all_targets.cpu().numpy()
        all_labels_np = all_labels.cpu().numpy()

        # Dimensionality reduction
        if args.method == 'pca':
            visualizer = PCA(n_components=2)
        elif args.method == 'tsne':
            visualizer = TSNE(n_components=2, random_state=args.seed)
        elif args.method == 'umap':
            visualizer = UMAP(n_components=2, random_state=args.seed)
        elif args.method == 'isomap':
            visualizer = Isomap(n_components=2)
        elif args.method == 'lle':
            visualizer = LocallyLinearEmbedding(n_components=2)
        elif args.method == 'kpca':
            visualizer = KernelPCA(n_components=2, kernel='rbf')
        elif args.method == 'spectral':
            visualizer = SpectralEmbedding(n_components=2)
        else:
            raise NotImplementedError('Visualization Method %s is Not Implemented!' % args.method)

        reduced_features = visualizer.fit_transform(all_features_np)

        # Create a DataFrame for easier plotting
        df = pd.DataFrame()
        df['dim1'] = reduced_features[:, 0]
        df['dim2'] = reduced_features[:, 1]
        df['Class'] = all_targets_np.astype(int)
        df['Poisoned'] = all_labels_np.astype(int)

        # Plotting
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df, x='dim1', y='dim2', hue='Class', style='Poisoned',
                        palette='tab10', markers={0: 'o', 1: 'X'}, alpha=0.7)

        plt.title(f'{args.method.upper()} Visualization of Features with Classes and Poisoned Labels')
        plt.legend(title='Class / Poisoned', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        save_path = 'umap_assets/%s_%s_%s_class=%d.png' % (
            args.method, supervisor.get_dir_core(args, include_poison_seed=True), alias_list[vid], target_class)
        plt.tight_layout()
        plt.savefig(save_path)
        print("Saved figure at {}".format(save_path))
        plt.clf()

