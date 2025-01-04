# This is the test code of TED defense.
# Robust Backdoor Detection for Deep Learning via Topological Evolution Dynamics [IEEE, 2024] (https://arxiv.org/abs/2312.02673)

import os
import random
from collections import Counter, defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn import metrics
from sklearn.decomposition import PCA as sklearn_PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from pyod.models.pca import PCA
from umap import UMAP
from numpy.random import choice
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchmetrics.functional import pairwise_euclidean_distance
import plotly.express as px
import config
from utils import supervisor, tools
from utils.supervisor import get_transforms
from other_defenses_tool_box.tools import generate_dataloader
from other_defenses_tool_box.backdoor_defense import BackdoorDefense

# ------------------------------
# Seed settings for reproducibility
# ------------------------------
# Set seed for Python
random.seed(42)
# Set seed for NumPy
np.random.seed(42)
# Set seed for PyTorch
torch.manual_seed(42)
# Optionally set the CUDA seed if a GPU is used
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
# ------------------------------

class TED(BackdoorDefense):
    def __init__(self, args):
        """
        Initialize the TED Defense, load the model and data, and set up necessary variables.
        """
        super().__init__(args)  # Calls the constructor of the parent class, BackdoorDefense
        self.args = args

        # 1) Model Configuration
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # 2) Define the backdoor target class
        self.target = self.target_class
        print(f'Target Class: {self.target}')

        # 3) Create train_loader to scan the training set
        self.train_loader = generate_dataloader(
            dataset=self.dataset,
            dataset_path=config.data_dir,
            batch_size=50,
            split='train',
            data_transform=self.data_transform,
            shuffle=True,
            drop_last=False,
            noisy_test=False
        )

        # 4) Determine unique classes by scanning the training set
        all_labels = []
        for _, labels in self.train_loader:
            all_labels.extend(labels.tolist())
        unique_classes = set(all_labels)
        num_classes = len(unique_classes)

        print(f"Number of unique classes in the dataset (scanned from train_loader): {num_classes}")
        print(f"Number of unique classes from args: {self.num_classes}")

        # 5) Defense training size (for example)
        self.SAMPLES_PER_CLASS = 40 
        self.DEFENSE_TRAIN_SIZE = self.num_classes * self.SAMPLES_PER_CLASS

        # 6) Create a test_loader
        self.test_loader = generate_dataloader(
            dataset=self.dataset,
            dataset_path=config.data_dir,
            batch_size=50,
            split='test',
            data_transform=self.data_transform,
            shuffle=False,
            drop_last=False,
            noisy_test=False
        )
        self.testset = self.test_loader.dataset

        # 7) Define NUM_SAMPLES: we take 500 for clean + 500 for poison => total 1000
        self.NUM_SAMPLES = 500
        # self.NUM_SAMPLES = len(self.testset)

        # 8) Create defense_loader with self.SAMPLES_PER_CLASS per class, ensuring correct predictions
        trainset = self.train_loader.dataset

        # Check if trainset is a Subset and get the underlying dataset and indices
        if isinstance(trainset, data.Subset):
            underlying_dataset = trainset.dataset
            subset_indices = trainset.indices
        else:
            underlying_dataset = trainset
            subset_indices = np.arange(len(trainset))

        # Organize indices by class
        label_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(underlying_dataset):
            label_to_indices[label].append(idx)

        # Initialize a dictionary to hold correctly predicted indices per class
        correct_indices_per_class = defaultdict(list)

        # Create a DataLoader for the training set without shuffling to keep track of indices
        train_loader_no_shuffle = data.DataLoader(
            trainset,
            batch_size=50,
            num_workers=2,
            shuffle=False
        )

        # Initialize a counter to keep track of the current index in the dataset
        current_idx = 0

        # Iterate over the training set and collect correctly predicted samples
        with torch.no_grad():
            for inputs, labels in tqdm(train_loader_no_shuffle, desc="Evaluating trainset for correct predictions"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=1)
                correct_mask = preds == labels

                # Iterate through the batch to collect correct indices
                for i in range(len(labels)):
                    if correct_mask[i].item():
                        if isinstance(trainset, data.Subset):
                            # Get the original index from the subset
                            sample_idx = subset_indices[current_idx]
                        else:
                            sample_idx = current_idx
                        label = labels[i].item()
                        correct_indices_per_class[label].append(sample_idx)
                    current_idx += 1

        # Initialize a list to hold all defense indices
        defense_indices = []

        # For each class, sample self.SAMPLES_PER_CLASS correctly predicted samples
        for label in unique_classes:
            correct_indices = correct_indices_per_class[label]
            num_correct = len(correct_indices)

            if num_correct >= self.SAMPLES_PER_CLASS:
                # If enough correct samples, randomly choose without replacement
                sampled = np.random.choice(correct_indices, self.SAMPLES_PER_CLASS, replace=False)
            else:
                # If not enough, sample with replacement to meet the required number
                sampled = np.random.choice(correct_indices, self.SAMPLES_PER_CLASS, replace=True)
                print(f"Warning: Not enough correctly predicted samples for class {label}. "
                      f"Sampling with replacement to meet the required number.")

            defense_indices.extend(sampled)

        # Create a subset of the training set for defense using the sampled indices
        defense_subset = data.Subset(trainset, defense_indices)

        # Create a DataLoader for the defense subset
        self.defense_loader = data.DataLoader(
            defense_subset,
            batch_size=50,
            num_workers=2,
            shuffle=True
        )

        # Calculate and print the number of samples per class in the defense subset
        from collections import Counter

        # Extract labels from the defense_subset
        defense_labels = [label for _, label in defense_subset]

        # Count the number of samples per class
        label_counts = Counter(defense_labels)

        print("Number of samples per class in defense_subset:")
        for label, count in label_counts.items():
            print(f"Class {label}: {count} samples")

        # 9) Filter the defense set to keep correctly predicted samples only (optional)
        h_benign_preds = []
        h_benign_ori_labels = []

        with torch.no_grad():
            for inputs, labels in self.defense_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=1)
                h_benign_preds.extend(preds.cpu().numpy())
                h_benign_ori_labels.extend(labels.cpu().numpy())

        h_benign_preds = np.array(h_benign_preds)
        h_benign_ori_labels = np.array(h_benign_ori_labels)

        benign_mask = h_benign_ori_labels == h_benign_preds
        benign_indices = np.array(defense_indices)[benign_mask]

        # Giới hạn kích thước của defense subset nếu cần
        if len(benign_indices) > self.DEFENSE_TRAIN_SIZE:
            benign_indices = np.random.choice(benign_indices, self.DEFENSE_TRAIN_SIZE, replace=False)

        # Tái tạo defense_subset từ các chỉ số đã lọc
        defense_subset = data.Subset(trainset, benign_indices)
        self.defense_loader = data.DataLoader(
            defense_subset,
            batch_size=50,
            num_workers=2,
            shuffle=True
        )

        # 10) Define two temporary labels: Poison and Clean
        self.POISON_TEMP_LABEL = "Poison"
        self.CLEAN_TEMP_LABEL = "Clean"
        self.label_mapping = {
            "Poison": 101,
            "Clean": 102
        }

        # Poison and Clean sample counters
        self.poison_count = 0
        self.clean_count = 0

        # Temporary containers for building Poison and Clean sets
        self.temp_poison_inputs_set = []
        self.temp_poison_labels_set = []
        self.temp_poison_pred_set = []

        self.temp_clean_inputs_set = []
        self.temp_clean_labels_set = []
        self.temp_clean_pred_set = []

        # Hooks for activation extraction
        self.hook_handles = []
        self.activations = {}
        self.register_hooks()

        # Additional intermediate variables
        self.Test_C = self.num_classes + 2
        self.topological_representation = {}
        self.candidate_ = {}

        # Directory to save visualizations
        self.save_dir = f"TED/{self.dataset}/{self.poison_type}"
        os.makedirs(self.save_dir, exist_ok=True)

    # ==============================
    #     HELPER FUNCTIONS
    # ==============================
    def register_hooks(self):
        """
        Register forward hooks for layers to extract activations.
        """

        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()

            return hook

        # Remove previous hooks if any
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

        net_children = self.model.modules()
        index = 0
        for child in net_children:
            # Register hooks for specific layers
            if isinstance(child, nn.Conv2d) and child.kernel_size != (1, 1):
                self.hook_handles.append(
                    child.register_forward_hook(get_activation("Conv2d_" + str(index)))
                )
                index += 1
            if isinstance(child, nn.ReLU):
                self.hook_handles.append(
                    child.register_forward_hook(get_activation("Relu_" + str(index)))
                )
                index += 1
            if isinstance(child, nn.Linear):
                self.hook_handles.append(
                    child.register_forward_hook(get_activation("Linear_" + str(index)))
                )
                index += 1

    def create_targets(self, targets, label):
        """
        Assign a new label to targets (e.g., Poison=101, Clean=102).
        """
        new_targets = torch.ones_like(targets) * label
        return new_targets.to(self.device)

    # ==============================
    #   CREATE POISON & CLEAN SETS
    # ==============================
    def generate_poison_clean_sets(self):
        """
        Use self.NUM_SAMPLES to pick a fixed number of samples from the test set. 
        We generate 500 clean samples and then transform the same set into 500 poisoned samples.
        """
        all_indices = np.arange(len(self.testset))
        if len(all_indices) < self.NUM_SAMPLES:
            print(f"Warning: testset size < {self.NUM_SAMPLES}, adjusting.")
            chosen = all_indices
        else:
            chosen = np.random.choice(all_indices, size=self.NUM_SAMPLES, replace=False)

        subset = data.Subset(self.testset, chosen)
        loader = data.DataLoader(subset, batch_size=50, shuffle=False)

        # Create CLEAN set
        for (inputs, labels) in loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            label_value = self.label_mapping[self.CLEAN_TEMP_LABEL]
            targets_clean = self.create_targets(labels, label_value)
            preds = torch.argmax(self.model(inputs), dim=1)

            self.temp_clean_inputs_set.append(inputs.cpu())
            self.temp_clean_labels_set.append(targets_clean.cpu())
            self.temp_clean_pred_set.append(preds.cpu())

            self.clean_count += labels.size(0)

        # Create POISON set from the same subset
        poison_loader = data.DataLoader(subset, batch_size=50, shuffle=False)
        for (inputs, labels) in poison_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            poisoned_inputs, poisoned_labels = self.poison_transform.transform(inputs, labels)
            preds_bd = torch.argmax(self.model(poisoned_inputs), dim=1)

            label_value = self.label_mapping[self.POISON_TEMP_LABEL]
            targets_poison = self.create_targets(labels, label_value)

            self.temp_poison_inputs_set.append(poisoned_inputs.cpu())
            self.temp_poison_labels_set.append(targets_poison.cpu())
            self.temp_poison_pred_set.append(preds_bd.cpu())

            self.poison_count += labels.size(0)

        print(
            f"Finished generate_poison_clean_sets. Clean_count = {self.clean_count}, Poison_count = {self.poison_count}"
        )

    def create_poison_clean_dataloaders(self):
        """
        Build DataLoaders for both Poison and Clean sets.
        """
        # Poison set
        bd_inputs_set = torch.cat(self.temp_poison_inputs_set, dim=0)
        bd_labels_set = np.hstack(self.temp_poison_labels_set)
        bd_pred_set = np.hstack(self.temp_poison_pred_set)

        # Clean set
        clean_inputs_set = torch.cat(self.temp_clean_inputs_set, dim=0)
        clean_labels_set = np.hstack(self.temp_clean_labels_set)
        clean_pred_set = np.hstack(self.temp_clean_pred_set)

        # Define a custom Dataset
        class CustomDataset(data.Dataset):
            def __init__(self, data_, labels_):
                super(CustomDataset, self).__init__()
                self.images = data_
                self.labels = labels_

            def __len__(self):
                return len(self.images)

            def __getitem__(self, index):
                img = self.images[index]
                label = self.labels[index]
                return img, label

        # Create poison_loader
        poison_set = CustomDataset(bd_inputs_set, bd_labels_set)
        self.poison_loader = data.DataLoader(poison_set, batch_size=50, num_workers=2, shuffle=True)
        print("Poison set size:", len(self.poison_loader))

        # Create clean_loader
        clean_set = CustomDataset(clean_inputs_set, clean_labels_set)
        self.clean_loader = data.DataLoader(clean_set, batch_size=50, num_workers=2, shuffle=True)
        print("Clean set size:", len(self.clean_loader))

        # Remove temporary variables
        del bd_inputs_set, bd_labels_set, bd_pred_set
        del clean_inputs_set, clean_labels_set, clean_pred_set

    # ==============================
    #       HOOK & MAIN TEST
    # ==============================
    def fetch_activation(self, loader):
        """
        Run the model on the given loader and fetch intermediate activations based on the registered hooks.
        """
        print("Starting fetch_activation")
        self.model.eval()
        all_h_label = []
        pred_set = []
        h_batch = {}
        activation_container = {}

        # Initialize hooks with one batch
        for (images, labels) in loader:
            print("Running the first batch to init hooks")
            _ = self.model(images.to(self.device))
            break

        for key in self.activations:
            activation_container[key] = []

        self.activations.clear()

        for batch_idx, (images, labels) in enumerate(loader, start=1):
            print(f"Running batch {batch_idx} - Images shape: {images.shape}, Labels shape: {labels.shape}")
            try:
                output = self.model(images.to(self.device))
            except Exception as e:
                print(f"Error running model on batch {batch_idx}: {e}")
                break
            pred_set.append(torch.argmax(output, -1).to(self.device))

            # Collect activations from hooks
            for key in self.activations:
                h_batch[key] = self.activations[key].view(images.shape[0], -1)
                for h in h_batch[key]:
                    activation_container[key].append(h.to(self.device))

            # Save original labels
            for label_ in labels:
                all_h_label.append(label_.to(self.device))

            self.activations.clear()

            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx} batches")

        # Stack everything
        for key in activation_container:
            activation_container[key] = torch.stack(activation_container[key])
        all_h_label = torch.stack(all_h_label)
        pred_set = torch.cat(pred_set)

        print("Finished fetch_activation")
        return all_h_label, activation_container, pred_set

    def calculate_accuracy(self, ori_labels, preds):
        """
        Compute classification accuracy given original labels and predictions.
        """
        if len(ori_labels) == 0:
            return 0.0
        correct = torch.sum(ori_labels == preds).item()
        total = len(ori_labels)
        accuracy = (correct / total) * 100
        return accuracy

    def display_images_grid(self, images, predictions, title_prefix):
        """
        Display a small grid of images with their predictions. Saves the figure to self.save_dir.
        """
        num_images = len(images)
        if num_images == 0:
            return
        cols = 3
        rows = (num_images + cols - 1) // cols
        plt.figure(figsize=(cols * 2, rows * 2))

        for i, (img, prediction) in enumerate(zip(images, predictions)):
            plt.subplot(rows, cols, i + 1)
            img = img.squeeze().cpu().numpy()

            if img.ndim == 3:
                plt.imshow(np.transpose(img, (1, 2, 0)))  # RGB
            else:
                plt.imshow(img, cmap='gray')  # Grayscale

            title = f"{title_prefix} {i + 1}\n(Prediction: {prediction.item()})"
            plt.title(title)
            plt.axis('off')

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"{title_prefix}_grid.png")
        plt.savefig(save_path, dpi=300)
        plt.show()

    def gather_activation_into_class(self, target, h):
        """
        Group activations by class index and store them for further analysis.
        """
        h_c_c = [0 for _ in range(self.Test_C)]
        for c in range(self.Test_C):
            idxs = (target == c).nonzero(as_tuple=True)[0]
            if len(idxs) == 0:
                continue
            h_c = h[idxs, :]
            h_c_c[c] = h_c
        return h_c_c

    def get_dis_sort(self, item, destinations):
        """
        Sort distances between a single item and all destinations, returning sorted indices.
        """
        item_ = item.reshape(1, item.shape[0])
        dev = self.device
        new_dis = pairwise_euclidean_distance(item_.to(dev), destinations.to(dev))
        _, indices_individual = torch.sort(new_dis)
        return indices_individual.to("cpu")

    def getDefenseRegion(self, final_prediction, h_defense_activation, processing_label, layer,
                         layer_test_region_individual):
        """
        For each sample in the specified label, compute the region by distance ranking with defense samples.
        """
        if layer not in layer_test_region_individual:
            layer_test_region_individual[layer] = {}
        layer_test_region_individual[layer][processing_label] = []

        self.candidate_[layer] = self.gather_activation_into_class(final_prediction, h_defense_activation)

        if np.ndim(self.candidate_[layer][processing_label]) == 0:
            print("No sample in this class for label =", processing_label)
        else:
            for index, item in enumerate(self.candidate_[layer][processing_label]):
                ranking_array = self.get_dis_sort(item, h_defense_activation)[0]
                ranking_array = ranking_array[1:]
                r_ = [final_prediction[i] for i in ranking_array]
                if processing_label in r_:
                    itemindex = r_.index(processing_label)
                    layer_test_region_individual[layer][processing_label].append(itemindex)

        return layer_test_region_individual

    def getLayerRegionDistance(self, new_prediction, new_activation, new_temp_label,
                               h_defense_prediction, h_defense_activation,
                               layer, layer_test_region_individual):
        """
        Compute the distance-based region for a new label, comparing to the defense activations.
        """
        if layer not in layer_test_region_individual:
            layer_test_region_individual[layer] = {}
        layer_test_region_individual[layer][new_temp_label] = []

        candidate__ = self.gather_activation_into_class(new_prediction, new_activation)
        labels = torch.unique(new_prediction)

        for processing_label in labels:
            for index, item in enumerate(candidate__[processing_label]):
                ranking_array = self.get_dis_sort(item, h_defense_activation)[0]
                r_ = [h_defense_prediction[i] for i in ranking_array]
                if processing_label in r_:
                    itemindex = r_.index(processing_label)
                    layer_test_region_individual[layer][new_temp_label].append(itemindex)

        return layer_test_region_individual

    def test(self):
        """
        The main testing procedure:
          1) Generate poison/clean sets.
          2) Create corresponding DataLoaders.
          3) Extract activations via hooks.
          4) Compute topological representations.
          5) (Optional) Visualization and outlier detection steps.
        """
        print('STEP 1')
        self.generate_poison_clean_sets()

        print('STEP 2')
        self.create_poison_clean_dataloaders()

        print('STEP 3')
        images_to_display = []
        predictions_to_display = []

        pairs = [
            (self.poison_loader, 3, "Poison Image"),
            (self.clean_loader, 9, "Clean Image")
        ]
        for loader, limit, prefix in pairs:
            count = 0
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                predictions = torch.argmax(self.model(inputs), dim=1).cpu()
                for input_image, pred_ in zip(inputs, predictions):
                    if count < limit:
                        images_to_display.append(input_image.unsqueeze(0))
                        predictions_to_display.append(pred_)
                        count += 1
                    else:
                        break
                if count >= limit:
                    break
            self.display_images_grid(images_to_display, predictions_to_display, title_prefix=prefix)
            images_to_display.clear()
            predictions_to_display.clear()

        print('DEBUG')

        def print_loader_info(loader, name):
            print(f"{name} loader info:")
            print(f"Total samples: {len(loader.dataset)}")
            for batch in loader:
                inputs, labels = batch
                print(f"Batch - Inputs shape: {inputs.shape}, Labels shape: {labels.shape}")
                print(f"Sample labels: {labels[:5]}")
                for i in range(3):
                    img = inputs[i].squeeze().cpu().numpy()
                    plt.imshow(img.transpose(1, 2, 0) if img.ndim == 3 else img, cmap="gray")
                    plt.title(f"{name} - Label: {labels[i]}")
                    plt.axis("off")
                    sample_path = os.path.join(self.save_dir, f"{name}_sample_{i}.png")
                    plt.savefig(sample_path, dpi=300)
                    plt.show()
                break

        print_loader_info(self.poison_loader, "Poison")
        print_loader_info(self.clean_loader, "Clean")
        print_loader_info(self.defense_loader, "Defense")

        try:
            for images, labels in self.defense_loader:
                print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
                break
        except Exception as e:
            print(f"Error loading data from defense_loader: {e}")

        print(f"Using device: {self.device}")
        print(f"Model is on device: {next(self.model.parameters()).device}")

        print('DEBUG')
        print('STEP 4')

        self.h_defense_ori_labels, self.h_defense_activations, self.h_defense_preds = self.fetch_activation(
            self.defense_loader)
        self.h_poison_ori_labels, self.h_poison_activations, self.h_poison_preds = self.fetch_activation(
            self.poison_loader)
        self.h_clean_ori_labels, self.h_clean_activations, self.h_clean_preds = self.fetch_activation(
            self.clean_loader)

        print('DEBUG')

        print(f"Poison Original Labels: {self.h_poison_ori_labels.shape}")
        print(f"Poison Predictions: {self.h_poison_preds.shape}")
        print(f"Number of Poison Activations Layers: {len(self.h_poison_activations)}")
        for layer, activation in self.h_poison_activations.items():
            print(f"Layer: {layer}, Activation Shape: {activation.shape}")

        print(f"Clean Original Labels: {self.h_clean_ori_labels.shape}")
        print(f"Clean Predictions: {self.h_clean_preds.shape}")
        print(f"Number of Clean Activations Layers: {len(self.h_clean_activations)}")
        for layer, activation in self.h_clean_activations.items():
            print(f"Layer: {layer}, Activation Shape: {activation.shape}")

        print(f"Defense Original Labels: {self.h_defense_ori_labels.shape}")
        print(f"Defense Predictions: {self.h_defense_preds.shape}")
        print(f"Number of Defense Activations Layers: {len(self.h_defense_activations)}")
        for layer, activation in self.h_defense_activations.items():
            print(f"Layer: {layer}, Activation Shape: {activation.shape}")
        print('DEBUG')

        print('STEP 5')
        accuracy_defense = self.calculate_accuracy(self.h_defense_ori_labels, self.h_defense_preds)

        poison_GT = torch.ones_like(self.h_poison_preds) * self.target
        correct_poison = torch.sum(poison_GT == self.h_poison_preds).item()
        total_poison = len(self.h_poison_preds)
        accuracy_poison = (correct_poison / total_poison) * 100

        print(f"\nAccuracy on defense_loader (Clean): {accuracy_defense:.2f}%")
        print(f"Accuracy on poison_loader (Poison) : {accuracy_poison:.2f}%")

        print('STEP 7')
        class_names = np.unique(self.h_defense_ori_labels.cpu().numpy())
        for index, label in enumerate(class_names):
            for layer in self.h_defense_activations:
                self.topological_representation = self.getDefenseRegion(
                    final_prediction=self.h_defense_preds,
                    h_defense_activation=self.h_defense_activations[layer],
                    processing_label=label,
                    layer=layer,
                    layer_test_region_individual=self.topological_representation
                )
                topo_rep_array = np.array(self.topological_representation[layer][label])
                print(f"Topological Representation Label [{label}] & layer [{layer}]: {topo_rep_array}")
                print(f"Mean: {np.mean(topo_rep_array)}\n")

        for layer_ in self.h_poison_activations:
            self.topological_representation = self.getLayerRegionDistance(
                new_prediction=self.h_poison_preds,
                new_activation=self.h_poison_activations[layer_],
                new_temp_label=self.POISON_TEMP_LABEL,
                h_defense_prediction=self.h_defense_preds,
                h_defense_activation=self.h_defense_activations[layer_],
                layer=layer_,
                layer_test_region_individual=self.topological_representation
            )
            topo_rep_array_poison = np.array(self.topological_representation[layer_][self.POISON_TEMP_LABEL])
            print(
                f"Topological Representation Label [{self.POISON_TEMP_LABEL}] & layer [{layer_}]: {topo_rep_array_poison}")
            print(f"Mean: {np.mean(topo_rep_array_poison)}\n")

        for layer_ in self.h_clean_activations:
            self.topological_representation = self.getLayerRegionDistance(
                new_prediction=self.h_clean_preds,
                new_activation=self.h_clean_activations[layer_],
                new_temp_label=self.CLEAN_TEMP_LABEL,
                h_defense_prediction=self.h_defense_preds,
                h_defense_activation=self.h_defense_activations[layer_],
                layer=layer_,
                layer_test_region_individual=self.topological_representation
            )
            topo_rep_array_clean = np.array(self.topological_representation[layer_][self.CLEAN_TEMP_LABEL])
            print(
                f"Topological Representation Label [{self.CLEAN_TEMP_LABEL}] - layer [{layer_}]: {topo_rep_array_clean}")
            print(f"Mean: {np.mean(topo_rep_array_clean)}\n")

        print('STEP 8')

        def aggregate_by_all_layers(output_label):
            inputs_container = []
            first_key = list(self.topological_representation.keys())[0]
            labels_container = np.repeat(output_label,
                                         len(self.topological_representation[first_key][output_label]))
            for l in self.topological_representation.keys():
                temp = []
                for j in range(len(self.topological_representation[l][output_label])):
                    temp.append(self.topological_representation[l][output_label][j])
                if temp:
                    inputs_container.append(np.array(temp))
            return np.array(inputs_container).T, np.array(labels_container)

        inputs_all_benign = []
        labels_all_benign = []
        inputs_all_unknown = []
        labels_all_unknown = []

        first_key = list(self.topological_representation.keys())[0]
        class_name = list(self.topological_representation[first_key])

        for inx in class_name:
            inputs, labels = aggregate_by_all_layers(output_label=inx)
            if inx != self.POISON_TEMP_LABEL and inx != self.CLEAN_TEMP_LABEL:
                inputs_all_benign.append(np.array(inputs))
                labels_all_benign.append(np.array(labels))
            else:
                inputs_all_unknown.append(np.array(inputs))
                labels_all_unknown.append(np.array(labels))

        inputs_all_benign = np.concatenate(inputs_all_benign)
        labels_all_benign = np.concatenate(labels_all_benign)

        inputs_all_unknown = np.concatenate(inputs_all_unknown)
        labels_all_unknown = np.concatenate(labels_all_unknown)

        print('STEP 9')
        pca_t = sklearn_PCA(n_components=2)
        pca_fit = pca_t.fit(inputs_all_benign)

        benign_trajectories = pca_fit.transform(inputs_all_benign)
        trajectories = pca_fit.transform(np.concatenate((inputs_all_unknown, inputs_all_benign), axis=0))

        df_classes = pd.DataFrame(np.concatenate((labels_all_unknown, labels_all_benign), axis=0))

        fig_ = px.scatter(
            trajectories, x=0, y=1, color=df_classes[0].astype(str), labels={'color': 'digit'},
            color_discrete_sequence=px.colors.qualitative.Dark24,
        )

        pca = PCA(contamination=0.01, n_components='mle')
        pca.fit(inputs_all_benign)

        y_train_scores = pca.decision_function(inputs_all_benign)
        y_test_scores = pca.decision_function(inputs_all_unknown)
        y_test_pred = pca.predict(inputs_all_unknown)
        prediction_mask = (y_test_pred == 1)
        prediction_labels = labels_all_unknown[prediction_mask]
        label_counts = Counter(prediction_labels)

        print("\n----------- DETECTION RESULTS -----------")
        for label, count in label_counts.items():
            print(f'Label {label}: {count}')

        is_poison_mask = (labels_all_unknown == self.POISON_TEMP_LABEL).astype(int)
        fpr, tpr, thresholds = metrics.roc_curve(is_poison_mask, y_test_scores, pos_label=1)
        auc_val = metrics.auc(fpr, tpr)

        tn, fp, fn, tp = confusion_matrix(is_poison_mask, y_test_pred).ravel()
        TPR = tp / (tp + fn) if (tp + fn) > 0 else 0
        FPR = fp / (fp + tn) if (fp + tn) > 0 else 0
        f1 = metrics.f1_score(is_poison_mask, y_test_pred)

        print("TPR: {:.2f}%".format(TPR * 100))
        print("FPR: {:.2f}%".format(FPR * 100))
        print("AUC: {:.4f}".format(auc_val))
        print(f"F1 score: {f1:.4f}")
        print("True Positives (TP):", tp)
        print("False Positives (FP):", fp)
        print("True Negatives (TN):", tn)
        print("False Negatives (FN):", fn)

        print("\n[INFO] TED run completed.")

    def detect(self):
        """
        Entry point for the detection procedure.
        """
        self.test()

    def __del__(self):
        for h in self.hook_handles:
            h.remove()
        torch.cuda.empty_cache()

