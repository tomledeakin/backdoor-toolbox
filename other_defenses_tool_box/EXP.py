# This is the test code of TED defense.  
# Robust Backdoor Detection for Deep Learning via Topological Evolution Dynamics [IEEE, 2024] (https://arxiv.org/abs/2312.02673)
import os
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

# ==>> Thêm thư viện StandardScaler
from sklearn.preprocessing import StandardScaler

from utils import supervisor, tools
from utils.supervisor import get_transforms
from other_defenses_tool_box.tools import generate_dataloader
from other_defenses_tool_box.backdoor_defense import BackdoorDefense


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
        print(f'[DEBUG] Target Class: {self.target}')

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

        print(f"[DEBUG] Number of unique classes in the dataset (scanned from train_loader): {num_classes}")
        print(f"[DEBUG] Number of unique classes from args: {self.num_classes}")

        # 5) Defense training size (for example)
        self.DEFENSE_TRAIN_SIZE = num_classes * 40

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
        
        # 8) Create defense_loader from the training set rather than the test set
        trainset = self.train_loader.dataset
        indices = np.arange(len(trainset))

        # Randomly select 10% of the training set for the defense subset
        _, defense_subset_indices = train_test_split(
            indices, test_size=0.1, random_state=42
        )

        defense_subset = data.Subset(trainset, defense_subset_indices)
        self.defense_loader = data.DataLoader(
            defense_subset,
            batch_size=50,
            num_workers=2,
            shuffle=True
        )

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
        benign_indices = defense_subset_indices[benign_mask]

        # Limit the size of the defense subset if needed
        if len(benign_indices) > self.DEFENSE_TRAIN_SIZE:
            benign_indices = np.random.choice(benign_indices, self.DEFENSE_TRAIN_SIZE, replace=False)

        # Rebuild defense_loader from the filtered training subset
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
        self.Test_C = num_classes + 2
        self.topological_representation = {}
        self.candidate_ = {}

        # ===> Lưu centroid
        self.centroids = {}  # self.centroids[layer][class_label] = centroid_vector

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
            print(f"[DEBUG] Warning: testset size < {self.NUM_SAMPLES}, adjusting.")
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
            f"[DEBUG] Finished generate_poison_clean_sets. Clean_count = {self.clean_count}, Poison_count = {self.poison_count}"
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

        # Debug shapes
        print(f"[DEBUG] Poison set shapes: inputs={bd_inputs_set.shape}, labels={bd_labels_set.shape}")
        print(f"[DEBUG] Clean set shapes: inputs={clean_inputs_set.shape}, labels={clean_labels_set.shape}")

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
        print("[DEBUG] Poison set size:", len(self.poison_loader.dataset))

        # Create clean_loader
        clean_set = CustomDataset(clean_inputs_set, clean_labels_set)
        self.clean_loader = data.DataLoader(clean_set, batch_size=50, num_workers=2, shuffle=True)
        print("[DEBUG] Clean set size:", len(self.clean_loader.dataset))

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
        print("[DEBUG] Starting fetch_activation")
        self.model.eval()
        all_h_label = []
        pred_set = []
        h_batch = {}
        activation_container = {}

        # Initialize hooks with one batch
        for (images, labels) in loader:
            print("[DEBUG] Running the first batch to init hooks, shapes:", images.shape, labels.shape)
            _ = self.model(images.to(self.device))
            break

        for key in self.activations:
            activation_container[key] = []

        self.activations.clear()

        for batch_idx, (images, labels) in enumerate(loader, start=1):
            # debug batch
            print(f"[DEBUG] fetch_activation - batch {batch_idx}: images shape = {images.shape}, labels shape = {labels.shape}")
            try:
                output = self.model(images.to(self.device))
            except Exception as e:
                print(f"[DEBUG] Error running model on batch {batch_idx}: {e}")
                break
            pred_set.append(torch.argmax(output, -1).cpu())

            # Collect activations from hooks
            for key in self.activations:
                # (N, D)
                h_batch[key] = self.activations[key].view(images.shape[0], -1).cpu()
                if torch.isnan(h_batch[key]).any():
                    print(f"[DEBUG] Found NaN in activation {key} at batch {batch_idx}")
                activation_container[key].append(h_batch[key])

            # Lưu labels
            all_h_label.append(labels)

            self.activations.clear()

            if batch_idx % 10 == 0:
                print(f"[DEBUG] Processed {batch_idx} batches (fetch_activation).")

        # Stack everything
        for key in activation_container:
            activation_container[key] = torch.cat(activation_container[key], dim=0)
            # debug activation shape
            print(f"[DEBUG] After cat, {key} shape = {activation_container[key].shape}")
            if torch.isnan(activation_container[key]).any():
                print(f"[DEBUG] Found NaN in activation_container[{key}] after cat()")

        all_h_label = torch.cat(all_h_label, dim=0)
        pred_set = torch.cat(pred_set, dim=0)

        print("[DEBUG] Finished fetch_activation")
        print(f"[DEBUG] all_h_label shape = {all_h_label.shape}, pred_set shape = {pred_set.shape}")
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
        h: (N, D)
        target: (N,)
        Return: h_c_c[class_idx] = tensor of shape (num_sample_class, D)
        """
        h_c_c = [0 for _ in range(self.Test_C)]
        for c in range(self.Test_C):
            idxs = (target == c).nonzero(as_tuple=True)[0]
            if len(idxs) == 0:
                continue
            h_c = h[idxs, :]
            h_c_c[c] = h_c
        return h_c_c

    # ==============================
    #       CENTROID LOGIC
    # ==============================
    def compute_defense_centroids(self, defense_activations, defense_preds):
        """
        Tính centroid (vector trung bình) của từng class (label) trong defense_activations.
        defense_activations: {layer_name -> (N, D) tensor}
        defense_preds: shape (N,)
        Return: self.centroids[layer][label] = 1-D tensor (vector trung bình)
        """
        print("[DEBUG] compute_defense_centroids start")
        centroids_dict = {}
        for layer_name, act_mat in defense_activations.items():
            print(f"[DEBUG] Layer {layer_name}, act_mat shape = {act_mat.shape}")
            if torch.isnan(act_mat).any():
                print(f"[DEBUG] Found NaN in defense_activations[{layer_name}] before grouping.")
            # group by class
            group_ = self.gather_activation_into_class(defense_preds, act_mat)
            centroids_dict[layer_name] = {}
            for class_label, sample_mat in enumerate(group_):
                if isinstance(sample_mat, int) and sample_mat == 0:
                    # no sample
                    print(f"[DEBUG] No sample in layer {layer_name}, class_label={class_label}")
                    continue
                # Tính trung bình (theo chiều 0 => vector shape (D,))
                centroid_vec = torch.mean(sample_mat, dim=0)
                if torch.isnan(centroid_vec).any():
                    print(f"[DEBUG] Found NaN in centroid of layer={layer_name}, class_label={class_label}")
                centroids_dict[layer_name][class_label] = centroid_vec
                print(f"[DEBUG] Centroid layer={layer_name}, class_label={class_label}, shape={centroid_vec.shape}")
        print("[DEBUG] compute_defense_centroids end")
        return centroids_dict

    def getDefenseRegion(self, final_prediction, h_defense_activation, processing_label, layer,
                         layer_test_region_individual):
        """
        Giờ ta đo khoảng cách tới centroid của class 'processing_label' trong layer defense.
        """
        if layer not in layer_test_region_individual:
            layer_test_region_individual[layer] = {}
        if processing_label not in layer_test_region_individual[layer]:
            layer_test_region_individual[layer][processing_label] = []

        # Lấy activation "candidate_" thuộc về lớp processing_label
        self.candidate_[layer] = self.gather_activation_into_class(final_prediction, h_defense_activation)

        # Nếu không có sample nào thuộc class 'processing_label' => bỏ qua
        if np.ndim(self.candidate_[layer][processing_label]) == 0:
            print(f"[DEBUG] getDefenseRegion: No sample in this class for label = {processing_label}, layer={layer}")
            return layer_test_region_individual

        # Lấy centroid vector của class processing_label (nằm trong self.centroids)
        if processing_label not in self.centroids[layer]:
            print(f"[DEBUG] [WARNING] Layer {layer}, label {processing_label} không có centroid (có thể do data rỗng?).")
            return layer_test_region_individual

        centroid_vector = self.centroids[layer][processing_label]

        candidate_data = self.candidate_[layer][processing_label]
        if torch.isnan(candidate_data).any():
            print(f"[DEBUG] Found NaN in candidate_[{layer}][{processing_label}]")

        for index, item in enumerate(candidate_data):
            distance = torch.norm(item - centroid_vector, p=2).item()  # L2 distance
            if np.isnan(distance):
                print(f"[DEBUG] distance is NaN at index={index}, layer={layer}, label={processing_label}")
                print(f"[DEBUG] item = {item}, centroid = {centroid_vector}")
            layer_test_region_individual[layer][processing_label].append(distance)

        # Kiểm tra nan
        arr_ = layer_test_region_individual[layer][processing_label]
        if np.isnan(arr_).any():
            print(f"[DEBUG] getDefenseRegion output NaN, layer={layer}, label={processing_label}: {arr_}")

        return layer_test_region_individual

    def getLayerRegionDistance(self, new_prediction, new_activation, new_temp_label,
                               h_defense_prediction, h_defense_activation,
                               layer, layer_test_region_individual):
        """
        Tính khoảng cách tới centroid cho new_temp_label, dựa trên nhãn thực sự (new_prediction).
        """
        if layer not in layer_test_region_individual:
            layer_test_region_individual[layer] = {}
        if new_temp_label not in layer_test_region_individual[layer]:
            layer_test_region_individual[layer][new_temp_label] = []

        # group new_activation by class
        candidate__ = self.gather_activation_into_class(new_prediction, new_activation)

        # Lặp qua từng class c
        labels = torch.unique(new_prediction)
        for processing_label in labels:
            # Nếu class c ko tồn tại, bỏ qua
            if isinstance(candidate__[processing_label], int) and candidate__[processing_label] == 0:
                print(f"[DEBUG] getLayerRegionDistance: No sample for label={processing_label} in new_prediction")
                continue

            # Lấy centroid vector
            if processing_label not in self.centroids[layer]:
                # Trường hợp hiếm gặp: defense ko có class processing_label => ko có centroid
                print(f"[DEBUG] getLayerRegionDistance: layer={layer}, label={processing_label} has no centroid.")
                continue

            centroid_vector = self.centroids[layer][processing_label]
            sample_data = candidate__[processing_label]
            if torch.isnan(sample_data).any():
                print(f"[DEBUG] Found NaN in sample_data, layer={layer}, label={processing_label}")

            # Tính distance
            for index, item in enumerate(sample_data):
                distance = torch.norm(item - centroid_vector, p=2).item()
                if np.isnan(distance):
                    print(f"[DEBUG] distance is NaN at index={index}, new_temp_label={new_temp_label}")
                    print(f"[DEBUG] item = {item}, centroid={centroid_vector}")
                layer_test_region_individual[layer][new_temp_label].append(distance)

        arr_ = layer_test_region_individual[layer][new_temp_label]
        if len(arr_) > 0 and np.isnan(arr_).any():
            print(f"[DEBUG] getLayerRegionDistance output NaN, layer={layer}, new_temp_label={new_temp_label}: {arr_}")

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
        print('[DEBUG] STEP 1')
        self.generate_poison_clean_sets()

        print('[DEBUG] STEP 2')
        self.create_poison_clean_dataloaders()

        print('[DEBUG] STEP 3')
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

        print('[DEBUG] INFO about poison_loader, clean_loader, defense_loader')
        def print_loader_info(loader, name):
            print(f"[DEBUG] {name} loader info:")
            print(f"    => Total samples: {len(loader.dataset)}")
            for batch in loader:
                inputs, labels = batch
                print(f"[DEBUG] {name} - a batch: inputs shape={inputs.shape}, labels shape={labels.shape}")
                break

        print_loader_info(self.poison_loader, "Poison")
        print_loader_info(self.clean_loader, "Clean")
        print_loader_info(self.defense_loader, "Defense")

        print(f"[DEBUG] Using device: {self.device}")
        print(f"[DEBUG] Model is on device: {next(self.model.parameters()).device}")

        print('[DEBUG] STEP 4: fetch_activation')
        # Lấy activation của 3 loader (defense, poison, clean)
        self.h_defense_ori_labels, self.h_defense_activations, self.h_defense_preds = self.fetch_activation(
            self.defense_loader)
        self.h_poison_ori_labels, self.h_poison_activations, self.h_poison_preds = self.fetch_activation(
            self.poison_loader)
        self.h_clean_ori_labels, self.h_clean_activations, self.h_clean_preds = self.fetch_activation(
            self.clean_loader)

        # ============== TÍNH CENTROID ==============
        print('[DEBUG] STEP 4.1: compute centroids')
        self.centroids = self.compute_defense_centroids(
            defense_activations=self.h_defense_activations,
            defense_preds=self.h_defense_preds
        )
        # ===========================================

        print('[DEBUG] STEP 4.2: shapes after fetch_activation')
        print(f"[DEBUG] h_poison_ori_labels shape = {self.h_poison_ori_labels.shape}, h_poison_preds shape={self.h_poison_preds.shape}")
        for layer, activation in self.h_poison_activations.items():
            print(f"[DEBUG] Poison layer {layer} shape= {activation.shape}")

        print(f"[DEBUG] h_clean_ori_labels shape = {self.h_clean_ori_labels.shape}, h_clean_preds shape={self.h_clean_preds.shape}")
        for layer, activation in self.h_clean_activations.items():
            print(f"[DEBUG] Clean layer {layer} shape= {activation.shape}")

        print(f"[DEBUG] h_defense_ori_labels shape = {self.h_defense_ori_labels.shape}, h_defense_preds shape={self.h_defense_preds.shape}")
        for layer, activation in self.h_defense_activations.items():
            print(f"[DEBUG] Defense layer {layer} shape= {activation.shape}")

        print('[DEBUG] STEP 5: check accuracy')
        accuracy_defense = self.calculate_accuracy(self.h_defense_ori_labels, self.h_defense_preds)
        poison_GT = torch.ones_like(self.h_poison_preds) * self.target
        correct_poison = torch.sum(poison_GT == self.h_poison_preds).item()
        total_poison = len(self.h_poison_preds)
        accuracy_poison = (correct_poison / total_poison) * 100

        print(f"[DEBUG] Accuracy on defense_loader (Clean): {accuracy_defense:.2f}%")
        print(f"[DEBUG] Accuracy on poison_loader (Poison) : {accuracy_poison:.2f}%")

        print('[DEBUG] STEP 7: topological representation - defense, poison, clean')

        # Tính khoảng cách topological cho Defense
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
                print(f"[DEBUG] => Defense Label [{label}] & layer [{layer}], shape={topo_rep_array.shape}")
                print(f"[DEBUG] => Defense distance array: {topo_rep_array}")
                if len(topo_rep_array) > 0:
                    print(f"[DEBUG] => Mean: {np.mean(topo_rep_array)}\n")
                else:
                    print(f"[DEBUG] => No sample for label={label} on layer={layer}?\n")

        # Tính khoảng cách topological cho Poison
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
            print(f"[DEBUG] => Poison layer [{layer_}], shape={topo_rep_array_poison.shape}")
            print(f"[DEBUG] => Poison distance array: {topo_rep_array_poison}")
            if len(topo_rep_array_poison) > 0:
                print(f"[DEBUG] => Mean: {np.mean(topo_rep_array_poison)}\n")
            else:
                print(f"[DEBUG] => Poison array empty on layer={layer_}\n")

        # Tính khoảng cách topological cho Clean
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
            print(f"[DEBUG] => Clean layer [{layer_}], shape={topo_rep_array_clean.shape}")
            print(f"[DEBUG] => Clean distance array: {topo_rep_array_clean}")
            if len(topo_rep_array_clean) > 0:
                print(f"[DEBUG] => Mean: {np.mean(topo_rep_array_clean)}\n")
            else:
                print(f"[DEBUG] => Clean array empty on layer={layer_}\n")

        print('[DEBUG] STEP 8: aggregate_by_all_layers')
        def aggregate_by_all_layers(output_label):
            inputs_container = []
            first_key = list(self.topological_representation.keys())[0]
            # if label ko có trong first_key => cẩn thận
            if output_label not in self.topological_representation[first_key]:
                print(f"[DEBUG] => aggregate_by_all_layers: {output_label} not in topological_rep of layer {first_key}")
                return np.array([]), np.array([])
            total_len = len(self.topological_representation[first_key][output_label])
            labels_container = np.repeat(output_label, total_len)
            for l in self.topological_representation.keys():
                if output_label not in self.topological_representation[l]:
                    # label đó k có trong layer l
                    # => ta push 1 mảng rỗng => cẩn thận shape
                    continue
                temp = []
                for j in range(len(self.topological_representation[l][output_label])):
                    temp.append(self.topological_representation[l][output_label][j])
                if len(temp) > 0:
                    inputs_container.append(np.array(temp))
            if len(inputs_container) == 0:
                return np.array([]), np.array([])
            return np.array(inputs_container).T, np.array(labels_container)

        inputs_all_benign = []
        labels_all_benign = []
        inputs_all_unknown = []
        labels_all_unknown = []

        first_key = list(self.topological_representation.keys())[0]
        class_name = list(self.topological_representation[first_key])

        for inx in class_name:
            # debug
            print(f"[DEBUG] aggregate label = {inx}")
            inputs, labels = aggregate_by_all_layers(output_label=inx)
            if inputs.size == 0:
                print(f"[DEBUG] => skip label {inx}, inputs.size=0")
                continue
            if inx != self.POISON_TEMP_LABEL and inx != self.CLEAN_TEMP_LABEL:
                inputs_all_benign.append(np.array(inputs))
                labels_all_benign.append(np.array(labels))
            else:
                inputs_all_unknown.append(np.array(inputs))
                labels_all_unknown.append(np.array(labels))

        if len(inputs_all_benign) == 0:
            print("[DEBUG] No benign data aggregated => possible empty. Check your dataset / logic.")
        else:
            inputs_all_benign = np.concatenate(inputs_all_benign, axis=0)
            labels_all_benign = np.concatenate(labels_all_benign, axis=0)

        if len(inputs_all_unknown) == 0:
            print("[DEBUG] No unknown data aggregated => possible empty. Check your logic.")
        else:
            inputs_all_unknown = np.concatenate(inputs_all_unknown, axis=0)
            labels_all_unknown = np.concatenate(labels_all_unknown, axis=0)

        print(f"[DEBUG] inputs_all_benign shape={inputs_all_benign.shape}, labels_all_benign shape={labels_all_benign.shape}")
        print(f"[DEBUG] inputs_all_unknown shape={inputs_all_unknown.shape}, labels_all_unknown shape={labels_all_unknown.shape}")

        # Kiểm tra NaN / Inf
        if np.isnan(inputs_all_benign).any():
            print("[DEBUG] inputs_all_benign has NaN!")
        if np.isinf(inputs_all_benign).any():
            print("[DEBUG] inputs_all_benign has Inf!")
        if np.isnan(inputs_all_unknown).any():
            print("[DEBUG] inputs_all_unknown has NaN!")
        if np.isinf(inputs_all_unknown).any():
            print("[DEBUG] inputs_all_unknown has Inf!")

        print('[DEBUG] STEP 9: scaling & PCA')

        if inputs_all_benign.shape[0] == 0 or inputs_all_unknown.shape[0] == 0:
            print("[DEBUG] Not enough data to proceed PCA => Possibly everything is empty.")
            return

        scaler = StandardScaler()
        inputs_all_benign = scaler.fit_transform(inputs_all_benign)
        inputs_all_unknown = scaler.transform(inputs_all_unknown)

        pca_t = sklearn_PCA(n_components=2)
        pca_fit = pca_t.fit(inputs_all_benign)

        benign_trajectories = pca_fit.transform(inputs_all_benign)
        trajectories = pca_fit.transform(np.concatenate((inputs_all_unknown, inputs_all_benign), axis=0))

        df_classes = pd.DataFrame(np.concatenate((labels_all_unknown, labels_all_benign), axis=0))

        fig_ = px.scatter(
            trajectories, x=0, y=1, color=df_classes[0].astype(str), labels={'color': 'digit'},
            color_discrete_sequence=px.colors.qualitative.Dark24,
        )

        # PyOD PCA - cũng sẽ hoạt động với dữ liệu đã scale
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
            print(f'[DEBUG] Label {label}: {count}')

        is_poison_mask = (labels_all_unknown == self.POISON_TEMP_LABEL).astype(int)
        if len(is_poison_mask) == 0:
            print("[DEBUG] is_poison_mask is empty => no test data => cannot compute metrics.")
            return

        if np.all(is_poison_mask == 0):
            print("[DEBUG] is_poison_mask is all 0 => no poison in unknown => cannot compute metrics.")
            return

        fpr, tpr, thresholds = metrics.roc_curve(is_poison_mask, y_test_scores, pos_label=1)
        auc_val = metrics.auc(fpr, tpr)

        try:
            tn, fp, fn, tp = confusion_matrix(is_poison_mask, y_test_pred).ravel()
        except:
            print(f"[DEBUG] confusion_matrix error => is_poison_mask shape={is_poison_mask.shape}, y_test_pred shape={y_test_pred.shape}")
            return

        TPR = tp / (tp + fn) if (tp + fn) > 0 else 0
        FPR = fp / (fp + tn) if (fp + tn) > 0 else 0
        f1 = metrics.f1_score(is_poison_mask, y_test_pred)

        print("[DEBUG] TPR: {:.2f}%".format(TPR * 100))
        print("[DEBUG] FPR: {:.2f}%".format(FPR * 100))
        print("[DEBUG] AUC: {:.4f}".format(auc_val))
        print(f"[DEBUG] F1 score: {f1:.4f}")
        print("[DEBUG] True Positives (TP):", tp)
        print("[DEBUG] False Positives (FP):", fp)
        print("[DEBUG] True Negatives (TN):", tn)
        print("[DEBUG] False Negatives (FN):", fn)

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
