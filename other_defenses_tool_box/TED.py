# This is the test code of TED defense.
# Robust Backdoor Detection for Deep Learning via Topological Evolution Dynamics [IEEE, 2024] (https://arxiv.org/abs/2312.02673)

from torchvision.utils import save_image
import os
import random
from collections import Counter, defaultdict
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import pdist, squareform
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
from utils.resnet import ResNet18, ResNet34
from utils.supervisor import get_transforms
from other_defenses_tool_box.tools import generate_dataloader
from other_defenses_tool_box.backdoor_defense import BackdoorDefense
from networks.models import Generator, NetC_MNIST
from defense_dataloader import get_dataset, get_dataloader
import seaborn as sns
from matplotlib.lines import Line2D
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

        print(self.poison_type)

        # 2) Define the backdoor target class
        self.target = self.target_class
        print(f'Target Class: {self.target}')

        # 3) Create train_loader to scan the training set
        if self.poison_type == 'SSDT':
            self.train_loader = get_dataloader(args, train=True)
        else:
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

        print(f'Number samples of train set: {len(self.train_loader.dataset)}')
        # 4) Determine unique classes by scanning the training set
        all_labels = []
        for _, labels in self.train_loader:
            all_labels.extend(labels.tolist())
        unique_classes = set(all_labels)
        num_classes = len(unique_classes)

        print(f"Number of unique classes in the dataset (scanned from train_loader): {num_classes}")
        print(f"Number of unique classes from args: {self.num_classes}")

        # 5) Defense training size (for example)
        self.LAYER_RATIO = 0.3
        self.NUM_LAYERS = 1
        self.SAMPLES_PER_CLASS = 20
        self.DEFENSE_TRAIN_SIZE = self.num_classes * self.SAMPLES_PER_CLASS

        # 6) Create a test_loader
        if self.poison_type == 'SSDT':
            self.test_loader = get_dataloader(args, train=False)
            self.testset = get_dataset(args, train=False)
        else:
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

        print(f'Number samples of test set: {len(self.testset)}')

        # 7) Define NUM_SAMPLES: we take 500 for clean + 500 for poison => total 1000
        self.DATA_RATIO = 0.005
        # self.NUM_SAMPLES = 500
        self.NUM_SAMPLES = int(self.DATA_RATIO * len(self.testset))

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
        from collections import defaultdict
        label_to_indices = defaultdict(list)
        for idx in subset_indices:
            try:
                _, label = underlying_dataset[idx]
                label_to_indices[label].append(idx)
            except FileNotFoundError:
                print(f"Warning: File {idx}.png not exist.")

        # Initialize a dictionary to hold correctly predicted indices per class
        correct_indices_per_class = defaultdict(list)

        # Create a DataLoader for the training set without shuffling to keep track of indices
        train_loader_no_shuffle = data.DataLoader(
            trainset,
            batch_size=50,
            num_workers=0,
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
            num_workers=0,
            shuffle=True
        )

        # Calculate and print the number of samples per class in the defense subset
        from collections import Counter

        # Extract labels from the defense_subset
        defense_labels = [label for _, label in defense_subset]

        # Count the number of samples per class
        label_counts = Counter(defense_labels)

        # print("Number of samples per class in defense_subset:")
        # for label, count in label_counts.items():
        #     print(f"Class {label}: {count} samples")

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

        if len(benign_indices) > self.DEFENSE_TRAIN_SIZE:
            benign_indices = np.random.choice(benign_indices, self.DEFENSE_TRAIN_SIZE, replace=False)

        defense_subset = data.Subset(trainset, benign_indices)
        self.defense_loader = data.DataLoader(
            defense_subset,
            batch_size=50,
            num_workers=0,
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
        self.nnb_distance_dictionary = {}

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

    def create_bd(self, inputs):
        """
        Tạo backdoor inputs
        """
        patterns = self.netG(inputs)
        patterns = self.netG.normalize_pattern(patterns)
        masks_output = self.netM.threshold(self.netM(inputs))
        bd_inputs = inputs + (patterns - inputs) * masks_output
        return bd_inputs

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
        if self.poison_type == 'TaCT' or self.poison_type == 'SSDT':
            print(self.poison_type)

            while self.poison_count < self.NUM_SAMPLES or self.clean_count < self.NUM_SAMPLES:
                for batch_idx, (inputs, labels) in enumerate(self.test_loader):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # ---------------------------------------------------------
                    # 1) Tạo POISON (chỉ áp dụng trigger cho victim_indices)
                    # ---------------------------------------------------------
                    if self.poison_count < self.NUM_SAMPLES:
                        victim_indices = (labels == config.source_class)
                        if victim_indices.sum().item() > 0:
                            # Tách victim_inputs và victim_labels
                            victim_inputs = inputs[victim_indices]
                            victim_labels = labels[victim_indices]

                            # Tạo trigger
                            if self.poison_type == 'SSDT':
                                bd_inputs = self.create_bd(victim_inputs)  # Tạo backdoor
                            else:
                                bd_inputs, _ = self.poison_transform.transform(
                                    victim_inputs, victim_labels
                                )

                            # Dự đoán
                            preds_bd = torch.argmax(self.model(bd_inputs), dim=1)

                            # Kiểm tra mẫu nào predict == self.target
                            correct_pred_indices = (preds_bd == self.target)
                            if correct_pred_indices.sum().item() > 0:
                                final_poison = bd_inputs[correct_pred_indices]
                                # Tạo label tạm "Poison"=101
                                label_value = self.label_mapping[self.POISON_TEMP_LABEL]
                                final_poison_targets = self.create_targets(
                                    victim_labels[correct_pred_indices],
                                    label_value
                                )
                                final_poison_preds = preds_bd[correct_pred_indices]

                                # Thêm vào list
                                self.temp_poison_inputs_set.append(final_poison.cpu())
                                self.temp_poison_labels_set.append(final_poison_targets.cpu())
                                self.temp_poison_pred_set.append(final_poison_preds.cpu())

                                self.poison_count += final_poison.shape[0]

                    # ---------------------------------------------------------
                    # 2) Tạo CLEAN (các mẫu không phải victim class)
                    # ---------------------------------------------------------
                    if self.clean_count < self.NUM_SAMPLES:
                        non_victim_indices = (labels != config.source_class)
                        if non_victim_indices.sum().item() > 0:
                            clean_inputs = inputs[non_victim_indices]
                            clean_labels_ori = labels[non_victim_indices]

                            # Dùng ảnh gốc => KHÔNG gắn trigger
                            preds_clean = torch.argmax(self.model(clean_inputs), dim=1)

                            # Tạo label tạm "Clean"=102
                            label_value = self.label_mapping[self.CLEAN_TEMP_LABEL]
                            clean_targets = self.create_targets(
                                clean_labels_ori, label_value
                            )

                            self.temp_clean_inputs_set.append(clean_inputs.cpu())
                            self.temp_clean_labels_set.append(clean_targets.cpu())
                            self.temp_clean_pred_set.append(preds_clean.cpu())

                            self.clean_count += clean_inputs.shape[0]

                    # ---------------------------------------------------------
                    # 3) Kiểm tra đã đủ số lượng chưa
                    # ---------------------------------------------------------
                    if self.poison_count >= self.NUM_SAMPLES and self.clean_count >= self.NUM_SAMPLES:
                        break

                if self.poison_count >= self.NUM_SAMPLES and self.clean_count >= self.NUM_SAMPLES:
                    break

            # Giới hạn lại nếu thừa
            if self.poison_count > self.NUM_SAMPLES:
                combined_inputs = torch.cat(self.temp_poison_inputs_set, dim=0)[:self.NUM_SAMPLES]
                combined_labels = torch.cat(self.temp_poison_labels_set, dim=0)[:self.NUM_SAMPLES]
                combined_preds = torch.cat(self.temp_poison_pred_set, dim=0)[:self.NUM_SAMPLES]
                self.temp_poison_inputs_set = [combined_inputs]
                self.temp_poison_labels_set = [combined_labels]
                self.temp_poison_pred_set = [combined_preds]
                self.poison_count = self.NUM_SAMPLES

            if self.clean_count > self.NUM_SAMPLES:
                combined_inputs = torch.cat(self.temp_clean_inputs_set, dim=0)[:self.NUM_SAMPLES]
                combined_labels = torch.cat(self.temp_clean_labels_set, dim=0)[:self.NUM_SAMPLES]
                combined_preds = torch.cat(self.temp_clean_pred_set, dim=0)[:self.NUM_SAMPLES]
                self.temp_clean_inputs_set = [combined_inputs]
                self.temp_clean_labels_set = [combined_labels]
                self.temp_clean_pred_set = [combined_preds]
                self.clean_count = self.NUM_SAMPLES

        else:
            """
            Sử dụng cùng một subset test để tạo ra Clean set và Poison set (VD: 500 mẫu).
            """
            all_indices = np.arange(len(self.testset))
            if len(all_indices) < self.NUM_SAMPLES:
                print(f"Warning: testset size < {self.NUM_SAMPLES}, adjusting.")
                chosen = all_indices
            else:
                chosen = np.random.choice(all_indices, size=self.NUM_SAMPLES, replace=False)

            clean_subset = data.Subset(self.testset, chosen)
            clean_loader = data.DataLoader(clean_subset, batch_size=50, shuffle=False)

            poison_subset = data.Subset(self.testset, chosen)
            poison_loader = data.DataLoader(poison_subset, batch_size=50, shuffle=False)

            # Tạo CLEAN set
            for (inputs, labels) in clean_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                label_value = self.label_mapping[self.CLEAN_TEMP_LABEL]
                targets_clean = self.create_targets(labels, label_value)
                preds = torch.argmax(self.model(inputs), dim=1)

                self.temp_clean_inputs_set.append(inputs.cpu())
                self.temp_clean_labels_set.append(targets_clean.cpu())
                self.temp_clean_pred_set.append(preds.cpu())

                self.clean_count += labels.size(0)

            # Tạo POISON set từ cùng subset
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

        ############################################
        # TÍCH HỢP ĐOẠN CODE LƯU ẢNH (POISON/CLEAN)
        ############################################
        # Tạo thư mục lưu
        poison_save_dir = os.path.join(self.save_dir, "poison_images")
        clean_save_dir = os.path.join(self.save_dir, "clean_images")
        os.makedirs(poison_save_dir, exist_ok=True)
        os.makedirs(clean_save_dir, exist_ok=True)

        # Gom các tensor lại
        poison_images = torch.cat(self.temp_poison_inputs_set, dim=0)
        poison_labels = torch.cat(self.temp_poison_labels_set, dim=0)

        clean_images = torch.cat(self.temp_clean_inputs_set, dim=0)
        clean_labels = torch.cat(self.temp_clean_labels_set, dim=0)

        # Lưu poison
        for idx in range(poison_images.shape[0]):
            image_tensor = poison_images[idx]
            label_val = int(poison_labels[idx].item())
            # Lưu với tên file: poison_{idx}_label_{label_val}.png
            filename = f"poison_{idx}_label_{label_val}.png"
            save_path = os.path.join(poison_save_dir, filename)
            save_image(image_tensor, save_path)

        # Lưu clean
        for idx in range(clean_images.shape[0]):
            image_tensor = clean_images[idx]
            label_val = int(clean_labels[idx].item())
            # Lưu với tên file: clean_{idx}_label_{label_val}.png
            filename = f"clean_{idx}_label_{label_val}.png"
            save_path = os.path.join(clean_save_dir, filename)
            save_image(image_tensor, save_path)

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
        self.poison_loader = data.DataLoader(poison_set, batch_size=50, num_workers=0, shuffle=True)
        print("Poison set size:", len(self.poison_loader))

        # Create clean_loader
        clean_set = CustomDataset(clean_inputs_set, clean_labels_set)
        self.clean_loader = data.DataLoader(clean_set, batch_size=50, num_workers=0, shuffle=True)
        print("Clean set size:", len(self.clean_loader))

        # Remove temporary variables
        del bd_inputs_set, bd_labels_set, bd_pred_set
        del clean_inputs_set, clean_labels_set, clean_pred_set

    # ==============================
    #       HOOK & MAIN TEST
    # ==============================
    def fetch_activation(self, loader):
        print("Starting fetch_activation")
        self.model.eval()

        all_h_label, pred_set = [], []
        activation_container = {}

        if self.dataset == 'tinyimagenet200':
            # Dùng phiên bản uncomment (sử dụng torch.no_grad)
            with torch.no_grad():
                # Khởi tạo hook bằng 1 batch
                for images, labels in loader:
                    _ = self.model(images.to(self.device))
                    break
                for key in self.activations:
                    activation_container[key] = []
                self.activations.clear()

                for batch_idx, (images, labels) in enumerate(loader, start=1):
                    images = images.to(self.device)
                    output = self.model(images)
                    pred_set.append(torch.argmax(output, dim=1).cpu())

                    for key in self.activations:
                        h_batch = self.activations[key].view(images.shape[0], -1).cpu()
                        activation_container[key].append(h_batch)

                    all_h_label.append(labels.cpu())
                    self.activations.clear()
                    del images, labels, output
                    torch.cuda.empty_cache()

                    if batch_idx % 10 == 0:
                        print(f"Processed {batch_idx} batches")

            for key in activation_container:
                activation_container[key] = torch.cat(activation_container[key], dim=0)
            all_h_label = torch.cat(all_h_label, dim=0)
            pred_set = torch.cat(pred_set, dim=0)
        else:
            # Dùng phiên bản commented
            # Khởi tạo hook bằng 1 batch
            for images, labels in loader:
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

                for key in self.activations:
                    h_batch = self.activations[key].view(images.shape[0], -1)
                    for h in h_batch:
                        activation_container[key].append(h.to(self.device))

                for label in labels:
                    all_h_label.append(label.to(self.device))
                self.activations.clear()

                if batch_idx % 10 == 0:
                    print(f"Processed {batch_idx} batches")

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

        # --- Loop over each layer for visualization ---
        for layer_name in self.h_defense_activations.keys():
            # --- Extract activations for the current layer ---
            defense_act = self.h_defense_activations[layer_name]  # shape: (n_defense, dim)
            poison_act = self.h_poison_activations[layer_name]  # shape: (n_poison, dim)
            clean_act = self.h_clean_activations[layer_name]  # shape: (n_clean, dim)

            # --- Convert tensors to NumPy arrays ---
            defense_np = defense_act.cpu().detach().numpy()
            poison_np = poison_act.cpu().detach().numpy()
            clean_np = clean_act.cpu().detach().numpy()

            # --- Process defense labels using self.h_defense_ori_labels ---
            defense_ori_labels = self.h_defense_ori_labels.cpu().detach().numpy()
            # If original label equals self.target then label it as "Target", otherwise "Defense"
            defense_labels = np.where(defense_ori_labels == self.target, "Target", "Defense")

            # --- Create labels for poison and clean ---
            poison_labels = np.array(["Poison"] * poison_np.shape[0])
            clean_labels = np.array(["Clean"] * clean_np.shape[0])

            # --- Concatenate activations and labels ---
            all_activations = np.concatenate([defense_np, poison_np, clean_np], axis=0)
            all_labels = np.concatenate([defense_labels, poison_labels, clean_labels], axis=0)

            # Number of samples in each dataset
            n_defense = defense_np.shape[0]
            n_poison = poison_np.shape[0]
            n_clean = clean_np.shape[0]

            # --- Set default color mapping ---
            default_color_map = {
                "Defense": "purple",  # defense non-target
                "Target": "black",  # defense target
                "Poison": "red",
                "Clean": "green"
            }
            # Apply default colors according to the original labels
            colors = [default_color_map[label] for label in all_labels]

            # --- Override color and label for clean special samples ---
            clean_special_idxs = [2, 37, 39]
            for idx in clean_special_idxs:
                global_idx = n_defense + n_poison + idx
                if idx < n_clean:
                    colors[global_idx] = "blue"  # change color
                    all_labels[global_idx] = "Target Clean"  # update label

            # --- Set marker sizes (default = 10) ---
            marker_sizes = [10] * len(all_labels)
            # For defense samples with "Target" label, set marker size to 30
            for i in range(n_defense):
                if defense_labels[i] == "Target":
                    marker_sizes[i] = 100
            # For clean special samples, set marker size to 30
            for idx in clean_special_idxs:
                global_idx = n_defense + n_poison + idx
                if idx < n_clean:
                    marker_sizes[global_idx] = 100

            # --- Apply UMAP to reduce dimensions to 2D ---
            umap_model = UMAP(n_components=2, random_state=42)
            embedding = umap_model.fit_transform(all_activations)

            # --- Set a publication-friendly style ---
            plt.style.use("seaborn-v0_8-paper")
            plt.figure(figsize=(10, 8))

            # Draw scatter plot with high zorder so that points are on top of lines
            plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=marker_sizes, alpha=0.7, zorder=3)

            # --- Draw connection lines for selected poison samples ---
            # Compute pairwise Euclidean distances within poison_np to find a cluster of nearby samples
            distance_matrix = squareform(pdist(poison_np, metric='euclidean'))
            best_indices = None
            best_sum = np.inf
            for i in range(n_poison):
                nearest_idxs = np.argsort(distance_matrix[i])[:40]
                total_distance = distance_matrix[i, nearest_idxs].sum()
                if total_distance < best_sum:
                    best_sum = total_distance
                    best_indices = nearest_idxs

            # For each poison sample in the best_indices cluster, connect to the nearest defense sample if its label is "Target"
            for idx in best_indices:
                if idx < n_poison:
                    global_idx = n_defense + idx  # index of poison sample in all_activations
                    # Get the high-dimensional vector for the poison sample
                    special_vector_np = poison_np[idx:idx + 1]  # shape (1, dim)
                    special_vector = torch.from_numpy(special_vector_np).to(self.device)
                    # Convert defense_np to tensor
                    defense_tensor = torch.from_numpy(defense_np).to(self.device)
                    # Compute pairwise Euclidean distances
                    distances = pairwise_euclidean_distance(special_vector, defense_tensor)
                    nearest_defense_idx = torch.argmin(distances).item()
                    # Draw the connection line in red dashed if the nearest defense sample is labeled "Target"
                    if defense_labels[nearest_defense_idx] == "Target":
                        plt.plot([embedding[global_idx, 0], embedding[nearest_defense_idx, 0]],
                                 [embedding[global_idx, 1], embedding[nearest_defense_idx, 1]],
                                 c='red', linestyle='--', linewidth=2, zorder=2)
                        # Redraw the poison sample with an enlarged marker (size 30)
                        plt.scatter(embedding[global_idx, 0], embedding[global_idx, 1],
                                    c=colors[global_idx], s=100, zorder=4)

            # --- Draw connection lines for clean special samples ---
            for idx in clean_special_idxs:
                if idx < n_clean:
                    global_idx = n_defense + n_poison + idx  # index of clean special sample in all_activations
                    special_vector_np = clean_np[idx:idx + 1]  # shape (1, dim)
                    special_vector = torch.from_numpy(special_vector_np).to(self.device)
                    defense_tensor = torch.from_numpy(defense_np).to(self.device)
                    distances = pairwise_euclidean_distance(special_vector, defense_tensor)
                    nearest_defense_idx = torch.argmin(distances).item()
                    # Only draw the connection if the nearest defense sample has the "Target" label
                    if defense_labels[nearest_defense_idx] == "Target":
                        plt.plot([embedding[global_idx, 0], embedding[nearest_defense_idx, 0]],
                                 [embedding[global_idx, 1], embedding[nearest_defense_idx, 1]],
                                 c='blue', linestyle='--', linewidth=2, zorder=2)

            # --- Customize legend and ticks ---
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='Validation Sample', markerfacecolor='purple',
                       markersize=10),
                Line2D([0], [0], marker='o', color='w', label='Target Validation Sample', markerfacecolor='black',
                       markersize=10),
                Line2D([0], [0], marker='o', color='w', label='Clean Sample', markerfacecolor='green', markersize=10),
                Line2D([0], [0], marker='o', color='w', label='Target Clean Sample', markerfacecolor='blue',
                       markersize=10),
                Line2D([0], [0], marker='o', color='w', label='Poison Sample', markerfacecolor='red', markersize=10),
                Line2D([0], [0], linestyle='--', color='gray', label='Ranking = 0', linewidth=2)
            ]
            plt.legend(handles=legend_elements, loc='best', fontsize=25)
            plt.xticks(fontsize=25)
            plt.yticks(fontsize=25)
            plt.tight_layout()

            # --- Save the figure in PDF format for publication ---
            plot_save_path = os.path.join(self.save_dir, f"umap_{layer_name}.pdf")
            plt.savefig(plot_save_path, bbox_inches='tight', format='pdf', dpi=300)
            plt.close()
            print(f"UMAP plot for layer '{layer_name}' saved to: {plot_save_path}")

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

        pca = PCA(contamination=0.01, n_components=2)
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