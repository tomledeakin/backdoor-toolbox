import os
from collections import Counter, defaultdict
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
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
import wandb

import config
from utils import supervisor, tools
from utils.supervisor import get_transforms
from other_defenses_tool_box.tools import generate_dataloader
from other_defenses_tool_box.backdoor_defense import BackdoorDefense

wandb.login(key="e09f73bb0df882dd4606253c95e1bc68801828a0")


class TED(BackdoorDefense):
    def __init__(self, args):
        """
        Initialize the TED Defense, load model and data, set up necessary variables.
        """
        super().__init__(args)  # Call the constructor of BackdoorDefense (or parent class)
        self.args = args

        self.model.eval()

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.target = self.target_class

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

        # Fallback if 'classes' attribute is not available
        all_labels = []
        for _, labels in self.train_loader:
            all_labels.extend(labels.tolist())
        unique_classes = set(all_labels)
        num_classes = len(unique_classes)
        print(f"Number of unique classes in the dataset: {num_classes}")

        # Set DEFENSE_TRAIN_SIZE
        self.DEFENSE_TRAIN_SIZE = 400

        # Create test_loader
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

        # Load the entire testset
        self.testset = self.test_loader.dataset
        indices = np.arange(len(self.testset))

        # Split dataset into benign_unknown and defense_subset
        benign_unknown_indices, defense_subset_indices = train_test_split(
            indices, test_size=0.1, random_state=42
        )

        benign_unknown_subset = data.Subset(self.testset, benign_unknown_indices)
        defense_subset = data.Subset(self.testset, defense_subset_indices)

        self.benign_unknown_loader = data.DataLoader(
            benign_unknown_subset,
            batch_size=50,
            num_workers=2,
            shuffle=True
        )

        self.defense_loader = data.DataLoader(
            defense_subset,
            batch_size=50,
            num_workers=2,
            shuffle=True
        )

        # ------------------- 4) Filter defense dataset (only take correctly predicted samples) -------------------
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

        if len(benign_indices) > self.DEFENSE_TRAIN_SIZE:
            benign_indices = np.random.choice(benign_indices, self.DEFENSE_TRAIN_SIZE, replace=False)

        defense_subset = data.Subset(self.testset, benign_indices)
        self.defense_loader = data.DataLoader(defense_subset, batch_size=50, num_workers=2,
                                              shuffle=True)

        # ------------------- 5) Define two classes: Poison and Clean -------------------
        self.POISON_TEMP_LABEL = "Poison"
        self.CLEAN_TEMP_LABEL = "Clean"

        self.label_mapping = {
            "Poison": 101,
            "Clean": 102
        }

        self.VICTIM = config.source_class

        self.UNKNOWN_SIZE_POISON = 400
        self.UNKNOWN_SIZE_CLEAN = 200

        # Count Poison / Clean samples
        self.poison_count = 0
        self.clean_count = 0

        # ------------------- 6) Temporary containers for Poison/Clean -------------------
        self.temp_poison_inputs_set = []
        self.temp_poison_labels_set = []
        self.temp_poison_pred_set = []

        self.temp_clean_inputs_set = []
        self.temp_clean_labels_set = []
        self.temp_clean_pred_set = []

        # ------------------- 7) Hooks for activation extraction -------------------
        # We use "hook_handles"
        self.hook_handles = []
        self.activations = {}

        # Register hooks
        self.register_hooks()

        # ------------------- 8) Other intermediate variables -------------------
        self.Test_C = num_classes + 2
        self.topological_representation = {}
        self.candidate_ = {}

        # Create directory for saving visualizations: TED/<dataset>/<attack_name>
        self.save_dir = f"TED/{self.dataset}/{self.poison_type}"
        os.makedirs(self.save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # --------------------------- HELPER FUNCS --------------------------
    # ------------------------------------------------------------------

    def register_hooks(self):
        """
        Register forward hooks for layers to extract activations.
        """

        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()

            return hook

        # Remove old hooks if any
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

        net_children = self.model.modules()

        index = 0
        for child in net_children:
            # Conv2d (skip kernel_size == (1,1) - depends on logic)
            if isinstance(child, nn.Conv2d) and child.kernel_size != (1, 1):
                self.hook_handles.append(
                    child.register_forward_hook(get_activation("Conv2d_" + str(index)))
                )
                index += 1

            # ReLU
            if isinstance(child, nn.ReLU):
                self.hook_handles.append(
                    child.register_forward_hook(get_activation("Relu_" + str(index)))
                )
                index += 1

            # Linear
            if isinstance(child, nn.Linear):
                self.hook_handles.append(
                    child.register_forward_hook(get_activation("Linear_" + str(index)))
                )
                index += 1

    # ------------------------------------------------------------------
    # ---------------------- CREATE POISON/CLEAN -----------------------
    # ------------------------------------------------------------------

    def create_targets(self, targets, label):
        """
        Assign new labels (Poison=101, Clean=102, etc.)
        """
        new_targets = torch.ones_like(targets) * label
        return new_targets.to(self.device)

    def generate_poison_clean_sets(self):
        """
        Generate Poison and Clean sets (merging NVT + NoT).
        """
        while self.poison_count < self.UNKNOWN_SIZE_POISON or self.clean_count < self.UNKNOWN_SIZE_CLEAN:
            for batch_idx, (inputs, labels) in enumerate(self.benign_unknown_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Call poison_transform to embed the trigger
                # (In original code, you have self.poison_transform; here we assume it exists)
                poisoned_inputs, poisoned_labels = self.poison_transform.transform(inputs, labels)
                preds_bd = torch.argmax(self.model(poisoned_inputs), dim=1)

                # Determine relevant indices
                victim_indices = (labels == self.VICTIM)
                non_victim_indices = (labels != self.VICTIM)

                # 1) Poison samples (Victim)
                if self.poison_count < self.UNKNOWN_SIZE_POISON:
                    label_value = self.label_mapping[self.POISON_TEMP_LABEL]
                    targets_poison = self.create_targets(labels, label_value)
                    correct_preds_indices = (preds_bd == self.target)
                    final_indices = victim_indices & correct_preds_indices

                    self.temp_poison_inputs_set.append(poisoned_inputs[final_indices].cpu())
                    self.temp_poison_labels_set.append(targets_poison[final_indices].cpu())
                    self.temp_poison_pred_set.append(preds_bd[final_indices].cpu())

                    self.poison_count += final_indices.sum().item()

                # 2) Clean samples (merge NVT + NoT, no trigger)
                if self.clean_count < self.UNKNOWN_SIZE_CLEAN:
                    label_value = self.label_mapping[self.CLEAN_TEMP_LABEL]
                    targets_clean = self.create_targets(labels, label_value)

                    self.temp_clean_inputs_set.append(inputs[non_victim_indices].cpu())
                    self.temp_clean_labels_set.append(targets_clean[non_victim_indices].cpu())
                    self.temp_clean_pred_set.append(preds_bd[non_victim_indices].cpu())

                    self.clean_count += non_victim_indices.sum().item()

                if self.poison_count >= self.UNKNOWN_SIZE_POISON and self.clean_count >= self.UNKNOWN_SIZE_CLEAN:
                    break

            if self.poison_count >= self.UNKNOWN_SIZE_POISON and self.clean_count >= self.UNKNOWN_SIZE_CLEAN:
                break

    # ------------------------------------------------------------------
    # ------------------- CREATE DATALOADERS FOR POISON/CLEAN ----------
    # ------------------------------------------------------------------
    def create_poison_clean_dataloaders(self):
        # Poison
        bd_inputs_set = torch.cat(self.temp_poison_inputs_set)[:self.UNKNOWN_SIZE_POISON]
        bd_labels_set = np.hstack(self.temp_poison_labels_set)[:self.UNKNOWN_SIZE_POISON]
        bd_pred_set = np.hstack(self.temp_poison_pred_set)[:self.UNKNOWN_SIZE_POISON]

        # Clean
        clean_inputs_set = torch.cat(self.temp_clean_inputs_set)[:self.UNKNOWN_SIZE_CLEAN]
        clean_labels_set = np.hstack(self.temp_clean_labels_set)[:self.UNKNOWN_SIZE_CLEAN]
        clean_pred_set = np.hstack(self.temp_clean_pred_set)[:self.UNKNOWN_SIZE_CLEAN]

        # Create dataset & loader
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

        # Poison loader
        poison_set = CustomDataset(bd_inputs_set, bd_labels_set)
        self.poison_loader = data.DataLoader(poison_set, batch_size=50, num_workers=2, shuffle=True)
        print("Poison set size:", len(self.poison_loader))

        # Clean loader
        clean_set = CustomDataset(clean_inputs_set, clean_labels_set)
        self.clean_loader = data.DataLoader(clean_set, batch_size=50, num_workers=2, shuffle=True)
        print("Clean set size:", len(self.clean_loader))

        # Delete temporary variables
        del bd_inputs_set, bd_labels_set, bd_pred_set
        del clean_inputs_set, clean_labels_set, clean_pred_set

    # ------------------------------------------------------------------
    # -------------------- HOOK & EXTRACT ACTIVATION -------------------
    # ------------------------------------------------------------------
    def fetch_activation(self, loader):
        print("Starting fetch_activation")
        self.model.eval()
        all_h_label = []
        pred_set = []
        h_batch = {}
        activation_container = {}

        # Run 1 batch to init keys (ensure self.activations is present)
        for (images, labels) in loader:
            print("Running the first batch to init hooks")
            _ = self.model(images.to(self.device))
            break

        # Initialize activation_container
        for key in self.activations:
            activation_container[key] = []

        # Clear self.activations for each batch
        self.activations.clear()

        for batch_idx, (images, labels) in enumerate(loader, start=1):
            print(f"Running batch {batch_idx} - Images shape: {images.shape}, Labels shape: {labels.shape}")
            try:
                output = self.model(images.to(self.device))
                print(f"Output shape: {output.shape}")
            except Exception as e:
                print(f"Error running model on batch {batch_idx}: {e}")
                break
            pred_set.append(torch.argmax(output, -1).to(self.device))

            # Save activation
            for key in self.activations:
                h_batch[key] = self.activations[key].view(images.shape[0], -1)
                # Save each sample
                for h in h_batch[key]:
                    activation_container[key].append(h.to(self.device))

            # Save ori_label
            for label_ in labels:
                all_h_label.append(label_.to(self.device))

            # Reset self.activations
            self.activations.clear()

            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx} batches")

        # Stack
        for key in activation_container:
            activation_container[key] = torch.stack(activation_container[key])
        all_h_label = torch.stack(all_h_label)
        pred_set = torch.cat(pred_set)

        print("Finished fetch_activation")
        return all_h_label, activation_container, pred_set

    # ------------------------------------------------------------------
    # ------------------------ ACCURACY CALC ---------------------------
    # ------------------------------------------------------------------
    def calculate_accuracy(self, ori_labels, preds):
        if len(ori_labels) == 0:
            return 0.0
        correct = torch.sum(ori_labels == preds).item()
        total = len(ori_labels)
        accuracy = (correct / total) * 100
        return accuracy

    # ------------------------------------------------------------------
    # ------------------------- VISUALIZATION --------------------------
    # ------------------------------------------------------------------
    def display_images_grid(self, images, predictions, title_prefix):
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
        # Save the figure in the designated directory
        save_path = os.path.join(self.save_dir, f"{title_prefix}_grid.png")
        plt.savefig(save_path, dpi=300)
        plt.show()

    # ------------------------------------------------------------------
    # ------------------ SOME TOPOLOGICAL FUNCTIONS --------------------
    # ------------------------------------------------------------------
    def gather_activation_into_class(self, target, h):
        """
        Group activations by class (0 -> Test_C-1)
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
        Sort by distance
        """
        item_ = item.reshape(1, item.shape[0])
        dev = self.device
        new_dis = pairwise_euclidean_distance(item_.to(dev), destinations.to(dev))
        _, indices_individual = torch.sort(new_dis)
        return indices_individual.to("cpu")

    def getDefenseRegion(self, final_prediction, h_defense_activation, processing_label, layer,
                         layer_test_region_individual):
        if layer not in layer_test_region_individual:
            layer_test_region_individual[layer] = {}
        layer_test_region_individual[layer][processing_label] = []

        self.candidate_[layer] = self.gather_activation_into_class(final_prediction, h_defense_activation)

        # Check if there are samples
        if np.ndim(self.candidate_[layer][processing_label]) == 0:
            print("No sample in this class for label =", processing_label)
        else:
            # Compute distance
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

    # ------------------------------------------------------------------
    # ----------------------- MAIN TEST METHOD -------------------------
    # ------------------------------------------------------------------
    def test(self):
        """
        Run the entire process: generate Poison/Clean, create dataloaders,
        hook, fetch activation, compute topological, PCA, and final results.
        """
        # 1) Generate Poison + Clean
        print('STEP 1')
        self.generate_poison_clean_sets()

        # 2) Create dataloaders for Poison and Clean
        print('STEP 2')
        self.create_poison_clean_dataloaders()

        # 3) Display some images (Poison & Clean)
        print('STEP 3')
        images_to_display = []
        predictions_to_display = []

        # Get 3 Poison images + 9 Clean images for illustration
        pairs = [
            (self.poison_loader, 3, "Poison Image"),
            (self.clean_loader, 9, "Clean Image")
        ]

        for loader, limit, prefix in pairs:
            count = 0
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                predictions = torch.argmax(self.model(inputs), dim=1).to('cpu')

                for input_image, pred_ in zip(inputs, predictions):
                    if count < limit:
                        images_to_display.append(input_image.unsqueeze(0))
                        predictions_to_display.append(pred_)
                        count += 1
                    else:
                        break
                if count >= limit:
                    break

            # Display
            self.display_images_grid(images_to_display, predictions_to_display, title_prefix=prefix)
            images_to_display.clear()
            predictions_to_display.clear()

        # 4) Fetch activation for Poison, Clean, Defense
        print('DEBUG')

        def print_loader_info(loader, name):
            print(f"{name} loader info:")
            print(f"Total samples: {len(loader.dataset)}")
            for batch in loader:
                inputs, labels = batch
                print(f"Batch - Inputs shape: {inputs.shape}, Labels shape: {labels.shape}")
                print(f"Sample labels: {labels[:5]}")
                # Display 3 samples
                for i in range(3):
                    img = inputs[i].squeeze().cpu().numpy()
                    plt.imshow(img.transpose(1, 2, 0) if img.ndim == 3 else img, cmap="gray")
                    plt.title(f"{name} - Label: {labels[i]}")
                    plt.axis("off")
                    # Save each sample image
                    sample_path = os.path.join(self.save_dir, f"{name}_sample_{i}.png")
                    plt.savefig(sample_path, dpi=300)
                    plt.show()
                break  # Only display the first batch

        print_loader_info(self.poison_loader, "Poison")
        print_loader_info(self.clean_loader, "Clean")
        print_loader_info(self.defense_loader, "Defense")

        # Try to load one batch from defense_loader
        try:
            for images, labels in self.defense_loader:
                print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
                break  # Only load one batch
        except Exception as e:
            print(f"Error loading data from defense_loader: {e}")

        print(f"Using device: {self.device}")
        print(f"Model is on device: {next(self.model.parameters()).device}")

        print('DEBUG')
        print('STEP 4')
        self.h_defense_ori_labels, self.h_defense_activations, self.h_defense_preds = self.fetch_activation(self.defense_loader)
        self.h_poison_ori_labels, self.h_poison_activations, self.h_poison_preds = self.fetch_activation(self.poison_loader)
        self.h_clean_ori_labels, self.h_clean_activations, self.h_clean_preds = self.fetch_activation(self.clean_loader)
        print('DEBUG')

        # Print general info
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

        # 5) Calculate accuracy
        print('STEP 5')
        accuracy_defense = self.calculate_accuracy(self.h_defense_ori_labels, self.h_defense_preds)

        poison_GT = torch.ones_like(self.h_poison_preds) * self.target
        correct_poison = torch.sum(poison_GT == self.h_poison_preds).item()
        total_poison = len(self.h_poison_preds)
        accuracy_poison = (correct_poison / total_poison) * 100

        print(f"\nAccuracy on defense_loader (Clean): {accuracy_defense:.2f}%")
        print(f"Accuracy on poison_loader (Poison) : {accuracy_poison:.2f}%")

        # # 6) Apply UMAP for visualization (sample 20%)
        # print('STEP 6')
        # sample_rate = 0.2
        # 
        # total_bd = len(self.h_poison_activations[next(iter(self.h_poison_activations))])
        # total_defense = len(self.h_defense_activations[next(iter(self.h_defense_activations))])
        # 
        # bd_indices = choice(total_bd, int(total_bd * sample_rate), replace=False)
        # defense_indices = choice(total_defense, int(total_defense * sample_rate), replace=False)
        # 
        # print(total_bd)
        # print(total_defense)
        # 
        # # Create a function to plot with UMAP
        # def plot_activations(activations, labels, title):
        #     umap_2d = UMAP(random_state=0)
        #     projections = umap_2d.fit_transform(activations)
        #     df_classes = pd.DataFrame(labels)
        #     fig = px.scatter(
        #         projections, x=0, y=1,
        #         color=df_classes[0].astype(str), labels={'color': 'label'}
        #     )
        #     fig.update_layout(title=title)
        #     # Save figure as HTML and PNG
        #     html_path = os.path.join(self.save_dir, f"{title.replace(' ', '_')}.html")
        #     png_path = os.path.join(self.save_dir, f"{title.replace(' ', '_')}.png")
        #     fig.write_html(html_path)
        #     fig.write_image(png_path)
        #     fig.show()
        # 
        # # Prepare prefix label
        # h_bd_ori_labels_prefixed = [f"Poison {x.item()}" for x in self.h_poison_ori_labels]
        # 
        # for key in self.h_poison_activations:
        #     sampled_bd = self.h_poison_activations[key][bd_indices]
        #     sampled_defense = self.h_defense_activations[key][defense_indices]
        # 
        #     # Concatenate
        #     activations_concat = np.concatenate((sampled_bd.cpu(), sampled_defense.cpu()), axis=0)
        #     labels_concat = np.concatenate((
        #         np.array(h_bd_ori_labels_prefixed)[bd_indices],
        #         self.h_defense_ori_labels[defense_indices].cpu()
        #     ), axis=0)
        # 
        #     plot_activations(activations_concat, labels_concat, title=f"UMAP for {key}")

        # 7) Compute topological representation
        print('STEP 7')
        # Build representation for defense labels
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

        # Compute region distance for Poison
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

        # Compute region distance for Clean
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

        # 8) Aggregate all topological
        print('STEP 8')

        def aggregate_by_all_layers(output_label):
            inputs_container = []
            first_key = list(self.topological_representation.keys())[0]
            labels_container = np.repeat(output_label, len(self.topological_representation[first_key][output_label]))
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
            # Any label not "Poison"/"Clean" => belongs to benign
            if inx != self.POISON_TEMP_LABEL and inx != self.CLEAN_TEMP_LABEL:
                inputs_all_benign.append(np.array(inputs))
                labels_all_benign.append(np.array(labels))
            else:
                inputs_all_unknown.append(np.array(inputs))
                labels_all_unknown.append(np.array(labels))

        # Concatenate
        inputs_all_benign = np.concatenate(inputs_all_benign)
        labels_all_benign = np.concatenate(labels_all_benign)

        inputs_all_unknown = np.concatenate(inputs_all_unknown)
        labels_all_unknown = np.concatenate(labels_all_unknown)

        # 9) PCA & outlier detection
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
        # Save figure as HTML and PNG
        # pca_html_path = os.path.join(self.save_dir, "PCA_scatter.html")
        # pca_png_path = os.path.join(self.save_dir, "PCA_scatter.png")
        # fig_.write_html(pca_html_path)
        # fig_.write_image(pca_png_path)
        # fig_.show()

        pca = PCA(contamination=0.01, n_components='mle')
        pca.fit(inputs_all_benign)

        y_train_pred = pca.labels_
        y_train_scores = pca.decision_scores_
        y_train_scores = pca.decision_function(inputs_all_benign)
        y_train_pred = pca.predict(inputs_all_benign)

        y_test_scores = pca.decision_function(inputs_all_unknown)
        y_test_pred = pca.predict(inputs_all_unknown)
        prediction_mask = (y_test_pred == 1)
        prediction_labels = labels_all_unknown[prediction_mask]
        label_counts = Counter(prediction_labels)

        print("\n----------- DETECTION RESULTS -----------")
        for label, count in label_counts.items():
            print(f'Label {label}: {count}')

        # Calculate AUC
        # Convert (labels_all_unknown == POISON_TEMP_LABEL) -> 1
        is_poison_mask = (labels_all_unknown == self.POISON_TEMP_LABEL).astype(int)
        fpr, tpr, thresholds = metrics.roc_curve(is_poison_mask, y_test_scores, pos_label=1)
        auc_val = metrics.auc(fpr, tpr)

        # Confusion matrix
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
        In your codebase, 'detect()' is the main entry point.
        """
        self.test()

    def __del__(self):
        """
        Remove hooks to avoid memory leaks
        """
        for h in self.hook_handles:
            h.remove()
        torch.cuda.empty_cache()
