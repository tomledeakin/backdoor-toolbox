# This is the test code of TED defense.   
# Robust Backdoor Detection for Deep Learning via Topological Evolution Dynamics [IEEE, 2024]
# (https://arxiv.org/abs/2312.02673)

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

# Thêm thư viện StandardScaler
from sklearn.preprocessing import StandardScaler

from utils import supervisor, tools
from utils.supervisor import get_transforms
from other_defenses_tool_box.tools import generate_dataloader
from other_defenses_tool_box.backdoor_defense import BackdoorDefense

class TED(BackdoorDefense):
    def __init__(self, args):
        super().__init__(args)  # Gọi constructor của BackdoorDefense
        self.args = args

        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Backdoor target class
        self.target = self.target_class
        print(f"Target Class: {self.target}")

        # (1) Dataloader
        self.train_loader = generate_dataloader(
            dataset=self.dataset,
            dataset_path=config.data_dir,
            batch_size=50,
            split="train",
            data_transform=self.data_transform,
            shuffle=True,
            drop_last=False,
            noisy_test=False
        )

        # Scan unique classes
        all_labels = []
        for _, labels in self.train_loader:
            all_labels.extend(labels.tolist())
        unique_classes = set(all_labels)
        num_classes = len(unique_classes)
        self.num_classes = num_classes

        print(f"Number of unique classes (train_loader scan): {num_classes}")
        print(f"Number of classes from args: {self.num_classes}")

        self.DEFENSE_TRAIN_SIZE = num_classes * 40

        # (2) test_loader
        self.test_loader = generate_dataloader(
            dataset=self.dataset,
            dataset_path=config.data_dir,
            batch_size=50,
            split="test",
            data_transform=self.data_transform,
            shuffle=False,
            drop_last=False,
            noisy_test=False
        )
        self.testset = self.test_loader.dataset

        # (3) number sample
        self.NUM_SAMPLES = 500

        # (4) create defense_loader
        trainset = self.train_loader.dataset
        indices = np.arange(len(trainset))
        _, defense_subset_indices = train_test_split(indices, test_size=0.1, random_state=42)
        defense_subset = data.Subset(trainset, defense_subset_indices)
        self.defense_loader = data.DataLoader(
            defense_subset,
            batch_size=50,
            num_workers=2,
            shuffle=True
        )

        # (5) filter defense => keep correct predicted
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

        # Rebuild
        defense_subset = data.Subset(trainset, benign_indices)
        self.defense_loader = data.DataLoader(
            defense_subset,
            batch_size=50,
            num_workers=2,
            shuffle=True
        )

        # (6) define label mapping
        self.POISON_TEMP_LABEL = "Poison"
        self.CLEAN_TEMP_LABEL = "Clean"
        self.label_mapping = {"Poison": 101, "Clean": 102}

        # counters
        self.poison_count = 0
        self.clean_count = 0

        # Temp containers
        self.temp_poison_inputs_set = []
        self.temp_poison_labels_set = []
        self.temp_poison_pred_set = []

        self.temp_clean_inputs_set = []
        self.temp_clean_labels_set = []
        self.temp_clean_pred_set = []

        # Hooks
        self.hook_handles = []
        self.activations = {}
        self.register_hooks()

        self.Test_C = num_classes + 2  # Extra for "Poison" & "Clean"
        self.topological_representation = {}
        self.candidate_ = {}

        self.save_dir = f"TED/{self.dataset}/{self.poison_type}"
        os.makedirs(self.save_dir, exist_ok=True)

    def register_hooks(self):
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook

        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

        net_children = self.model.modules()
        index = 0
        for child in net_children:
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
        new_targets = torch.ones_like(targets) * label
        return new_targets.to(self.device)

    def generate_poison_clean_sets(self):
        all_indices = np.arange(len(self.testset))
        if len(all_indices) < self.NUM_SAMPLES:
            chosen = all_indices
        else:
            chosen = np.random.choice(all_indices, size=self.NUM_SAMPLES, replace=False)

        subset = data.Subset(self.testset, chosen)
        loader = data.DataLoader(subset, batch_size=50, shuffle=False)

        # Clean
        for inputs, labels in loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            label_value = self.label_mapping[self.CLEAN_TEMP_LABEL]
            targets_clean = self.create_targets(labels, label_value)
            preds = torch.argmax(self.model(inputs), dim=1)

            self.temp_clean_inputs_set.append(inputs.cpu())
            self.temp_clean_labels_set.append(targets_clean.cpu())
            self.temp_clean_pred_set.append(preds.cpu())

            self.clean_count += labels.size(0)

        # Poison
        poison_loader = data.DataLoader(subset, batch_size=50, shuffle=False)
        for inputs, labels in poison_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            poisoned_inputs, poisoned_labels = self.poison_transform.transform(inputs, labels)
            preds_bd = torch.argmax(self.model(poisoned_inputs), dim=1)

            label_value = self.label_mapping[self.POISON_TEMP_LABEL]
            targets_poison = self.create_targets(labels, label_value)

            self.temp_poison_inputs_set.append(poisoned_inputs.cpu())
            self.temp_poison_labels_set.append(targets_poison.cpu())
            self.temp_poison_pred_set.append(preds_bd.cpu())

            self.poison_count += labels.size(0)

        print(f"Finished generate_poison_clean_sets. Clean_count={self.clean_count}, Poison_count={self.poison_count}")

    def create_poison_clean_dataloaders(self):
        bd_inputs_set = torch.cat(self.temp_poison_inputs_set, dim=0)
        bd_labels_set = np.hstack(self.temp_poison_labels_set)
        bd_pred_set = np.hstack(self.temp_poison_pred_set)

        clean_inputs_set = torch.cat(self.temp_clean_inputs_set, dim=0)
        clean_labels_set = np.hstack(self.temp_clean_labels_set)
        clean_pred_set = np.hstack(self.temp_clean_pred_set)

        class CustomDataset(data.Dataset):
            def __init__(self, data_, labels_):
                super().__init__()
                self.images = data_
                self.labels = labels_

            def __len__(self):
                return len(self.images)

            def __getitem__(self, index):
                return self.images[index], self.labels[index]

        poison_set = CustomDataset(bd_inputs_set, bd_labels_set)
        self.poison_loader = data.DataLoader(poison_set, batch_size=50, num_workers=2, shuffle=True)
        print("Poison set size:", len(self.poison_loader))

        clean_set = CustomDataset(clean_inputs_set, clean_labels_set)
        self.clean_loader = data.DataLoader(clean_set, batch_size=50, num_workers=2, shuffle=True)
        print("Clean set size:", len(self.clean_loader))

        del bd_inputs_set, bd_labels_set, bd_pred_set
        del clean_inputs_set, clean_labels_set, clean_pred_set

    def fetch_activation(self, loader):
        print("Starting fetch_activation")
        self.model.eval()

        all_h_label = []
        pred_set = []
        activation_container = {}
        h_batch = {}

        # init hook
        for images, labels in loader:
            _ = self.model(images.to(self.device))
            break

        for k in self.activations:
            activation_container[k] = []
        self.activations.clear()

        for batch_idx, (images, labels) in enumerate(loader, start=1):
            try:
                out = self.model(images.to(self.device))
            except Exception as e:
                print(f"Error in batch {batch_idx}, {e}")
                break

            pred_set.append(torch.argmax(out, dim=1).cpu())
            for k in self.activations:
                h_batch[k] = self.activations[k].view(images.shape[0], -1).cpu()
                activation_container[k].append(h_batch[k])

            all_h_label.append(labels.cpu())
            self.activations.clear()

        for k in activation_container:
            activation_container[k] = torch.cat(activation_container[k], dim=0)
        all_h_label = torch.cat(all_h_label, dim=0)
        pred_set = torch.cat(pred_set, dim=0)

        print("Finished fetch_activation")
        return all_h_label, activation_container, pred_set

    def calculate_accuracy(self, ori_labels, preds):
        if len(ori_labels) == 0:
            return 0.0
        correct = torch.sum(ori_labels == preds).item()
        return 100.0 * correct / len(ori_labels)

    def display_images_grid(self, images, predictions, title_prefix):
        num_images = len(images)
        if num_images == 0:
            return
        cols = 3
        rows = (num_images + cols - 1) // cols
        plt.figure(figsize=(cols*2, rows*2))

        for i, (img, pred_) in enumerate(zip(images, predictions)):
            plt.subplot(rows, cols, i+1)
            img = img.squeeze().cpu().numpy()
            if img.ndim == 3:
                plt.imshow(np.transpose(img, (1,2,0)))
            else:
                plt.imshow(img, cmap='gray')
            plt.title(f"{title_prefix} {i+1}\nPred={pred_.item()}")
            plt.axis("off")

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"{title_prefix}_grid.png")
        plt.savefig(save_path, dpi=300)
        plt.show()

    def gather_activation_into_class(self, target, h):
        """
        Gom h => list [0..Test_C], h_c_c[c] = activation của class c
        """
        h_c_c = [None for _ in range(self.Test_C)]
        for c in range(self.Test_C):
            idxs = (target == c).nonzero(as_tuple=True)[0]
            if len(idxs) > 0:
                h_c_c[c] = h[idxs, :]
            else:
                h_c_c[c] = None
        return h_c_c

    def test(self):
        print("STEP 1")
        self.generate_poison_clean_sets()

        print("STEP 2")
        self.create_poison_clean_dataloaders()

        print("STEP 3")
        # Just to visualize
        pairs = [
            (self.poison_loader, 3, "Poison Image"),
            (self.clean_loader, 9, "Clean Image")
        ]
        for loader, limit, prefix in pairs:
            images_disp, preds_disp = [], []
            count = 0
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                preds = torch.argmax(self.model(inputs), dim=1).cpu()
                for im, pr in zip(inputs, preds):
                    if count < limit:
                        images_disp.append(im.unsqueeze(0))
                        preds_disp.append(pr)
                        count += 1
                    else:
                        break
                if count>=limit:
                    break
            self.display_images_grid(images_disp, preds_disp, prefix)

        # debug
        print(f"Using device: {self.device}")
        print(f"Model param device: {next(self.model.parameters()).device}")

        print("STEP 4")
        # get activation
        self.h_defense_ori_labels, self.h_defense_activations, self.h_defense_preds = self.fetch_activation(self.defense_loader)
        self.h_poison_ori_labels, self.h_poison_activations, self.h_poison_preds = self.fetch_activation(self.poison_loader)
        self.h_clean_ori_labels, self.h_clean_activations, self.h_clean_preds = self.fetch_activation(self.clean_loader)

        # union test
        self.union_test_activations = {}
        self.union_test_preds = torch.cat([self.h_poison_preds, self.h_clean_preds], dim=0)
        for layer_name in self.h_poison_activations:
            if layer_name not in self.h_clean_activations:
                continue
            self.union_test_activations[layer_name] = torch.cat(
                [self.h_poison_activations[layer_name], self.h_clean_activations[layer_name]],
                dim=0
            )

        print("STEP 5")
        accuracy_defense = self.calculate_accuracy(self.h_defense_ori_labels, self.h_defense_preds)

        poison_GT = torch.ones_like(self.h_poison_preds)*self.target
        correct_poison = torch.sum(poison_GT == self.h_poison_preds).item()
        total_poison = len(self.h_poison_preds)
        accuracy_poison = 100.0 * correct_poison / total_poison

        print(f"Accuracy on defense_loader: {accuracy_defense:.2f}%")
        print(f"Attack success rate on poison_loader: {accuracy_poison:.2f}%")

        print("STEP 6) Tính centroid cho defense_loader")
        self.centroids_defense = {}
        for layer_name, act_mat in self.h_defense_activations.items():
            per_class = self.gather_activation_into_class(self.h_defense_preds, act_mat)
            self.centroids_defense[layer_name] = {}
            for c in range(self.Test_C):
                if per_class[c] is not None:
                    self.centroids_defense[layer_name][c] = per_class[c].mean(dim=0)  
                else:
                    self.centroids_defense[layer_name][c] = None

        print("STEP 7) Tìm representative trong union_test (closest to centroid)")
        self.representative_points_test = {}
        for layer_name, centroid_dict in self.centroids_defense.items():
            self.representative_points_test[layer_name] = {}
            if layer_name not in self.union_test_activations:
                continue
            act_union = self.union_test_activations[layer_name]
            preds_union = self.union_test_preds

            for c, centroid_vec in centroid_dict.items():
                if centroid_vec is None:
                    self.representative_points_test[layer_name][c] = None
                    continue
                idxs_c = (preds_union == c).nonzero(as_tuple=True)[0]
                if len(idxs_c)==0:
                    self.representative_points_test[layer_name][c] = None
                    continue

                dev = self.device
                centroid_vec_ = centroid_vec.unsqueeze(0).to(dev)
                cand_ = act_union[idxs_c].to(dev)
                dists = pairwise_euclidean_distance(centroid_vec_, cand_).squeeze(0)
                min_local = torch.argmin(dists).item()
                min_global = idxs_c[min_local]

                rep_point = act_union[min_global].clone().detach()
                self.representative_points_test[layer_name][c] = rep_point.cpu()

        print("STEP 8) Tính khoảng cách topological (đo khoảng cách item->representative)")

        # Hai hàm dưới ta override logic
        def getDefenseRegion(final_prediction, h_defense_activation, processing_label, layer, layer_test_region_individual):
            if layer not in layer_test_region_individual:
                layer_test_region_individual[layer] = {}
            layer_test_region_individual[layer][processing_label] = []

            # Gom sample class = processing_label
            per_class = self.gather_activation_into_class(final_prediction, h_defense_activation)
            if per_class[processing_label] is None:
                return layer_test_region_individual

            rep_point = self.representative_points_test[layer].get(processing_label, None)
            if rep_point is None:
                return layer_test_region_individual

            dev = self.device
            rep_point_ = rep_point.unsqueeze(0).to(dev)
            for item in per_class[processing_label]:
                item_ = item.unsqueeze(0).to(dev)
                dist_ = pairwise_euclidean_distance(item_, rep_point_).item()
                layer_test_region_individual[layer][processing_label].append(dist_)

            return layer_test_region_individual

        def getLayerRegionDistance(new_prediction, new_activation, new_temp_label,
                                   h_defense_prediction, h_defense_activation,
                                   layer, layer_test_region_individual):
            if layer not in layer_test_region_individual:
                layer_test_region_individual[layer] = {}
            layer_test_region_individual[layer][new_temp_label] = []

            # Gom sample
            per_class = self.gather_activation_into_class(new_prediction, new_activation)
            labels_in_batch = torch.unique(new_prediction)

            for c_ in labels_in_batch:
                rep_point = self.representative_points_test[layer].get(c_.item(), None)
                if rep_point is None:
                    continue

                dev = self.device
                rep_point_ = rep_point.unsqueeze(0).to(dev)

                if per_class[c_] is None:
                    continue
                for item in per_class[c_]:
                    dist_ = pairwise_euclidean_distance(item.unsqueeze(0).to(dev), rep_point_).item()
                    layer_test_region_individual[layer][new_temp_label].append(dist_)

            return layer_test_region_individual

        # Thay đổi reference
        self.getDefenseRegion = getDefenseRegion
        self.getLayerRegionDistance = getLayerRegionDistance

        # => Tính cho defense
        class_names_defense = np.unique(self.h_defense_ori_labels.numpy())
        for lbl in class_names_defense:
            for layer in self.h_defense_activations:
                self.topological_representation = self.getDefenseRegion(
                    final_prediction=self.h_defense_preds,
                    h_defense_activation=self.h_defense_activations[layer],
                    processing_label=lbl,
                    layer=layer,
                    layer_test_region_individual=self.topological_representation
                )
                arr_ = self.topological_representation[layer][lbl]
                print(f"Topological Representation Label [{lbl}] & layer [{layer}]: {arr_}")
                if len(arr_)>0:
                    print(f"Mean: {np.mean(arr_)}\n")
                else:
                    print("Mean: None\n")

        # => Tính cho poison
        self.topological_representation = self.topological_representation  # reuse
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
            arr_poison = self.topological_representation[layer_][self.POISON_TEMP_LABEL]
            print(f"Topological Representation Label [{self.POISON_TEMP_LABEL}] & layer [{layer_}]: {arr_poison}")
            if len(arr_poison)>0:
                print(f"Mean: {np.mean(arr_poison)}\n")
            else:
                print("Mean: None\n")

        # => Tính cho clean
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
            arr_clean = self.topological_representation[layer_][self.CLEAN_TEMP_LABEL]
            print(f"Topological Representation Label [{self.CLEAN_TEMP_LABEL}] & layer [{layer_}]: {arr_clean}")
            if len(arr_clean)>0:
                print(f"Mean: {np.mean(arr_clean)}\n")
            else:
                print("Mean: None\n")

        print("STEP 9) Chuẩn bị cho PCA & Scaler")
        # Gom
        def aggregate_by_all_layers(output_label):
            first_key = list(self.topological_representation.keys())[0]
            arr_size = len(self.topological_representation[first_key][output_label])
            if arr_size==0:
                return None, None
            # sắp xếp shape cho PCA: 
            # chúng ta đang có shape [layer1 => (k khoảng cách), layer2 => (k khoảng cách), ...].
            # Mình sẽ xếp theo cột => (N, num_layers)
            # Ở đây, "N" có thể khác nhau do 1 layer => x sample => cẩn thận. 
            # Thông thường ta assume x sample = arr_size => fix cẩn thận:

            n_layers = len(self.topological_representation.keys())
            container = []
            for layer_ in self.topological_representation.keys():
                arr_ = self.topological_representation[layer_][output_label]
                if len(arr_)<arr_size:
                    # trường hợp mismatch => fix cắt (phòng hiếm)
                    pass
                arr_ = arr_[:arr_size]
                container.append(arr_)
            # (num_layers, arr_size) => transpose
            container = np.array(container, dtype=object)
            # do arr_ length = arr_size => ta convert sang 2D
            # cẩn thận python object => unify
            mat_ = []
            for row in container:
                row = np.array(row, dtype=float)
                mat_.append(row)
            mat_ = np.array(mat_)  # shape (num_layers, arr_size)
            mat_ = mat_.T
            labs_ = np.repeat(output_label, arr_size)
            return mat_, labs_

        inputs_all_benign = []
        labels_all_benign = []
        inputs_all_unknown = []
        labels_all_unknown = []

        # Thu thập label & layers
        first_layer = list(self.topological_representation.keys())[0]
        all_labels = list(self.topological_representation[first_layer].keys())

        for lb_ in all_labels:
            aggregated, labs_ = aggregate_by_all_layers(lb_)
            if aggregated is None:
                # empty => skip
                continue
            if lb_ not in [self.POISON_TEMP_LABEL, self.CLEAN_TEMP_LABEL]:
                inputs_all_benign.append(aggregated)
                labels_all_benign.append(labs_)
            else:
                inputs_all_unknown.append(aggregated)
                labels_all_unknown.append(labs_)

        if len(inputs_all_benign)==0:
            print("[WARNING] No benign data aggregated => skipping PCA.")
            return
        if len(inputs_all_unknown)==0:
            print("[WARNING] No unknown data aggregated => skipping outlier detection.")
            return

        inputs_all_benign = np.concatenate(inputs_all_benign, axis=0)
        labels_all_benign = np.concatenate(labels_all_benign, axis=0)

        inputs_all_unknown = np.concatenate(inputs_all_unknown, axis=0)
        labels_all_unknown = np.concatenate(labels_all_unknown, axis=0)

        if inputs_all_benign.size==0:
            print("[WARNING] inputs_all_benign is empty => skip.")
            return
        if inputs_all_unknown.size==0:
            print("[WARNING] inputs_all_unknown is empty => skip.")
            return

        print("Scaling => PCA => PyOD PCA")
        scaler = StandardScaler()
        inputs_all_benign = scaler.fit_transform(inputs_all_benign)
        inputs_all_unknown = scaler.transform(inputs_all_unknown)

        pca_t = sklearn_PCA(n_components=2)
        pca_fit = pca_t.fit(inputs_all_benign)

        benign_trajectories = pca_fit.transform(inputs_all_benign)
        trajectories = pca_fit.transform(np.concatenate((inputs_all_unknown, inputs_all_benign), axis=0))

        df_classes = pd.DataFrame(np.concatenate((labels_all_unknown, labels_all_benign), axis=0))

        fig_ = px.scatter(
            trajectories, x=0, y=1, 
            color=df_classes[0].astype(str), 
            labels={'color': 'digit'},
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
        for label, cnt in label_counts.items():
            print(f"Label {label}: {cnt}")

        is_poison_mask = (labels_all_unknown == self.POISON_TEMP_LABEL).astype(int)
        fpr, tpr, thresholds = metrics.roc_curve(is_poison_mask, y_test_scores, pos_label=1)
        auc_val = metrics.auc(fpr, tpr)

        tn, fp, fn, tp = confusion_matrix(is_poison_mask, y_test_pred).ravel()
        TPR = tp/(tp+fn) if (tp+fn)>0 else 0
        FPR = fp/(fp+tn) if (fp+tn)>0 else 0
        f1_ = metrics.f1_score(is_poison_mask, y_test_pred)

        print(f"TPR={TPR*100:.2f}%, FPR={FPR*100:.2f}%, AUC={auc_val:.4f}, F1={f1_:.4f}")
        print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        print("\n[INFO] TED run completed.")

    def detect(self):
        self.test()

    def __del__(self):
        for h in self.hook_handles:
            h.remove()
        torch.cuda.empty_cache()

