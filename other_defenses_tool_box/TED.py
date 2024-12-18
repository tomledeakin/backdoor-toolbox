import os
import torch
import config
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA as SKPCA
from other_defenses_tool_box.backdoor_defense import BackdoorDefense
from other_defenses_tool_box.tools import generate_dataloader
from utils.supervisor import get_transforms
from utils import supervisor, tools

class TED(BackdoorDefense):
    """
    TED Defense Method Implementation
    
    Updated logic based on ScaleUp and IBD_PSC:
    - If poison_type == 'TaCT': malicious samples have labels == config.source_class
    - Otherwise: malicious samples are those whose labels != poison_labels (after poison_transform)
    
    Additionally, uses poison_transform similar to IBD_PSC/ScaleUp to determine which samples are malicious.
    According to the original paper:
    - m = 20 for datasets like MNIST and CIFAR-10.
    - alpha (α) = 0.05.
    The threshold τ is computed dynamically from PCA scores.
    """

    name = 'TED'

    def __init__(self, args):
        super().__init__(args)
        self.model.eval()
        self.args = args

        # From the paper:
        # For MNIST and CIFAR-10 (10 classes), m=20 per class.
        # alpha = 0.05
        self.m = 20
        self.alpha = 0.05

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Assume poison_transform is provided by BackdoorDefense or args
        # self.poison_transform = self.args.poison_transform (if defined)
        # If not defined, ensure it's defined similarly to IBD_PSC or ScaleUp before proceeding.

        # Dataloaders
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

        self.val_loader = generate_dataloader(
            dataset=self.dataset,
            dataset_path=config.data_dir,
            batch_size=50,
            split='val',
            data_transform=self.data_transform,
            shuffle=True,
            drop_last=False,
            noisy_test=False
        )

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

        # According to the paper, we pick m=20 benign samples from each class.
        train_data, train_labels = self._collect_data_from_loader(self.train_loader)
        defense_data, defense_labels = self._sample_m_per_class(train_data, train_labels, self.m)
        self.defense_loader = self._create_loader_from_numpy(defense_data, defense_labels)

        # Register hooks to get layer activations
        self._register_hooks()
        self._test_labels = None
        self._poison_test_labels = None  # Will store poison labels for test data after transform

    def _collect_data_from_loader(self, loader):
        all_data = []
        all_labels = []
        for x, y in loader:
            all_data.append(x.cpu().numpy())
            all_labels.append(y.cpu().numpy())
        if len(all_data) == 0:
            return np.array([]), np.array([])
        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        return all_data, all_labels

    def _sample_m_per_class(self, data, labels, m):
        classes = np.unique(labels)
        selected_data = []
        selected_labels = []
        for c in classes:
            c_indices = np.where(labels == c)[0]
            # Ensure we have at least m samples per class.
            if len(c_indices) < m:
                chosen_indices = c_indices
            else:
                chosen_indices = np.random.choice(c_indices, m, replace=False)
            selected_data.append(data[chosen_indices])
            selected_labels.append(labels[chosen_indices])

        selected_data = np.concatenate(selected_data, axis=0)
        selected_labels = np.concatenate(selected_labels, axis=0)
        return selected_data, selected_labels

    def _create_loader_from_numpy(self, data, labels, batch_size=50):
        tensor_data = torch.from_numpy(data)
        tensor_labels = torch.from_numpy(labels)
        dataset = torch.utils.data.TensorDataset(tensor_data, tensor_labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader

    def _register_hooks(self):
        self.hook_handles = []
        self.activations = {}

        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook

        index = 0
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Linear)):
                handle = module.register_forward_hook(get_activation(f"{module.__class__.__name__}_{index}"))
                self.hook_handles.append(handle)
                index += 1

    @torch.no_grad()
    def _fetch_activation(self, loader):
        self.model.eval()
        all_labels = []
        pred_set = []
        activation_container = {}

        # Clear previous activations
        self.activations.clear()

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            preds = self.model(images)
            preds_class = torch.argmax(preds, dim=1)

            # Move activations to CPU
            for key in self.activations:
                h_batch = self.activations[key].view(images.shape[0], -1).cpu()
                if key not in activation_container:
                    activation_container[key] = []
                activation_container[key].append(h_batch)

            pred_set.append(preds_class.cpu())
            all_labels.append(labels.cpu())
            torch.cuda.empty_cache()

        if len(all_labels) == 0:
            return None, {}, None

        for key in activation_container:
            activation_container[key] = torch.cat(activation_container[key], dim=0)

        all_labels = torch.cat(all_labels, dim=0)
        pred_set = torch.cat(pred_set, dim=0)
        torch.cuda.empty_cache()
        return all_labels, activation_container, pred_set

    def _get_poisoned_flags(self):
        labels = self._test_labels
        if self.args.poison_type == 'TaCT':
            # If TaCT: malicious samples are those whose labels == config.source_class
            return (labels == config.source_class).int()
        else:
            # Otherwise: malicious samples are those whose labels != poison_labels
            # (Derived from ScaleUp logic: 
            #  For non-TaCT attacks, we consider samples malicious if their label changes after poison_transform)
            
            print(f'labels: {labels}')
            print('----------------------')
            print(f'self._poison_test_labels: {self._poison_test_labels}')
            return (labels != self._poison_test_labels).int()

    def _euclidean_distance(self, x, Y):
        # x: 1D, Y: 2D
        return np.sqrt(np.sum((Y - x)**2, axis=1))

    @torch.no_grad()
    def _build_rank_features(self, activations, preds, ref_activations, ref_labels):
        keys = list(activations.keys())
        features = []
        ref_labels_np = ref_labels.numpy()
        ref_act_np = {k: ref_activations[k].numpy() for k in ref_activations}
        preds_np = preds.numpy()

        for i in range(activations[keys[0]].shape[0]):
            x_pred = preds_np[i]
            same_class_ref_indices = np.where(ref_labels_np == x_pred)[0]

            # If no same-class reference samples found, skip this sample
            if len(same_class_ref_indices) == 0:
                continue

            x_features = []
            for lkey in keys:
                x_vec = activations[lkey][i].numpy()
                dist_all = self._euclidean_distance(x_vec, ref_act_np[lkey])
                sorted_indices = np.argsort(dist_all)
                sc_dists = dist_all[same_class_ref_indices]
                nearest_sc_idx = same_class_ref_indices[np.argmin(sc_dists)]
                rank = np.where(sorted_indices == nearest_sc_idx)[0][0]
                x_features.append(rank)
            features.append(x_features)

        features = np.array(features)
        return features

    def _compute_reconstruction_errors(self, features, pca):
        # Compute reconstruction error
        projected = pca.transform(features)
        reconstructed = pca.inverse_transform(projected)
        errors = np.sum((features - reconstructed)**2, axis=1)
        return errors

    def test(self):
        # First, fetch the test data and produce poison_imgs, poison_labels for entire test set
        clean_imgs_list = []
        labels_list = []
        for batch in self.test_loader:
            clean_img, labels = batch
            clean_imgs_list.append(clean_img)
            labels_list.append(labels)
        clean_imgs = torch.cat(clean_imgs_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        # Apply poison_transform to entire test set
        poison_imgs, poison_labels = self.poison_transform.transform(clean_imgs, labels)
        self._test_labels = labels
        self._poison_test_labels = poison_labels

        # Create a loader from poison_imgs to fetch activations
        # Note: We only need activations from clean or from what is the original approach?
        # TED uses the model activations on the original (clean) test samples to build rank features.
        # We'll adhere to original logic: activations from clean test set.
        test_dataset = torch.utils.data.TensorDataset(clean_imgs, labels)
        test_loader_activations = torch.utils.data.DataLoader(test_dataset, batch_size=50, shuffle=False)
        
        # Fetch reference set activations
        h_defense_labels, h_defense_activations, h_defense_preds = self._fetch_activation(self.defense_loader)
        if h_defense_labels is None:
            print("No defense data available.")
            return

        # Fetch test set activations from clean data
        h_test_labels, h_test_activations, h_test_preds = self._fetch_activation(test_loader_activations)
        if h_test_labels is None:
            print("No test data available.")
            return

        # Build rank features for defense set
        defense_features = self._build_rank_features(h_defense_activations, h_defense_preds, h_defense_activations, h_defense_labels)
        if defense_features.shape[0] == 0:
            print("No valid defense samples for rank features.")
            return

        # Fit PCA model
        pca = SKPCA(n_components=min(defense_features.shape[1], defense_features.shape[0]))
        pca.fit(defense_features)

        # Compute defense scores
        defense_scores = self._compute_reconstruction_errors(defense_features, pca)
        # alpha = 0.05, find tau
        tau = np.percentile(defense_scores, 100*(1 - self.alpha))

        # Build rank features for test set
        test_features = self._build_rank_features(h_test_activations, h_test_preds, h_defense_activations, h_defense_labels)
        if test_features.shape[0] == 0:
            print("No valid test samples for rank features.")
            return

        y_test_scores = self._compute_reconstruction_errors(test_features, pca)
        y_test_pred = (y_test_scores > tau).astype(int)
        y_true = self._get_poisoned_flags().numpy()

        # Check for NaN/Inf
        if not np.isfinite(y_test_scores).all():
            finite_mask = np.isfinite(y_test_scores)
            y_test_scores = y_test_scores[finite_mask]
            y_test_pred = y_test_pred[finite_mask]
            y_true = y_true[finite_mask]
            print("Warning: Removed non-finite values from test scores.")

        if len(y_test_scores) == 0:
            print("No valid test scores after removing non-finite values.")
            return

        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_test_scores, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        tn, fp, fn, tp = confusion_matrix(y_true, y_test_pred).ravel()
        TPR = tp / (tp + fn) if (tp + fn) > 0 else 0
        FPR = fp / (fp + tn) if (fp + tn) > 0 else 0
        f1 = metrics.f1_score(y_true, y_test_pred)

        print(f"TN: {tn}")
        print(f"FP: {fp}")
        print(f"FN: {fn}")
        print(f"TP: {tp}")
        print("TPR: {:.2f}%".format(TPR * 100))
        print("FPR: {:.2f}%".format(FPR * 100))
        print("AUC: {:.4f}".format(auc))
        print(f"F1 score: {f1:.4f}")

        malicious_indices = np.where(y_test_pred == 1)[0]
        print("Number of malicious samples detected:", len(malicious_indices))

    def detect(self):
        self.test()

    def __del__(self):
        for handle in self.hook_handles:
            handle.remove()
        torch.cuda.empty_cache()
