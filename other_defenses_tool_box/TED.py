import os
import torch
import config
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn import metrics
from sklearn.decomposition import PCA as sklearn_PCA
from sklearn.metrics import confusion_matrix
from pyod.models.pca import PCA
from other_defenses_tool_box.backdoor_defense import BackdoorDefense
from other_defenses_tool_box.tools import generate_dataloader
from utils.supervisor import get_transforms
from utils import supervisor, tools

'''
Run:
python other_defense.py -dataset mnist -poison_type badnet -poison_rate 0.1 -defense TED

Now we use dynamic labeling:
- If label != 0 => malicious (VT)
- If label == 0 => benign (NoT)
'''

class TED(BackdoorDefense):
    """TED Defense Method

    Changes from previous code:
    - Use torch.no_grad() during forward to save memory.
    - Reduce batch_size from 200 to 50 in create_loader_from_numpy.
    - Move activations to CPU immediately.
    - Use torch.cuda.empty_cache() after large operations.
    """

    name = 'TED'

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.model.eval()

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Create train, val, and test loaders
        self.train_loader = generate_dataloader(dataset=self.dataset,
                                                dataset_path=config.data_dir,
                                                batch_size=50,  # reduced batch size
                                                split='train',
                                                data_transform=self.data_transform,
                                                shuffle=True,
                                                drop_last=False,
                                                noisy_test=False)

        self.val_loader = generate_dataloader(dataset=self.dataset,
                                              dataset_path=config.data_dir,
                                              batch_size=50, # reduced batch size
                                              split='val',
                                              data_transform=self.data_transform,
                                              shuffle=True,
                                              drop_last=False,
                                              noisy_test=False)

        self.test_loader = generate_dataloader(dataset=self.dataset,
                                               dataset_path=config.data_dir,
                                               batch_size=50, # reduced batch size
                                               split='test',
                                               data_transform=self.data_transform,
                                               shuffle=False,
                                               drop_last=False,
                                               noisy_test=False)

        # Take 10% of train data as defense_loader (benign)
        train_data, train_labels = self._collect_data_from_loader(self.train_loader)
        total_indices = np.arange(len(train_data))
        np.random.shuffle(total_indices)
        defense_size = int(0.1 * len(train_data))
        defense_indices = total_indices[:defense_size]
        defense_data = train_data[defense_indices]
        defense_labels = train_labels[defense_indices]
        self.defense_loader = self._create_loader_from_numpy(defense_data, defense_labels)

        self._register_hooks()
        self._test_labels = None

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

    def _create_loader_from_numpy(self, data, labels, batch_size=50): # smaller batch
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
                # store on GPU, will move to CPU later
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

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # no_grad already applied above
            preds = self.model(images)
            preds = torch.argmax(preds, dim=1)

            # Now activations are in self.activations
            # Move them to CPU immediately
            for key in self.activations:
                h_batch = self.activations[key].view(images.shape[0], -1).cpu()
                if key not in activation_container:
                    activation_container[key] = []
                activation_container[key].append(h_batch)

            pred_set.append(preds.cpu())
            all_labels.append(labels.cpu())

            # Clear GPU cache after each batch
            torch.cuda.empty_cache()

        if len(all_labels) == 0:
            return None, {}, None

        for key in activation_container:
            activation_container[key] = torch.cat(activation_container[key], dim=0)

        all_labels = torch.cat(all_labels, dim=0)
        pred_set = torch.cat(pred_set, dim=0)

        # empty cache
        torch.cuda.empty_cache()
        return all_labels, activation_container, pred_set

    def _get_poisoned_flags(self):
        labels = self._test_labels
        flags = (labels != 0).int()
        return flags

    @torch.no_grad()
    def _build_topological_features(self, activation_container, pred_set, ref_data, ref_labels):
        # Now activation_container and ref_data are on CPU (since we moved them)
        keys = list(activation_container.keys())
        features = []

        # Convert all to NumPy for faster distance computation with NumPy ops
        ref_labels_np = ref_labels.numpy()
        ref_act = {k: ref_data[k].numpy() for k in ref_data}
        pred_np = pred_set.numpy()

        for i in range(activation_container[keys[0]].shape[0]):
            x_feature = []
            x_pred = pred_np[i]

            same_class_ref_indices = np.where(ref_labels_np == x_pred)[0]
            if len(same_class_ref_indices) == 0:
                x_feature = [999] * len(keys)
                features.append(x_feature)
                continue

            x_feature = []
            for lkey in keys:
                x_vec = activation_container[lkey][i].numpy()
                dist_all = np.sqrt(np.sum((ref_act[lkey] - x_vec)**2, axis=1))
                sorted_indices = np.argsort(dist_all)
                sc_dist = dist_all[same_class_ref_indices]
                min_sc_idx = np.argmin(sc_dist)
                nearest_sc_ref = same_class_ref_indices[min_sc_idx]
                rank = np.where(sorted_indices == nearest_sc_ref)[0][0]
                x_feature.append(rank)

            features.append(x_feature)

        return np.array(features)

    def test(self):
        print("Fetching defense activations...")
        h_defense_ori_labels, h_defense_activations, h_defense_preds = self._fetch_activation(self.defense_loader)
        print("Fetching test activations...")
        h_test_ori_labels, h_test_activations, h_test_preds = self._fetch_activation(self.test_loader)
        self._test_labels = h_test_ori_labels

        def calculate_accuracy(ori_labels, preds):
            if ori_labels is None or preds is None:
                return 0
            correct = torch.sum(ori_labels == preds)
            total = len(ori_labels)
            return (correct.float() / total * 100).item() if total > 0 else 0

        accuracy_defense = calculate_accuracy(h_defense_ori_labels, h_defense_preds)
        print(f"Accuracy on defense_loader: {accuracy_defense:.2f}%")

        if h_test_ori_labels is None:
            print("No test data.")
            return

        y_true = self._get_poisoned_flags().numpy()

        if len(h_test_activations) == 0 or len(h_defense_activations) == 0:
            print("No activation data.")
            return

        print("Building topological features for defense set...")
        defense_features = self._build_topological_features(h_defense_activations, h_defense_preds, h_defense_activations, h_defense_ori_labels)

        print("Building topological features for test set...")
        test_features = self._build_topological_features(h_test_activations, h_test_preds, h_defense_activations, h_defense_ori_labels)

        print("Training PCA-based detector...")
        detector = PCA(contamination=0.01)
        detector.fit(defense_features)
        y_test_scores = detector.decision_function(test_features)
        y_test_pred = detector.predict(test_features)

        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_test_scores, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        tn, fp, fn, tp = confusion_matrix(y_true, y_test_pred).ravel()
        TPR = tp / (tp + fn) if (tp + fn) > 0 else 0
        FPR = fp / (fp + tn) if (fp + tn) > 0 else 0
        myf1 = metrics.f1_score(y_true, y_test_pred)

        print("TPR: {:.2f}%".format(TPR * 100))
        print("FPR: {:.2f}%".format(FPR * 100))
        print("AUC: {:.4f}".format(auc))
        print(f"f1 score: {myf1:.4f}")

    def detect(self):
        self.test()

    def __del__(self):
        for handle in self.hook_handles:
            handle.remove()
        torch.cuda.empty_cache()

