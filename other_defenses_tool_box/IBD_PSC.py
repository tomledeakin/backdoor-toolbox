# This is the test code of IBD-PSC defense.
# IBD-PSC: Input-level Backdoor Detection via Parameter-oriented Scaling Consistency [ICML, 2024] (https://arxiv.org/abs/2405.09786)

import os
import pdb
import torch
import config
import torchvision
from sklearn import metrics
from tqdm import tqdm
import copy
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import Counter
import torch.nn as nn
import numpy as np

from other_defenses_tool_box.backdoor_defense import BackdoorDefense
from other_defenses_tool_box.tools import generate_dataloader
from utils.supervisor import get_transforms
from utils import supervisor, tools

'''
Run command example:
python other_defense.py -dataset cifar10 -poison_type badnet -poison_rate 0.1 -defense IBD_PSC
'''

class IBD_PSC(BackdoorDefense):
    """
    IBD-PSC Defense Method Implementation
    
    Updated with more debug info:
    - Print dataset sizes (train/val/test not shown previously, but we have val/test)
    - Explain why total samples become 16000 (clean + poison scenario)
    - Print out number of clean/poison samples considered
    - Save debug data if needed

    Args:
        n (int): Hyper-parameter for number of parameter-amplified model versions.
        xi (float): Hyper-parameter for error rate threshold.
        T (float): Threshold for PSC(x). If PSC(x) > T, it's considered a backdoor sample.
        scale (float): Scaling factor for amplifying BN layers.
    """

    name: str = 'IBD_PSC'

    def __init__(self, args, n=5, xi=0.6, T=0.9, scale=1.5):
        super().__init__(args)
        self.model.eval()
        self.args = args
        self.n = n
        self.xi = xi
        self.T = T
        self.scale = scale

        # Create a directory for debugging data if not exist
        if not os.path.exists('debug_data'):
            os.makedirs('debug_data')

        # Generate dataloaders
        self.test_loader = generate_dataloader(dataset=self.dataset,
                                               dataset_path=config.data_dir,
                                               batch_size=200,
                                               split='test',
                                               data_transform=self.data_transform,
                                               shuffle=False,
                                               drop_last=False,
                                               noisy_test=False
                                               )

        self.val_loader = generate_dataloader(dataset=self.dataset,
                                              dataset_path=config.data_dir,
                                              batch_size=200,
                                              split='val',
                                              data_transform=self.data_transform,
                                              shuffle=True,
                                              drop_last=False,
                                              noisy_test=False
                                              )

        # Print dataset sizes for debugging
        print("[DEBUG] Dataset and Loader Information:")
        print(f"Dataset: {self.dataset}")
        val_count = sum([batch[0].shape[0] for batch in self.val_loader])
        test_count = sum([batch[0].shape[0] for batch in self.test_loader])
        print(f"Validation set size: {val_count}")
        print(f"Test set size: {test_count}")
        # Note: The paper might have trained the model somewhere else, we focus on detection here.

        layer_num = self.count_BN_layers()
        print(f"[DEBUG] Number of BN layers: {layer_num}")
        sorted_indices = list(range(layer_num))
        sorted_indices = list(reversed(sorted_indices))
        self.sorted_indices = sorted_indices
        self.start_index = self.prob_start(self.scale, self.sorted_indices)
        print(f"[DEBUG] Start index determined by prob_start: {self.start_index}")

        total_num = 0
        clean_correct = 0
        bd_correct = 0
        bd_all = 0
        bd_predicts = []
        clean_predicts = []

        # Process test dataset to understand baseline performance
        for idx, batch in enumerate(self.test_loader):
            clean_img = batch[0]
            labels = batch[1]

            # Print batch shape info for debugging
            print(f"[DEBUG] Batch {idx}: clean_img shape={clean_img.shape}, labels shape={labels.shape}")

            total_num += labels.shape[0]
            clean_img = clean_img.cuda()  
            labels = labels.cuda()  

            target_flag = labels != 0
            poison_imgs, poison_labels = self.poison_transform.transform(clean_img[target_flag], labels[target_flag])

            bd_logits = self.model(poison_imgs)
            clean_logits = self.model(clean_img)

            clean_pred = torch.argmax(clean_logits, dim=1)
            poison_pred = torch.argmax(bd_logits, dim=1)

            clean_predicts.extend(clean_pred.cpu().tolist())
            bd_predicts.extend(poison_pred.cpu().tolist())

            # Depending on poison_type, determine correct counting method
            if self.args.poison_type == 'TaCT':
                mask = torch.eq(labels[target_flag], config.source_class)
                plabels = poison_labels[mask.clone()]
                ppred = poison_pred[mask.clone()]
                bd_correct += torch.sum(plabels == ppred)
                bd_all += plabels.size(0)
            else:
                bd_correct += torch.sum(poison_labels == poison_pred)
                bd_all += poison_labels.shape[0]

            clean_correct += torch.sum(labels == clean_pred)

        print(f'ba: {clean_correct * 100. / total_num}')  # Benign accuracy
        print(f'asr: {bd_correct * 100. / bd_all}')       # Attack success rate
        print(f'target label: {poison_labels[0:1]}')
        # Here we only processed the dataset once in baseline manner.

    def count_BN_layers(self):
        # Count how many BatchNorm2d layers in the model
        layer_num = 0
        for (name1, module1) in self.model.named_modules():
            if isinstance(module1, torch.nn.BatchNorm2d):
                layer_num += 1
        return layer_num

    def scale_var_index(self, index_bn, scale=1.5):
        # Create a copy of the model and scale the BN layers specified by index_bn
        copy_model = copy.deepcopy(self.model)
        index = -1
        for (name1, module1) in copy_model.named_modules():
            if isinstance(module1, torch.nn.BatchNorm2d):
                index += 1
                if index in index_bn:
                    module1.weight.data *= scale
                    module1.bias.data *= scale
        return copy_model

    def prob_start(self, scale, sorted_indices):
        # Determine start_index based on the error rate xi
        layer_num = len(sorted_indices)
        for layer_index in range(1, layer_num):
            layers = sorted_indices[:layer_index]
            smodel = self.scale_var_index(layers, scale=scale)
            smodel.cuda()
            smodel.eval()

            total_num = 0
            clean_wrong = 0
            with torch.no_grad():
                for idx, batch in enumerate(self.val_loader):
                    clean_img = batch[0]
                    labels = batch[1]
                    clean_img = clean_img.cuda()
                    clean_logits = smodel(clean_img).detach().cpu()
                    clean_pred = torch.argmax(clean_logits, dim=1)
                    clean_wrong += torch.sum(labels != clean_pred)
                    total_num += labels.shape[0]

                wrong_acc = clean_wrong / total_num
                print(f"[DEBUG] At layer_index {layer_index}, wrong_acc={wrong_acc}")
                if wrong_acc > self.xi:
                    print(f"[DEBUG] prob_start returning layer_index={layer_index}")
                    return layer_index

        # If no break condition met, return full length
        return layer_num

    def test(self, inspect_correct_predition_only=True):
        print(f'inspect_correct_predition_only: {inspect_correct_predition_only}')
        # This method calculates PSC scores and evaluates detection performance
        args = self.args
        print(f'start_index: {self.start_index}')

        total_num = 0
        y_score_clean = []
        y_score_poison = []

        # We will store how many samples we processed
        test_count = sum([batch[0].shape[0] for batch in self.test_loader])
        print(f"[DEBUG] Test set has {test_count} samples total.")

        with torch.no_grad():
            for idx, batch in enumerate(self.test_loader):
                clean_img = batch[0]
                labels = batch[1]
                total_num += labels.shape[0]
                clean_img = clean_img.cuda()
                labels = labels.cuda()

                # transform to poison
                poison_imgs, poison_labels = self.poison_transform.transform(clean_img, labels)

                # Compute predictions for poison and clean
                poison_pred = torch.argmax(self.model(poison_imgs), dim=1)
                clean_pred = torch.argmax(self.model(clean_img), dim=1)

                spc_poison = torch.zeros(labels.shape)
                spc_clean = torch.zeros(labels.shape)
                scale_count = 0

                # Scale up layers from start_index to start_index + n
                for layer_index in range(self.start_index, self.start_index + self.n):
                    layers = self.sorted_indices[:layer_index + 1]
                    smodel = self.scale_var_index(layers, scale=self.scale)
                    scale_count += 1
                    smodel.eval()

                    # For clean samples
                    logits_clean = smodel(clean_img).detach().cpu()
                    logits_clean = torch.nn.functional.softmax(logits_clean, dim=1)
                    device = logits_clean.device
                    clean_pred = clean_pred.to(device)
                    spc_clean += logits_clean[torch.arange(logits_clean.size(0), device=device), clean_pred]

                    # For poison samples
                    logits_poison = smodel(poison_imgs).detach().cpu()
                    logits_poison = torch.nn.functional.softmax(logits_poison, dim=1)
                    poison_pred = poison_pred.to(device)
                    spc_poison += logits_poison[torch.arange(logits_poison.size(0), device=device), poison_pred]

                spc_poison /= scale_count
                spc_clean /= scale_count

                # Append scores
                y_score_clean.append(spc_clean)
                y_score_poison.append(spc_poison)

            y_score_clean = torch.cat(y_score_clean, dim=0)
            y_score_poison = torch.cat(y_score_poison, dim=0)

        # Now we have y_score_clean and y_score_poison
        # Both have length = test_count (e.g., 8000 each)
        # Combine them: total length = 16000
        # This explains why tp+tn+fp+fn sum to 16000

        print(f"[DEBUG] y_score_clean length: {len(y_score_clean)}, y_score_poison length: {len(y_score_poison)}")
        print("[DEBUG] We combine both, so total length will be double the test set size.")

        # Construct labels for detection: 0 for clean, 1 for poison
        y_true = torch.cat((torch.zeros_like(y_score_clean), torch.ones_like(y_score_poison)))
        y_score = torch.cat((y_score_clean, y_score_poison), dim=0)
        y_pred = (y_score >= self.T)

        print(f"[DEBUG] After combination: total length: {len(y_true)}")
        print(f"[DEBUG] Clean samples: {len(y_score_clean)} (label=0), Poison samples: {len(y_score_poison)} (label=1)")

        # Compute metrics
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        auc = metrics.auc(fpr, tpr)
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()

        # Print confusion matrix details
        print("[DEBUG] Confusion Matrix Details:")
        print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print(f"TN+FP+FN+TP = {tn+fp+fn+tp} (Should be double of test set = {test_count*2})")

        myf1 = metrics.f1_score(y_true, y_pred)
        print("TPR: {:.2f}".format(tp / (tp + fn) * 100))
        print("FPR: {:.2f}".format(fp / (tn + fp) * 100))
        print("AUC: {:.4f}".format(auc))
        print(f"f1 score: {myf1:.4f}")

        if inspect_correct_predition_only:
            print(f'inspect_correct_predition_only: {inspect_correct_predition_only}')
            # Only consider correct-clean and successfully attacked-poison samples
            clean_pred_correct_mask = []
            poison_source_mask = []
            poison_attack_success_mask = []

            # This loop checks conditions again
            for batch_idx, batch in enumerate(tqdm(self.test_loader)):
                data = batch[0]
                label = batch[1]
                data, label = data.cuda(), label.cuda()

                clean_output = self.model(data)
                clean_pred = clean_output.argmax(dim=1)
                mask = torch.eq(clean_pred, label)
                clean_pred_correct_mask.append(mask)

                poison_data, poison_target = self.poison_transform.transform(data, label)

                if args.poison_type == 'TaCT':
                    mask1 = torch.eq(label, config.source_class)
                else:
                    # remove backdoor data whose original class == target class
                    mask1 = torch.not_equal(label, poison_target)
                poison_source_mask.append(mask1.clone())

                poison_output = self.model(poison_data)
                poison_pred = poison_output.argmax(dim=1)

                mask2 = torch.logical_and(torch.eq(poison_pred, poison_target), mask1)
                poison_attack_success_mask.append(mask2)

            clean_pred_correct_mask = torch.cat(clean_pred_correct_mask, dim=0)
            poison_source_mask = torch.cat(poison_source_mask, dim=0)
            poison_attack_success_mask = torch.cat(poison_attack_success_mask, dim=0)
            if args.poison_type == 'TaCT':
                clean_pred_correct_mask[torch.sum(poison_attack_success_mask).item():] = False

            mask = torch.cat((clean_pred_correct_mask, poison_attack_success_mask), dim=0)
            mask = mask.cpu()  # Đảm bảo mask ở CPU
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            y_score = y_score[mask]

            print('==========================partial testset results=========================')
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
            auc = metrics.auc(fpr, tpr)
            tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
            myf1 = metrics.f1_score(y_true, y_pred)
            print("[DEBUG] Partial set Confusion Matrix:")
            print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
            print("TPR: {:.2f}".format(tp / (tp + fn) * 100))
            print("FPR: {:.2f}".format(fp / (tn + fp) * 100))
            print("AUC: {:.4f}".format(auc))
            print(f"f1 score: {myf1:.4f}")

    def _detect(self, inputs):
        # Detect function to classify given inputs as malicious or not
        inputs = inputs.cuda()
        self.model.eval()
        original_pred = torch.argmax(self.model(inputs), dim=1)

        psc_score = torch.zeros(inputs.size(0))
        scale_count = 0
        for layer_index in range(self.start_index, self.start_index + self.n):
            layers = self.sorted_indices[:layer_index + 1]
            smodel = self.scale_var_index(layers, scale=self.scale)
            scale_count += 1
            smodel.eval()
            logits = smodel(inputs).detach().cpu()
            softmax_logits = torch.nn.functional.softmax(logits, dim=1)
            psc_score += softmax_logits[torch.arange(softmax_logits.size(0)), original_pred]

        psc_score /= scale_count
        y_pred = psc_score >= self.T
        return y_pred

    def detect(self):
        # Just run test and print predictions for the first batch
        with torch.no_grad():
            for idx, batch in enumerate(self.test_loader):
                imgs = batch[0]
                y_pred = self._detect(imgs)
                print(f'[DEBUG] inputs pred (first batch): {y_pred}')
                break

