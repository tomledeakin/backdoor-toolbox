import os
import torch
import random
from torchvision.utils import save_image

class poison_generator():
    def __init__(self, img_size, dataset, poison_rate, path, trigger_mark, trigger_mask, target_class=0, alpha=1.0):
        self.img_size = img_size
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.path = path  # path to save the dataset
        self.target_class = target_class  # by default: target_class = 0
        self.trigger_mark = trigger_mark
        self.trigger_mask = trigger_mask
        self.alpha = alpha

        # number of images
        self.num_img = len(dataset)

    def generate_poisoned_training_set(self):
        # random sampling
        id_set = list(range(self.num_img))
        random.shuffle(id_set)
        num_poison = int(self.num_img * self.poison_rate)
        poison_indices = sorted(id_set[:num_poison])
        print('poison_indices:', poison_indices)

        img_set = []
        label_set = []
        pt = 0
        for i in range(self.num_img):
            img, gt = self.dataset[i]

            # Kiểm tra nếu ảnh là grayscale (1 kênh), chuyển trigger về dạng grayscale nếu cần
            if img.shape[0] == 1:
                if self.trigger_mask.shape[0] != 1:
                    trigger_mask = self.trigger_mask.mean(dim=0, keepdim=True)
                else:
                    trigger_mask = self.trigger_mask
                if self.trigger_mark.shape[0] != 1:
                    trigger_mark = self.trigger_mark.mean(dim=0, keepdim=True)
                else:
                    trigger_mark = self.trigger_mark
            else:
                trigger_mask = self.trigger_mask
                trigger_mark = self.trigger_mark

            if pt < num_poison and poison_indices[pt] == i:
                gt = self.target_class
                # Áp dụng trigger: img = img + alpha * trigger_mask * (trigger_mark - img)
                img = img + self.alpha * trigger_mask * (trigger_mark - img)
                pt += 1

            img_set.append(img.unsqueeze(0))
            label_set.append(gt)

        img_set = torch.cat(img_set, dim=0)
        label_set = torch.LongTensor(label_set)

        return img_set, poison_indices, label_set


class poison_transform():
    def __init__(self, img_size, trigger_mark, trigger_mask, target_class=0, alpha=1.0):
        self.img_size = img_size
        self.target_class = target_class  # by default: target_class = 0
        self.trigger_mark = trigger_mark
        self.trigger_mask = trigger_mask
        self.alpha = alpha

    def transform(self, data, labels):
        data, labels = data.clone(), labels.clone()

        # Kiểm tra nếu ảnh là grayscale (1 kênh), chuyển trigger về dạng grayscale nếu cần
        if data.shape[1] == 1:  # data có shape (batch, channel, H, W)
            if self.trigger_mask.shape[0] != 1:
                trigger_mask = self.trigger_mask.mean(dim=0, keepdim=True)
            else:
                trigger_mask = self.trigger_mask
            if self.trigger_mark.shape[0] != 1:
                trigger_mark = self.trigger_mark.mean(dim=0, keepdim=True)
            else:
                trigger_mark = self.trigger_mark
        else:
            trigger_mask = self.trigger_mask
            trigger_mark = self.trigger_mark

        data = data + self.alpha * trigger_mask.to(data.device) * (trigger_mark.to(data.device) - data)
        labels[:] = self.target_class

        return data, labels
