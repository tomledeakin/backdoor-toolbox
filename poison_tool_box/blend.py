import os
import torch
import random
from torchvision.utils import save_image


class poison_generator():
    def __init__(self, img_size, dataset, poison_rate, trigger, path, target_class=0, alpha=0.2):
        self.img_size = img_size
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.trigger = trigger  # trigger được truyền vào (tensor)
        self.path = path  # path to save the dataset
        self.target_class = target_class  # by default: target_class = 0
        self.alpha = alpha

        # number of images
        self.num_img = len(dataset)

    def generate_poisoned_training_set(self):
        # random sampling
        id_set = list(range(self.num_img))
        random.shuffle(id_set)
        num_poison = int(self.num_img * self.poison_rate)
        poison_indices = sorted(id_set[:num_poison])


        img_set = []
        label_set = []
        pt = 0
        for i in range(self.num_img):
            img, gt = self.dataset[i]
            # In debug, in ra shape của ảnh trước khi áp dụng trigger
            # print(f"[DEBUG] Image {i} shape before: {img.shape}")

            if pt < num_poison and poison_indices[pt] == i:
                gt = self.target_class
                # Nếu số kênh của ảnh và trigger không khớp, điều chỉnh trigger về dạng grayscale
                if img.shape[0] != self.trigger.shape[0]:
                    adjusted_trigger = self.trigger.mean(dim=0, keepdim=True)

                else:
                    adjusted_trigger = self.trigger
                # Blend ảnh với trigger
                img = (1 - self.alpha) * img + self.alpha * adjusted_trigger
                pt += 1

            img_set.append(img.unsqueeze(0))
            label_set.append(gt)

        img_set = torch.cat(img_set, dim=0)
        label_set = torch.LongTensor(label_set)


        return img_set, poison_indices, label_set


class poison_transform():
    def __init__(self, img_size, trigger, target_class=0, alpha=0.2):
        self.img_size = img_size
        self.trigger = trigger  # trigger tensor
        self.target_class = target_class  # by default: target_class = 0
        self.alpha = alpha

    def transform(self, data, labels):
        data = data.clone()
        labels = labels.clone()

        # Nếu số kênh của data không khớp với trigger, điều chỉnh trigger về dạng grayscale
        if data.shape[1] != self.trigger.shape[0]:
            adjusted_trigger = self.trigger.mean(dim=0, keepdim=True)

        else:
            adjusted_trigger = self.trigger

        # Áp dụng blend: giữ lại phần (1-alpha)*data và trộn alpha*trigger
        data = (1 - self.alpha) * data + self.alpha * adjusted_trigger.to(data.device)
        labels[:] = self.target_class


        return data, labels
