import os
from math import sqrt
import random
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np
import config
from torchvision import transforms
from PIL import Image

"""Adaptive backdoor attack (with k triggers)
Just keep the original labels for some (say 50%) poisoned samples...
Poison with k triggers.
"""


def issquare(x):
    tmp = sqrt(x)
    tmp2 = round(tmp)
    return abs(tmp - tmp2) <= 1e-8


def get_trigger_mask(img_size, total_pieces, masked_pieces):
    div_num = sqrt(total_pieces)
    step = int(img_size // div_num)
    candidate_idx = random.sample(list(range(total_pieces)), k=masked_pieces)
    mask = torch.ones((img_size, img_size))
    # Lưu ý: dùng [y, x] khi index theo chiều cao, chiều rộng.
    for i in candidate_idx:
        x = int(i % div_num)  # cột
        y = int(i // div_num)  # hàng
        mask[y * step: (y + 1) * step, x * step: (x + 1) * step] = 0
    return mask


class poison_generator():
    def __init__(self, img_size, dataset, poison_rate, cover_rate, path, trigger_names, alphas, target_class=0):
        self.img_size = img_size
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.cover_rate = cover_rate
        self.path = path  # path to save the dataset
        self.target_class = target_class  # by default: target_class = 0

        # number of images
        self.num_img = len(dataset)

        # triggers: load trigger marks and masks, resize nếu cần
        trigger_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.trigger_marks = []
        self.trigger_masks = []
        self.alphas = []
        for i in range(len(trigger_names)):
            trigger_path = os.path.join(config.triggers_dir, trigger_names[i])
            trigger_mask_path = os.path.join(config.triggers_dir, 'mask_%s' % trigger_names[i])

            # Load trigger and convert to tensor (mặc định load dưới dạng RGB)
            trigger = Image.open(trigger_path).convert("RGB")
            trigger = trigger_transform(trigger)
            # Resize trigger nếu kích thước không bằng img_size
            if trigger.shape[-2:] != (self.img_size, self.img_size):
                trigger = F.interpolate(trigger.unsqueeze(0), size=(self.img_size, self.img_size),
                                        mode='bilinear', align_corners=False)[0]
            # [ADDED FOR 1-CHANNEL]:
            # Nếu ảnh của dataset là grayscale (1 kênh) nhưng trigger có nhiều kênh, chuyển trigger thành grayscale bằng cách lấy mean
            sample_img, _ = dataset[0]
            if sample_img.shape[0] == 1 and trigger.shape[0] != 1:
                trigger = trigger.mean(dim=0, keepdim=True)

            # Trigger mask: nếu có file mask, dùng nó, nếu không thì tự tạo
            if os.path.exists(trigger_mask_path):
                trigger_mask = Image.open(trigger_mask_path).convert("RGB")
                trigger_mask = transforms.ToTensor()(trigger_mask)[0]  # chỉ lấy 1 channel
            else:
                trigger_mask = torch.logical_or(torch.logical_or(trigger[0] > 0, trigger[1] > 0),
                                                trigger[2] > 0).float()
            # Resize mask: note mask ban đầu có shape (H, W); cần resize về (img_size, img_size)
            if trigger_mask.shape != (self.img_size, self.img_size):
                trigger_mask = F.interpolate(trigger_mask.unsqueeze(0).unsqueeze(0),
                                             size=(self.img_size, self.img_size),
                                             mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            self.trigger_marks.append(trigger)
            self.trigger_masks.append(trigger_mask)
            self.alphas.append(alphas[i])

    def generate_poisoned_training_set(self):
        # random sampling
        id_set = list(range(self.num_img))
        random.shuffle(id_set)
        num_poison = int(self.num_img * self.poison_rate)
        poison_indices = sorted(id_set[:num_poison])

        num_cover = int(self.num_img * self.cover_rate)
        cover_indices = sorted(id_set[num_poison:num_poison + num_cover])  # non-overlapping cover images

        img_set = []
        label_set = []
        pt = 0
        ct = 0
        cnt = 0

        poison_id = []
        cover_id = []
        k = len(self.trigger_marks)

        for i in range(self.num_img):
            img, gt = self.dataset[i]
            # Resize img nếu cần (đảm bảo có shape (C, img_size, img_size))
            if img.shape[-2:] != (self.img_size, self.img_size):
                img = F.interpolate(img.unsqueeze(0), size=(self.img_size, self.img_size),
                                    mode='bilinear', align_corners=True)[0]
            # [ADDED FOR 1-CHANNEL]:
            # Nếu dataset là grayscale (1 kênh) nhưng ảnh có >1 kênh (do transform nào đó), chuyển về 1 kênh.
            if self.dataset[0][0].shape[0] == 1 and img.shape[0] != 1:
                img = img.mean(dim=0, keepdim=True)

            # cover image: áp dụng trigger theo phân chia k triggers
            if ct < num_cover and cover_indices[ct] == i:
                cover_id.append(cnt)
                for j in range(k):
                    if ct < (j + 1) * (num_cover / k):
                        # Điều chỉnh trigger theo kênh nếu cần
                        if img.shape[0] == 1 and self.trigger_marks[j].shape[0] != 1:
                            t_mark = self.trigger_marks[j].mean(dim=0, keepdim=True)
                        else:
                            t_mark = self.trigger_marks[j]
                        img = img + self.alphas[j] * self.trigger_masks[j].to(img.device) * (
                                    t_mark.to(img.device) - img)
                        break
                ct += 1

            # poisoned image: đổi nhãn và thêm trigger
            if pt < num_poison and poison_indices[pt] == i:
                poison_id.append(cnt)
                gt = self.target_class  # đổi nhãn thành target_class
                for j in range(k):
                    if pt < (j + 1) * (num_poison / k):
                        if img.shape[0] == 1 and self.trigger_marks[j].shape[0] != 1:
                            t_mark = self.trigger_marks[j].mean(dim=0, keepdim=True)
                        else:
                            t_mark = self.trigger_marks[j]
                        img = img + self.alphas[j] * self.trigger_masks[j].to(img.device) * (
                                    t_mark.to(img.device) - img)
                        break
                pt += 1

            img_set.append(img.unsqueeze(0))
            label_set.append(gt)
            cnt += 1

        img_set = torch.cat(img_set, dim=0)
        label_set = torch.LongTensor(label_set)
        poison_indices = poison_id
        cover_indices = cover_id

        # Demo: xử lý ảnh đầu tiên với tất cả các trigger
        img, gt = self.dataset[0]
        if img.shape[-2:] != (self.img_size, self.img_size):
            img = F.interpolate(img.unsqueeze(0), size=(self.img_size, self.img_size),
                                mode='bilinear', align_corners=True)[0]
        for j in range(k):
            if img.shape[0] == 1 and self.trigger_marks[j].shape[0] != 1:
                t_mark = self.trigger_marks[j].mean(dim=0, keepdim=True)
            else:
                t_mark = self.trigger_marks[j]
            img = img + self.alphas[j] * self.trigger_masks[j].to(img.device) * (t_mark.to(img.device) - img)
        save_image(img, os.path.join(self.path, 'demo.png'))

        return img_set, poison_indices, cover_indices, label_set


class poison_transform():
    def __init__(self, img_size, test_trigger_names, test_alphas, target_class=0, denormalizer=None, normalizer=None):
        self.img_size = img_size
        self.target_class = target_class
        self.denormalizer = denormalizer
        self.normalizer = normalizer

        # triggers: load và resize về kích thước (img_size, img_size)
        trigger_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.trigger_marks = []
        self.trigger_masks = []
        self.alphas = []
        for i in range(len(test_trigger_names)):
            trigger_path = os.path.join(config.triggers_dir, test_trigger_names[i])
            trigger_mask_path = os.path.join(config.triggers_dir, 'mask_%s' % test_trigger_names[i])
            trigger = Image.open(trigger_path).convert("RGB")
            trigger = trigger_transform(trigger)
            if trigger.shape[-2:] != (self.img_size, self.img_size):
                trigger = F.interpolate(trigger.unsqueeze(0), size=(self.img_size, self.img_size),
                                        mode='bilinear', align_corners=False)[0]
            # [ADDED FOR 1-CHANNEL]:
            if self.normalizer is not None:
                # Giả sử nếu normalizer có mean là tuple với 1 phần tử thì dữ liệu là grayscale
                if isinstance(self.normalizer.transforms[0].mean, (tuple, list)) and len(
                        self.normalizer.transforms[0].mean) == 1:
                    trigger = trigger.mean(dim=0, keepdim=True)
            if os.path.exists(trigger_mask_path):
                trigger_mask = Image.open(trigger_mask_path).convert("RGB")
                trigger_mask = transforms.ToTensor()(trigger_mask)[0]
            else:
                trigger_mask = torch.logical_or(torch.logical_or(trigger[0] > 0, trigger[1] > 0),
                                                trigger[2] > 0).float()
            if trigger_mask.shape != (self.img_size, self.img_size):
                trigger_mask = F.interpolate(trigger_mask.unsqueeze(0).unsqueeze(0),
                                             size=(self.img_size, self.img_size),
                                             mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            self.trigger_marks.append(trigger.cuda())
            self.trigger_masks.append(trigger_mask.cuda())
            self.alphas.append(test_alphas[i])

    def transform(self, data, labels, denormalizer=None, normalizer=None):
        data, labels = data.clone(), labels.clone()
        # Nếu dữ liệu không đúng kích thước, resize về (img_size, img_size)
        if data.shape[-2:] != (self.img_size, self.img_size):
            data = F.interpolate(data, size=(self.img_size, self.img_size),
                                 mode='bilinear', align_corners=False)
        # [ADDED FOR 1-CHANNEL]:
        # Nếu data là ảnh 1 kênh nhưng trigger có nhiều kênh, điều chỉnh trigger
        for j in range(len(self.trigger_marks)):
            if data.shape[1] == 1 and self.trigger_marks[j].shape[0] != 1:
                adjusted_trigger = self.trigger_marks[j].mean(dim=0, keepdim=True)
            else:
                adjusted_trigger = self.trigger_marks[j]
            data = data + self.alphas[j] * self.trigger_masks[j].to(data.device) * (
                        adjusted_trigger.to(data.device) - data)
        data = self.normalizer(data)
        labels[:] = self.target_class
        return data, labels

# import os
# import torch
# from torch import nn
# import torch.nn.functional as F
# import random
# from torchvision.utils import save_image
# from config import poison_seed
#
# """
# WaNet (static poisoning). https://github.com/VinAIResearch/Warping-based_Backdoor_Attack-release
# """
#
#
# class poison_generator():
#
#     def __init__(self, img_size, dataset, poison_rate, cover_rate, path, identity_grid, noise_grid, s=0.5, k=4,
#                  grid_rescale=1, target_class=0):
#
#         self.img_size = img_size
#         self.dataset = dataset
#         self.poison_rate = poison_rate
#         self.cover_rate = cover_rate
#         self.path = path  # path to save the dataset
#         self.target_class = target_class  # by default : target_class = 0
#
#         # number of images
#         self.num_img = len(dataset)
#
#         self.s = s
#         self.k = k
#         self.grid_rescale = grid_rescale
#         self.identity_grid = identity_grid
#         self.noise_grid = noise_grid
#
#     def generate_poisoned_training_set(self):
#         torch.manual_seed(poison_seed)
#         random.seed(poison_seed)
#
#         # random sampling
#         id_set = list(range(0, self.num_img))
#         random.shuffle(id_set)
#         num_poison = int(self.num_img * self.poison_rate)
#         poison_indices = id_set[:num_poison]
#         poison_indices.sort()  # increasing order
#
#         num_cover = int(self.num_img * self.cover_rate)
#         cover_indices = id_set[num_poison:num_poison + num_cover]  # use **non-overlapping** images to cover
#         cover_indices.sort()
#
#         img_set = []
#         label_set = []
#         pt = 0
#         ct = 0
#         cnt = 0
#
#         poison_id = []
#         cover_id = []
#
#         grid_temps = (self.identity_grid + self.s * self.noise_grid / self.img_size) * self.grid_rescale
#         grid_temps = torch.clamp(grid_temps, -1, 1)
#
#         ins = torch.rand(1, self.img_size, self.img_size, 2) * 2 - 1
#         grid_temps2 = grid_temps + ins / self.img_size
#         grid_temps2 = torch.clamp(grid_temps2, -1, 1)
#
#         for i in range(self.num_img):
#             img, gt = self.dataset[i]
#
#             # Nếu kích thước ảnh không khớp với img_size, resize về kích thước chuẩn.
#             if img.shape[-2:] != (self.img_size, self.img_size):
#                 img = F.interpolate(img.unsqueeze(0), size=(self.img_size, self.img_size), mode='bilinear',
#                                     align_corners=True)[0]
#
#             # noise image
#             if ct < num_cover and cover_indices[ct] == i:
#                 cover_id.append(cnt)
#                 img = F.grid_sample(img.unsqueeze(0), grid_temps2, align_corners=True)[0]
#                 ct += 1
#
#             # poisoned image
#             if pt < num_poison and poison_indices[pt] == i:
#                 poison_id.append(cnt)
#                 gt = self.target_class  # change the label to the target class
#                 img = F.grid_sample(img.unsqueeze(0), grid_temps, align_corners=True)[0]
#                 pt += 1
#
#             # Lưu lại ảnh sau biến đổi
#             img_set.append(img.unsqueeze(0))
#             label_set.append(gt)
#             cnt += 1
#
#         img_set = torch.cat(img_set, dim=0)
#         label_set = torch.LongTensor(label_set)
#         poison_indices = poison_id
#         cover_indices = cover_id
#         print("Poison indices:", poison_indices)
#         print("Cover indices:", cover_indices)
#
#         # Demo: biến đổi ảnh đầu tiên và lưu lại
#         img, gt = self.dataset[0]
#         if img.shape[-2:] != (self.img_size, self.img_size):
#             img = \
#             F.interpolate(img.unsqueeze(0), size=(self.img_size, self.img_size), mode='bilinear', align_corners=True)[0]
#         img = F.grid_sample(img.unsqueeze(0), grid_temps, align_corners=True)[0]
#         save_image(img, os.path.join(self.path, 'demo.png'))
#
#         return img_set, poison_indices, cover_indices, label_set
#
#
# class poison_transform():
#
#     def __init__(self, img_size, normalizer, denormalizer, identity_grid, noise_grid, s=0.5, k=4, grid_rescale=1,
#                  target_class=0):
#         self.img_size = img_size
#         self.normalizer = normalizer
#         self.denormalizer = denormalizer
#         self.target_class = target_class
#
#         self.s = s
#         self.k = k
#         self.grid_rescale = grid_rescale
#         self.identity_grid = identity_grid.cuda()
#         self.noise_grid = noise_grid.cuda()
#
#     def transform(self, data, labels):
#         grid_temps = (self.identity_grid.to(data.device) + self.s * self.noise_grid.to(
#             data.device) / self.img_size) * self.grid_rescale
#         grid_temps = torch.clamp(grid_temps, -1, 1)
#
#         data, labels = data.clone(), labels.clone()
#         data = self.denormalizer(data)
#         data = F.grid_sample(data, grid_temps.repeat(data.shape[0], 1, 1, 1), align_corners=True)
#         data = self.normalizer(data)
#         labels[:] = self.target_class
#
#         return data, labels
