import os
from sklearn import config_context
import torch
import random
from torchvision.utils import save_image
import numpy as np
import config
from torchvision import transforms
from math import sqrt

"""Adaptive Mask backdoor attack
- Keep the original labels for some (say 50%) poisoned samples.
- Divide the blending trigger into multiple pieces, randomly masking some pieces while poisoning the trainset.
This version uses blending backdoor trigger: blending a mark with a mask and a transparency `alpha`
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
    for i in candidate_idx:
        x = int(i % div_num)  # column
        y = int(i // div_num)  # row
        mask[x * step: (x + 1) * step, y * step: (y + 1) * step] = 0
    return mask


class poison_generator():

    def __init__(self, img_size, dataset, poison_rate, path, trigger, target_class=0, alpha=0.2, cover_rate=0.01,
                 pieces=16, mask_rate=0.5):

        self.img_size = img_size
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.path = path  # path to save the dataset
        self.target_class = target_class  # by default : target_class = 0
        self.trigger = trigger
        self.alpha = alpha
        self.cover_rate = cover_rate
        assert abs(round(sqrt(pieces)) - sqrt(pieces)) <= 1e-8
        assert img_size % round(sqrt(pieces)) == 0
        self.pieces = pieces
        self.mask_rate = mask_rate
        self.masked_pieces = round(self.mask_rate * self.pieces)

        # number of images
        self.num_img = len(dataset)

    def generate_poisoned_training_set(self):

        # random sampling
        id_set = list(range(0, self.num_img))
        random.shuffle(id_set)
        num_poison = int(self.num_img * self.poison_rate)
        poison_indices = id_set[:num_poison]
        poison_indices.sort()  # increasing order

        num_cover = int(self.num_img * self.cover_rate)
        cover_indices = id_set[num_poison:num_poison + num_cover]  # use **non-overlapping** images to cover
        cover_indices.sort()

        label_set = []
        pt = 0
        ct = 0
        cnt = 0

        img_set = []
        poison_id = []
        cover_id = []

        for i in range(self.num_img):
            img, gt = self.dataset[i]

            # cover image
            if ct < num_cover and cover_indices[ct] == i:
                cover_id.append(cnt)
                mask = get_trigger_mask(self.img_size, self.pieces, self.masked_pieces)
                img = img + self.alpha * mask * (self.trigger - img)
                ct += 1

            # poisoned image
            if pt < num_poison and poison_indices[pt] == i:
                poison_id.append(cnt)
                gt = self.target_class  # change the label to the target class
                mask = get_trigger_mask(self.img_size, self.pieces, self.masked_pieces)
                img = img + self.alpha * mask * (self.trigger - img)
                pt += 1

            # img_file_name = '%d.png' % cnt
            # img_file_path = os.path.join(self.path, img_file_name)
            # save_image(img, img_file_path)
            # print('[Generate Poisoned Set] Save %s' % img_file_path)

            img_set.append(img.unsqueeze(0))
            label_set.append(gt)
            cnt += 1

        img_set = torch.cat(img_set, dim=0)
        label_set = torch.LongTensor(label_set)
        poison_indices = poison_id
        cover_indices = cover_id
        print("Poison indices:", poison_indices)
        print("Cover indices:", cover_indices)

        # demo
        img, gt = self.dataset[0]
        mask = get_trigger_mask(self.img_size, self.pieces, self.masked_pieces)
        img = img + self.alpha * mask * (self.trigger - img)
        save_image(img, os.path.join(self.path, 'demo.png'))

        return img_set, poison_indices, cover_indices, label_set


class poison_transform():

    def __init__(self, img_size, trigger, target_class=0, alpha=0.2):
        self.img_size = img_size
        self.target_class = target_class
        self.trigger = trigger
        self.alpha = alpha

    def transform(self, data, labels):
        data, labels = data.clone(), labels.clone()
        data = data + self.alpha * (self.trigger.to(data.device) - data)
        labels[:] = self.target_class

        # debug
        # from torchvision.utils import save_image
        # from torchvision import transforms
        # normalizer = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        # denormalizer = transforms.Normalize([-0.4914/0.247, -0.4822/0.243, -0.4465/0.261], [1/0.247, 1/0.243, 1/0.261])
        # # normalizer = transforms.Compose([
        # #     transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
        # # ])
        # # denormalizer = transforms.Compose([
        # #     transforms.Normalize((-0.3337 / 0.2672, -0.3064 / 0.2564, -0.3171 / 0.2629),
        # #                             (1.0 / 0.2672, 1.0 / 0.2564, 1.0 / 0.2629)),
        # # ])
        # save_image(denormalizer(data)[0], 'b.png')

        return data, labels


# import os
# import torch
# import random
# from torchvision.utils import save_image
# import numpy as np
# import config
# from torchvision import transforms
# from math import sqrt
# import torch.nn.functional as F
#
#
# def issquare(x):
#     tmp = sqrt(x)
#     tmp2 = round(tmp)
#     return abs(tmp - tmp2) <= 1e-8
#
#
# def get_trigger_mask(img_size, total_pieces, masked_pieces):
#     div_num = sqrt(total_pieces)
#     step = int(img_size // div_num)
#     candidate_idx = random.sample(list(range(total_pieces)), k=masked_pieces)
#     mask = torch.ones((img_size, img_size))
#     # Lưu ý: dùng [y, x] khi index theo chiều cao, chiều rộng.
#     for i in candidate_idx:
#         x = int(i % div_num)  # cột
#         y = int(i // div_num)  # hàng
#         mask[y * step: (y + 1) * step, x * step: (x + 1) * step] = 0
#     return mask
#
#
# class poison_generator():
#     def __init__(self, img_size, dataset, poison_rate, path, trigger, target_class=0, alpha=0.2, cover_rate=0.01,
#                  pieces=16, mask_rate=0.5):
#         # Lấy kích thước ảnh từ dataset
#         sample_img, _ = dataset[0]
#         actual_size = sample_img.shape[1]  # giả sử ảnh vuông (H == W)
#         if img_size != actual_size:
#             img_size = actual_size
#         self.img_size = img_size
#
#         self.dataset = dataset
#         self.poison_rate = poison_rate
#         self.path = path  # đường dẫn lưu dataset
#         self.target_class = target_class  # mặc định: target_class = 0
#         self.trigger = trigger
#         # Resize trigger nếu cần
#         if self.trigger.shape[-1] != self.img_size:
#             self.trigger = F.interpolate(self.trigger.unsqueeze(0), size=(self.img_size, self.img_size),
#                                          mode='bilinear', align_corners=False).squeeze(0)
#         # *** Sửa thêm: Nếu dataset là ảnh 1 kênh (grayscale) mà trigger có nhiều kênh, chuyển trigger về 1 kênh
#         if sample_img.shape[0] == 1 and self.trigger.shape[0] != 1:
#             self.trigger = self.trigger.mean(dim=0, keepdim=True)
#
#         self.alpha = alpha
#         self.cover_rate = cover_rate
#         assert abs(round(sqrt(pieces)) - sqrt(pieces)) <= 1e-8, "Pieces phải là số chính phương"
#         assert self.img_size % round(sqrt(pieces)) == 0, "img_size phải chia hết cho sqrt(pieces)"
#         self.pieces = pieces
#         self.mask_rate = mask_rate
#         self.masked_pieces = round(self.mask_rate * self.pieces)
#
#         # Số lượng ảnh trong dataset
#         self.num_img = len(dataset)
#
#     def generate_poisoned_training_set(self):
#         # Trộn ngẫu nhiên chỉ số ảnh
#         id_set = list(range(self.num_img))
#         random.shuffle(id_set)
#         num_poison = int(self.num_img * self.poison_rate)
#         poison_indices = sorted(id_set[:num_poison])
#
#         num_cover = int(self.num_img * self.cover_rate)
#         cover_indices = sorted(id_set[num_poison:num_poison + num_cover])  # dùng các ảnh không giao nhau cho cover
#
#         label_set = []
#         pt = 0
#         ct = 0
#         cnt = 0
#
#         img_set = []
#         poison_id = []
#         cover_id = []
#
#         for i in range(self.num_img):
#             img, gt = self.dataset[i]
#             # Resize ảnh nếu cần để khớp với self.img_size
#             if img.shape[1] != self.img_size:
#                 img = F.interpolate(img.unsqueeze(0), size=(self.img_size, self.img_size),
#                                     mode='bilinear', align_corners=False).squeeze(0)
#
#             # Nếu ảnh là grayscale nhưng trigger có nhiều kênh, điều chỉnh trigger cho branch này:
#             if img.shape[0] == 1 and self.trigger.shape[0] != 1:
#                 adjusted_trigger = self.trigger.mean(dim=0, keepdim=True)
#             else:
#                 adjusted_trigger = self.trigger
#
#             # Ảnh cover
#             if ct < num_cover and cover_indices[ct] == i:
#                 cover_id.append(cnt)
#                 mask = get_trigger_mask(self.img_size, self.pieces, self.masked_pieces)
#                 img = img + self.alpha * mask.unsqueeze(0) * (adjusted_trigger - img)
#                 ct += 1
#
#             # Ảnh bị nhiễm
#             if pt < num_poison and poison_indices[pt] == i:
#                 poison_id.append(cnt)
#                 gt = self.target_class  # đổi nhãn về target_class
#                 mask = get_trigger_mask(self.img_size, self.pieces, self.masked_pieces)
#                 img = img + self.alpha * mask.unsqueeze(0) * (adjusted_trigger - img)
#                 pt += 1
#
#             img_set.append(img.unsqueeze(0))
#             label_set.append(gt)
#             cnt += 1
#
#         img_set = torch.cat(img_set, dim=0)
#         label_set = torch.LongTensor(label_set)
#         poison_indices = poison_id
#         cover_indices = cover_id
#
#         # Tạo ảnh demo
#         img, gt = self.dataset[0]
#         if img.shape[1] != self.img_size:
#             img = F.interpolate(img.unsqueeze(0), size=(self.img_size, self.img_size),
#                                 mode='bilinear', align_corners=False).squeeze(0)
#         mask = get_trigger_mask(self.img_size, self.pieces, self.masked_pieces)
#         if img.shape[0] == 1 and self.trigger.shape[0] != 1:
#             demo_trigger = self.trigger.mean(dim=0, keepdim=True)
#         else:
#             demo_trigger = self.trigger
#         img = img + self.alpha * mask.unsqueeze(0) * (demo_trigger - img)
#         save_image(img, os.path.join(self.path, 'demo.png'))
#
#         return img_set, poison_indices, cover_indices, label_set
#
#
# class poison_transform():
#     def __init__(self, img_size, trigger, target_class=0, alpha=0.2):
#         self.img_size = img_size
#         self.target_class = target_class
#         self.trigger = trigger
#         # Resize trigger nếu cần
#         if self.trigger.shape[-1] != self.img_size:
#             self.trigger = F.interpolate(self.trigger.unsqueeze(0), size=(self.img_size, self.img_size),
#                                          mode='bilinear', align_corners=False).squeeze(0)
#         # *** Sửa thêm: Điều chỉnh trigger cho ảnh 1 kênh
#         if self.trigger.shape[0] != 1:
#             # Nếu dữ liệu cần là 1 kênh (ví dụ MNIST) và trigger có nhiều kênh, chuyển về 1 kênh
#             self.trigger = self.trigger.mean(dim=0, keepdim=True)
#         self.alpha = alpha
#
#     def transform(self, data, labels):
#         data = data.clone()
#         labels = labels.clone()
#         if data.shape[-1] != self.img_size:
#             data = F.interpolate(data, size=(self.img_size, self.img_size),
#                                  mode='bilinear', align_corners=False)
#         # *** Sửa thêm: Kiểm tra nếu data là 1 kênh nhưng trigger không khớp, điều chỉnh lại trigger
#         if data.shape[1] != self.trigger.shape[0]:
#             adjusted_trigger = self.trigger.mean(dim=0, keepdim=True)
#         else:
#             adjusted_trigger = self.trigger
#
#         data = data + self.alpha * (adjusted_trigger.to(data.device) - data)
#         labels[:] = self.target_class
#         return data, labels

# import os
# import torch
# import random
# from torchvision.utils import save_image
# import numpy as np
# import config
# from torchvision import transforms
# from math import sqrt
# import torch.nn.functional as F
#
# """Adaptive Mask backdoor attack
# - Giữ nhãn gốc cho một số (ví dụ 50%) mẫu bị nhiễm.
# - Chia trigger (dấu hiệu) backdoor thành nhiều mảnh, ẩn một số mảnh một cách ngẫu nhiên trong quá trình nhiễm dữ liệu huấn luyện.
# Phiên bản này sử dụng blending backdoor trigger: trộn một dấu hiệu với mask và độ trong suốt `alpha`
# """
#
#
# def issquare(x):
#     tmp = sqrt(x)
#     tmp2 = round(tmp)
#     return abs(tmp - tmp2) <= 1e-8
#
#
# def get_trigger_mask(img_size, total_pieces, masked_pieces):
#     div_num = sqrt(total_pieces)
#     step = int(img_size // div_num)
#     candidate_idx = random.sample(list(range(total_pieces)), k=masked_pieces)
#     mask = torch.ones((img_size, img_size))
#     # Lưu ý: dùng [y, x] khi index theo chiều cao, chiều rộng.
#     for i in candidate_idx:
#         x = int(i % div_num)  # cột
#         y = int(i // div_num)  # hàng
#         mask[y * step: (y + 1) * step, x * step: (x + 1) * step] = 0
#     return mask
#
#
# class poison_generator():
#     def __init__(self, img_size, dataset, poison_rate, path, trigger, target_class=0, alpha=0.2, cover_rate=0.01,
#                  pieces=16, mask_rate=0.5):
#         # Lấy kích thước ảnh từ dataset
#         sample_img, _ = dataset[0]
#         actual_size = sample_img.shape[1]  # giả sử ảnh vuông (H == W)
#         if img_size != actual_size:
#             print(
#                 f"Warning: Provided img_size {img_size} does not match actual image size {actual_size}. Using {actual_size}.")
#             img_size = actual_size
#         self.img_size = img_size
#
#         self.dataset = dataset
#         self.poison_rate = poison_rate
#         self.path = path  # đường dẫn lưu dataset
#         self.target_class = target_class  # mặc định: target_class = 0
#         self.trigger = trigger
#         # Resize trigger nếu cần
#         if self.trigger.shape[-1] != self.img_size:
#             self.trigger = F.interpolate(self.trigger.unsqueeze(0), size=(self.img_size, self.img_size),
#                                          mode='bilinear', align_corners=False).squeeze(0)
#         self.alpha = alpha
#         self.cover_rate = cover_rate
#         assert abs(round(sqrt(pieces)) - sqrt(pieces)) <= 1e-8, "Pieces phải là số chính phương"
#         assert self.img_size % round(sqrt(pieces)) == 0, "img_size phải chia hết cho sqrt(pieces)"
#         self.pieces = pieces
#         self.mask_rate = mask_rate
#         self.masked_pieces = round(self.mask_rate * self.pieces)
#
#         # Số lượng ảnh trong dataset
#         self.num_img = len(dataset)
#
#     def generate_poisoned_training_set(self):
#         # Trộn ngẫu nhiên chỉ số ảnh
#         id_set = list(range(self.num_img))
#         random.shuffle(id_set)
#         num_poison = int(self.num_img * self.poison_rate)
#         poison_indices = id_set[:num_poison]
#         poison_indices.sort()  # sắp xếp tăng dần
#
#         num_cover = int(self.num_img * self.cover_rate)
#         cover_indices = id_set[num_poison:num_poison + num_cover]  # dùng các ảnh không giao nhau cho cover
#         cover_indices.sort()
#
#         label_set = []
#         pt = 0
#         ct = 0
#         cnt = 0
#
#         img_set = []
#         poison_id = []
#         cover_id = []
#
#         for i in range(self.num_img):
#             img, gt = self.dataset[i]
#             # Resize ảnh nếu cần để khớp với self.img_size
#             if img.shape[1] != self.img_size:
#                 img = F.interpolate(img.unsqueeze(0), size=(self.img_size, self.img_size),
#                                     mode='bilinear', align_corners=False).squeeze(0)
#
#             # Ảnh cover
#             if ct < num_cover and cover_indices[ct] == i:
#                 cover_id.append(cnt)
#                 mask = get_trigger_mask(self.img_size, self.pieces, self.masked_pieces)
#                 print(f"img shape: {img.shape}")  # (C, H, W)
#                 print(f"trigger shape: {self.trigger.shape}")  # (C, H, W)
#                 print(f"mask shape: {mask.shape}")  # (H, W)
#                 print(f"alpha: {self.alpha}")
#                 print(f"(trigger - img) shape: {(self.trigger - img).shape}")
#                 print(f"mask * (trigger - img) shape: {(mask.unsqueeze(0) * (self.trigger - img)).shape}")
#
#                 img = img + self.alpha * mask.unsqueeze(0) * (self.trigger - img)
#                 ct += 1
#
#             # Ảnh bị nhiễm
#             if pt < num_poison and poison_indices[pt] == i:
#                 poison_id.append(cnt)
#                 gt = self.target_class  # đổi nhãn về target_class
#                 mask = get_trigger_mask(self.img_size, self.pieces, self.masked_pieces)
#                 print(f"img shape: {img.shape}")
#                 print(f"trigger shape: {self.trigger.shape}")
#                 print(f"mask shape: {mask.shape}")
#                 print(f"alpha: {self.alpha}")
#                 print(f"(trigger - img) shape: {(self.trigger - img).shape}")
#                 print(f"mask * (trigger - img) shape: {(mask.unsqueeze(0) * (self.trigger - img)).shape}")
#
#                 img = img + self.alpha * mask.unsqueeze(0) * (self.trigger - img)
#                 pt += 1
#
#             # Nếu cần lưu ảnh riêng lẻ, mở comment dòng dưới đây:
#             # img_file_name = '%d.png' % cnt
#             # img_file_path = os.path.join(self.path, img_file_name)
#             # save_image(img, img_file_path)
#             # print('[Generate Poisoned Set] Save %s' % img_file_path)
#
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
#         # Tạo ảnh demo
#         img, gt = self.dataset[0]
#         if img.shape[1] != self.img_size:
#             img = F.interpolate(img.unsqueeze(0), size=(self.img_size, self.img_size),
#                                 mode='bilinear', align_corners=False).squeeze(0)
#         mask = get_trigger_mask(self.img_size, self.pieces, self.masked_pieces)
#         img = img + self.alpha * mask.unsqueeze(0) * (self.trigger - img)
#         save_image(img, os.path.join(self.path, 'demo.png'))
#
#         return img_set, poison_indices, cover_indices, label_set
#
#
# class poison_transform():
#     def __init__(self, img_size, trigger, target_class=0, alpha=0.2):
#         self.img_size = img_size
#         self.target_class = target_class
#         self.trigger = trigger
#         # Resize trigger nếu cần
#         if self.trigger.shape[-1] != self.img_size:
#             self.trigger = F.interpolate(self.trigger.unsqueeze(0), size=(self.img_size, self.img_size),
#                                          mode='bilinear', align_corners=False).squeeze(0)
#         self.alpha = alpha
#
#     def transform(self, data, labels):
#         data, labels = data.clone(), labels.clone()
#         if data.shape[-1] != self.img_size:
#             data = F.interpolate(data, size=(self.img_size, self.img_size),
#                                  mode='bilinear', align_corners=False)
#         data = data + self.alpha * (self.trigger.to(data.device) - data)
#         labels[:] = self.target_class
#         return data, labels
