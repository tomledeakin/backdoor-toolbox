# import os
# import torch
# import random
# from torchvision.utils import save_image
#
# class poison_generator():
#     def __init__(self, img_size, dataset, poison_rate, cover_rate, trigger, mask, path, target_class=0,
#                  source_class=1, cover_classes=[5,7]):
#         self.img_size = img_size
#         self.dataset = dataset
#         self.poison_rate = poison_rate
#         self.cover_rate = cover_rate
#         self.trigger = trigger
#         self.mask = mask
#         self.path = path  # path to save the dataset
#         self.target_class = target_class  # by default : target_class = 0
#         self.source_class = source_class  # by default : source_class = 1
#         self.cover_classes = cover_classes
#
#         # number of images
#         self.num_img = len(dataset)
#
#         # shape of the patch trigger (mask)
#         self.dx, self.dy = mask.shape
#
#         # [ADDED FOR 1-CHANNEL]:
#         # Nếu dataset là ảnh một kênh (ví dụ MNIST), chuyển trigger và mask về dạng 1 kênh
#         sample_img, _ = dataset[0]
#         if sample_img.shape[0] == 1:
#             if self.trigger.shape[0] != 1:
#                 self.trigger = self.trigger.mean(dim=0, keepdim=True)
#             if self.mask.shape[0] != 1:
#                 # Có thể chọn cách dùng channel 0 hoặc trung bình các kênh
#                 self.mask = self.mask.mean(dim=0, keepdim=True)
#
#     def generate_poisoned_training_set(self):
#         # random sampling
#         all_source_indices = []
#         all_cover_indices = []
#         for i in range(self.num_img):
#             _, gt = self.dataset[i]
#             if gt == self.source_class:
#                 all_source_indices.append(i)
#             elif gt in self.cover_classes:
#                 all_cover_indices.append(i)
#         random.shuffle(all_source_indices)
#         random.shuffle(all_cover_indices)
#
#         num_poison = int(self.num_img * self.poison_rate)
#         num_cover = int(self.num_img * self.cover_rate)
#
#         poison_indices = sorted(all_source_indices[:num_poison])
#         cover_indices = sorted(all_cover_indices[:num_cover])
#
#         img_set = []
#         label_set = []
#         pt = 0
#         ct = 0
#         cnt = 0
#         poison_id = []
#
#         for i in range(self.num_img):
#             img, gt = self.dataset[i]
#
#             # [ADDED FOR 1-CHANNEL]:
#             # Nếu dataset là ảnh một kênh nhưng ảnh hiện tại có nhiều kênh, chuyển về 1 kênh.
#             if self.dataset[0][0].shape[0] == 1 and img.shape[0] != 1:
#                 img = img.mean(dim=0, keepdim=True)
#
#             # Nếu ảnh nằm trong danh sách source để poison
#             if pt < num_poison and poison_indices[pt] == i:
#                 poison_id.append(cnt)
#                 gt = self.target_class  # đổi nhãn thành target_class
#                 img = img + self.mask.to(img.device) * (self.trigger.to(img.device) - img)
#                 pt += 1
#
#             # Nếu ảnh nằm trong danh sách cover, cũng áp dụng trigger (theo yêu cầu)
#             if ct < num_cover and cover_indices[ct] == i:
#                 img = img + self.mask.to(img.device) * (self.trigger.to(img.device) - img)
#                 ct += 1
#
#             img_set.append(img.unsqueeze(0))
#             label_set.append(gt)
#             cnt += 1
#
#         img_set = torch.cat(img_set, dim=0)
#         label_set = torch.LongTensor(label_set)
#         return img_set, poison_id, cover_indices, label_set
#
#
# class poison_transform():
#     def __init__(self, img_size, trigger, mask, target_class=0):
#         self.img_size = img_size
#         self.target_class = target_class  # by default : target_class = 0
#         self.trigger = trigger
#         self.mask = mask
#
#         # [ADDED FOR 1-CHANNEL]:
#         # Nếu trigger có nhiều kênh nhưng dataset cần 1 kênh, chuyển về 1 kênh
#         if self.trigger.shape[0] != 1:
#             self.trigger = self.trigger.mean(dim=0, keepdim=True)
#         if self.mask.shape[0] != 1:
#             self.mask = self.mask.mean(dim=0, keepdim=True)
#
#     def transform(self, data, labels):
#         data = data.clone()
#         labels = labels.clone()
#
#         # [ADDED FOR 1-CHANNEL]:
#         # Nếu data là ảnh một kênh nhưng có nhiều kênh, chuyển về 1 kênh.
#         if data.shape[1] != 1:
#             # Giả sử dataset cần 1 kênh, chuyển bằng cách lấy trung bình
#             data = data.mean(dim=1, keepdim=True)
#
#         # Áp dụng trigger: data = data + mask*(trigger - data)
#         data = data + self.mask.to(data.device) * (self.trigger.to(data.device) - data)
#         labels[:] = self.target_class
#
#         return data, labels


import os
import torch
import random
from torchvision.utils import save_image

class poison_generator():

    def __init__(self, img_size, dataset, poison_rate, cover_rate, trigger, mask, path, target_class = 0,
                 source_class = 1, cover_classes = [5,7]):

        self.img_size = img_size
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.cover_rate = cover_rate
        self.trigger = trigger
        self.mask = mask
        self.path = path  # path to save the dataset
        self.target_class = target_class # by default : target_class = 0
        self.source_class= source_class # by default : source_classes = 1
        self.cover_classes = cover_classes

        # number of images
        self.num_img = len(dataset)

    def generate_poisoned_training_set(self):

        # random sampling
        all_source_indices = []
        all_cover_indices = []

        for i in range(self.num_img):
            _, gt = self.dataset[i]

            if gt == self.source_class:
                all_source_indices.append(i)
            elif gt in self.cover_classes:
                all_cover_indices.append(i)

        random.shuffle(all_source_indices)
        random.shuffle(all_cover_indices)

        num_poison = int(self.num_img * self.poison_rate)
        num_cover = int(self.num_img * self.cover_rate)

        poison_indices = all_source_indices[:num_poison]
        cover_indices = all_cover_indices[:num_cover]
        poison_indices.sort() # increasing order
        cover_indices.sort() # increasing order

        img_set = []
        label_set = []
        pt = 0
        ct = 0
        for i in range(self.num_img):
            img, gt = self.dataset[i]

            if pt < num_poison and poison_indices[pt] == i:
                gt = self.target_class
                img = img + self.mask*(self.trigger - img)
                pt+=1

            if ct < num_cover and cover_indices[ct] == i:
                img = img + self.mask*(self.trigger - img)
                ct+=1

            # img_file_name = '%d.png' % i
            # img_file_path = os.path.join(self.path, img_file_name)
            # save_image(img, img_file_path)
            #print('[Generate Poisoned Set] Save %s' % img_file_path)

            img_set.append(img.unsqueeze(0))
            label_set.append(gt)

        img_set = torch.cat(img_set, dim=0)
        label_set = torch.LongTensor(label_set)
        #print("Poison indices:", poison_indices)
        #print("Cover indices:", cover_indices)
        return img_set, poison_indices, cover_indices, label_set



class poison_transform():
    def __init__(self, img_size, trigger, mask, target_class = 0):
        self.img_size = img_size
        self.trigger = trigger
        self.mask = mask
        self.target_class = target_class # by default : target_class = 0

    def transform(self, data, labels):
        data = data.clone()
        labels = labels.clone()
        # transform clean samples to poison samples

        labels[:] = self.target_class
        data = data + self.mask.to(data.device) * (self.trigger.to(data.device) - data)

        return data, labels