# import os
# import torch
# import random
# from torchvision.utils import save_image
# from torch import nn
# from torchvision import transforms
#
# class poison_generator():
#     def __init__(self, img_size, dataset, poison_rate, path, trigger_mark, trigger_mask, target_class=0):
#         self.img_size = img_size
#         self.dataset = dataset
#         self.poison_rate = poison_rate
#         self.path = path  # path to save the dataset
#         self.target_class = target_class  # by default : target_class = 0
#         self.trigger_mark = trigger_mark
#         self.trigger_mask = trigger_mask
#
#         # number of images
#         self.num_img = len(dataset)
#
#         # shape of the patch trigger
#         self.dx, self.dy = trigger_mask.shape
#
#         # [ADDED FOR 1-CHANNEL]:
#         # Nếu dataset là ảnh grayscale (1 kênh) mà trigger_mark hoặc trigger_mask có nhiều kênh,
#         # chuyển chúng về grayscale bằng cách lấy trung bình các kênh.
#         sample_img, _ = dataset[0]
#         if sample_img.shape[0] == 1:
#             if self.trigger_mark.shape[0] != 1:
#                 self.trigger_mark = self.trigger_mark.mean(dim=0, keepdim=True)
#             if self.trigger_mask.shape[0] != 1:
#                 # Nếu trigger_mask là RGB, ta chỉ lấy channel 0 (hoặc có thể dùng mean)
#                 self.trigger_mask = self.trigger_mask.mean(dim=0, keepdim=True)
#
#     def generate_poisoned_training_set(self):
#         torch.manual_seed(0)
#         random.seed(0)
#         # random sampling
#         id_set = list(range(self.num_img))
#         random.shuffle(id_set)
#         num_poison = int(self.num_img * self.poison_rate)
#         poison_indices = sorted(id_set[:num_poison])
#
#         img_set = []
#         label_set = []
#         pt = 0
#         cnt = 0
#         poison_id = []
#
#         for i in range(self.num_img):
#             img, gt = self.dataset[i]
#
#             # [ADDED FOR 1-CHANNEL]:
#             # Nếu ảnh có số kênh khác với dataset mong đợi, chuyển về 1 kênh nếu cần
#             if self.dataset[0][0].shape[0] == 1 and img.shape[0] != 1:
#                 img = img.mean(dim=0, keepdim=True)
#
#             # Poisoned image: đổi nhãn và thêm trigger
#             if pt < num_poison and poison_indices[pt] == i:
#                 poison_id.append(cnt)
#                 gt = self.target_class  # đổi nhãn thành target_class
#                 # Blend ảnh: công thức: img = img + trigger_mask * (trigger_mark - img)
#                 img = img + self.trigger_mask.to(img.device) * (self.trigger_mark.to(img.device) - img)
#                 pt += 1
#
#             # (Nếu muốn lưu ảnh riêng lẻ, mở comment save_image ở đây)
#             img_set.append(img.unsqueeze(0))
#             label_set.append(gt)
#             cnt += 1
#
#         img_set = torch.cat(img_set, dim=0)
#         label_set = torch.LongTensor(label_set)
#         poison_indices = poison_id
#         return img_set, poison_indices, label_set
#
# class poison_transform():
#     def __init__(self, img_size, trigger_mark, trigger_mask, target_class=0):
#         self.img_size = img_size
#         self.target_class = target_class
#         self.trigger_mark = trigger_mark
#         self.trigger_mask = trigger_mask
#         self.dx, self.dy = trigger_mask.shape
#
#         # [ADDED FOR 1-CHANNEL]:
#         # Nếu dữ liệu cần là grayscale (1 kênh) nhưng trigger_mark hoặc trigger_mask có nhiều kênh, chuyển chúng về 1 kênh
#         # Ở đây không có dataset để kiểm tra, ta dựa trên shape của trigger_mark
#         if self.trigger_mark.shape[0] != 1:
#             self.trigger_mark = self.trigger_mark.mean(dim=0, keepdim=True)
#         if self.trigger_mask.shape[0] != 1:
#             self.trigger_mask = self.trigger_mask.mean(dim=0, keepdim=True)
#
#     def transform(self, data, labels):
#         data, labels = data.clone(), labels.clone()
#         # Nếu data không đúng kích thước, không cần resize ở đây vì mong đợi data có shape (B, C, H, W)
#
#         # Blend: img = img + trigger_mask * (trigger_mark - img)
#         data = data + self.trigger_mask.to(data.device) * (self.trigger_mark.to(data.device) - data)
#         labels[:] = self.target_class
#
#         return data, labels



import os
import torch
import random
from torchvision.utils import save_image

"""Trojan backdoor attack
Adopting the trojan patch trigger from [TrojanNN](https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=2782&context=cstech)
"""

class poison_generator():

    def __init__(self, img_size, dataset, poison_rate, path, trigger_mark, trigger_mask, target_class=0):

        self.img_size = img_size
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.path = path  # path to save the dataset
        self.target_class = target_class # by default : target_class = 0
        self.trigger_mark = trigger_mark
        self.trigger_mask = trigger_mask

        # number of images
        self.num_img = len(dataset)

        # shape of the patch trigger
        self.dx, self.dy = trigger_mask.shape

    def generate_poisoned_training_set(self):
        torch.manual_seed(0)
        random.seed(0)
        # torch.manual_seed(666)
        # random.seed(666)

        # random sampling
        id_set = list(range(0,self.num_img))
        random.shuffle(id_set)
        num_poison = int(self.num_img * self.poison_rate)
        poison_indices = id_set[:num_poison]
        poison_indices.sort() # increasing order


        img_set = []
        label_set = []
        pt = 0
        cnt = 0
        poison_id = []

        for i in range(self.num_img):
            img, gt = self.dataset[i]

            # poisoned image
            if pt < num_poison and poison_indices[pt] == i:
                poison_id.append(cnt)
                gt = self.target_class # change the label to the target class
                img = img + self.trigger_mask * (self.trigger_mark - img)
                pt+=1

            # img_file_name = '%d.png' % cnt
            # img_file_path = os.path.join(self.path, img_file_name)
            # save_image(img, img_file_path)
            # print('[Generate Poisoned Set] Save %s' % img_file_path)

            img_set.append(img.unsqueeze(0))
            label_set.append(gt)
            cnt+=1

        img_set = torch.cat(img_set, dim=0)
        label_set = torch.LongTensor(label_set)
        poison_indices = poison_id
        print("Poison indices:", poison_indices)
        return img_set, poison_indices, label_set


class poison_transform():

    def __init__(self, img_size, trigger_mark, trigger_mask, target_class=0):

        self.img_size = img_size
        self.target_class = target_class
        self.trigger_mark = trigger_mark
        self.trigger_mask = trigger_mask
        self.dx, self.dy = trigger_mask.shape

    def transform(self, data, labels):
        data, labels = data.clone(), labels.clone()
        data = data + self.trigger_mask.to(data.device) * (self.trigger_mark.to(data.device) - data)
        labels[:] = self.target_class

        # debug
        # from torchvision.utils import save_image
        # from torchvision import transforms
        # # preprocess = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        # reverse_preprocess = transforms.Normalize([-0.4914/0.247, -0.4822/0.243, -0.4465/0.261], [1/0.247, 1/0.243, 1/0.261])
        # save_image(reverse_preprocess(data)[0], 'a.png')

        return data, labels