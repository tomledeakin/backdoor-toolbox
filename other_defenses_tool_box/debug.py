import os
import torch
import config
import numpy as np
from tqdm import tqdm
from PIL import Image
from collections import Counter
from other_defenses_tool_box.tools import generate_dataloader
from utils.supervisor import get_transforms
from utils import supervisor, tools


def save_sample_images(loader, save_dir, prefix='', num_samples=10):
    """
    Lưu num_samples hình ảnh từ loader vào thư mục save_dir,
    đặt tên file theo dạng: prefix_index_label.png
    """
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    for i, (imgs, labels) in enumerate(loader):
        for j in range(imgs.size(0)):
            if count >= num_samples:
                return
            img = imgs[j]
            label = labels[j].item()

            # Chuyển tensor thành hình ảnh PIL
            # Giả sử img là CHW (C=3,H,W)
            img = img.permute(1, 2, 0)  # CHW -> HWC
            img = img.numpy()
            # Chuyển từ [0,1] hoặc [-1,1] sang [0,255] tùy vào transform
            # Giả sử data_transform đã normalize với mean/std
            # Nếu bạn có mean/std, bạn có thể denormalize trước. Nếu không, cứ lưu trực tiếp.
            # Ở đây ta giả định đã normalize theo mean/std như CIFAR-10 (0.5,0.5,0.5)
            # Thử denormalize tạm:
            # WARNING: Bạn cần điều chỉnh mean/std theo dataset thực tế
            mean = np.array([0.5, 0.5, 0.5])
            std = np.array([0.5, 0.5, 0.5])
            img = img * std + mean
            img = np.clip(img, 0, 1) * 255
            img = img.astype(np.uint8)

            pil_img = Image.fromarray(img)

            filename = f"{prefix}_{count}_label_{label}.png"
            pil_img.save(os.path.join(save_dir, filename))
            count += 1


def main(args):
    # Tạo thư mục debug_data để lưu dữ liệu
    debug_dir = "debug_data"
    os.makedirs(debug_dir, exist_ok=True)

    # Tạo dataloader cho train, val, test
    train_loader = generate_dataloader(
        dataset=args.dataset,
        dataset_path=config.data_dir,
        batch_size=50,
        split='train',
        data_transform=get_transforms(args.dataset)['train'],
        shuffle=True,
        drop_last=False,
        noisy_test=False
    )

    val_loader = generate_dataloader(
        dataset=args.dataset,
        dataset_path=config.data_dir,
        batch_size=50,
        split='val',
        data_transform=get_transforms(args.dataset)['test'],
        shuffle=False,
        drop_last=False,
        noisy_test=False
    )

    test_loader = generate_dataloader(
        dataset=args.dataset,
        dataset_path=config.data_dir,
        batch_size=50,
        split='test',
        data_transform=get_transforms(args.dataset)['test'],
        shuffle=False,
        drop_last=False,
        noisy_test=False
    )

    # Thu thập toàn bộ dữ liệu để in thông tin
    def collect_all_data(loader):
        all_labels = []
        total = 0
        for imgs, labels in loader:
            total += labels.size(0)
            all_labels.extend(labels.tolist())
        return total, all_labels

    train_total, train_labels = collect_all_data(train_loader)
    val_total, val_labels = collect_all_data(val_loader)
    test_total, test_labels = collect_all_data(test_loader)

    # In ra độ dài dataset
    print("=== Dataset Info ===")
    print(f"Train set size: {train_total}")
    print(f"Val set size: {val_total}")
    print(f"Test set size: {test_total}")

    # In phân phối lớp
    def print_class_distribution(labels, name):
        counter = Counter(labels)
        print(f"Class distribution in {name}:")
        for cls, cnt in sorted(counter.items()):
            print(f"  Class {cls}: {cnt} samples")

    print_class_distribution(train_labels, 'train')
    print_class_distribution(val_labels, 'val')
    print_class_distribution(test_labels, 'test')

    # In ra số lớp
    all_classes = sorted(list(set(train_labels + val_labels + test_labels)))
    print(f"Number of classes: {len(all_classes)}")
    print(f"Classes: {all_classes}")

    # In ra một số thông tin quan trọng khác trong args
    print("=== Args Info ===")
    for arg_key, arg_val in vars(args).items():
        print(f"{arg_key}: {arg_val}")

    # Lưu một số mẫu từ train, val, test
    print("Saving sample images...")
    save_sample_images(train_loader, os.path.join(debug_dir, 'train_samples'), prefix='train', num_samples=10)
    save_sample_images(val_loader, os.path.join(debug_dir, 'val_samples'), prefix='val', num_samples=10)
    save_sample_images(test_loader, os.path.join(debug_dir, 'test_samples'), prefix='test', num_samples=10)

    # Bạn có thể lưu thêm dữ liệu như nhãn ra file text nếu muốn
    # Ví dụ, lưu phân phối lớp ra file
    with open(os.path.join(debug_dir, "class_distribution.txt"), 'w') as f:
        f.write("=== Class Distribution ===\n")
        f.write("Train:\n")
        counter_train = Counter(train_labels)
        for cls, cnt in sorted(counter_train.items()):
            f.write(f"Class {cls}: {cnt}\n")

        f.write("Val:\n")
        counter_val = Counter(val_labels)
        for cls, cnt in sorted(counter_val.items()):
            f.write(f"Class {cls}: {cnt}\n")

        f.write("Test:\n")
        counter_test = Counter(test_labels)
        for cls, cnt in sorted(counter_test.items()):
            f.write(f"Class {cls}: {cnt}\n")

    print("Debug info saved successfully.")


if __name__ == "__main__":
    # Giả sử args được khởi tạo ở đâu đó, hoặc bạn có thể tạo args giả lập
    # Tuỳ thuộc vào cách bạn chạy code này, bạn có thể truyền args từ bên ngoài
    # Dưới đây là ví dụ tạo args giả, bạn cần thay đổi cho phù hợp với môi trường thực tế
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--poison_type', type=str, default='badnet')
    parser.add_argument('--poison_rate', type=float, default=0.1)
    parser.add_argument('--defense', type=str, default='IBD_PSC')
    args = parser.parse_args()

    main(args)

