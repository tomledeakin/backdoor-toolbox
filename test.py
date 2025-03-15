import torch

# Đường dẫn file labels trên Windows (chú ý dấu '\\' hoặc 'r' để tránh lỗi escape character)
label_path = r"C:\Users\Admin\Downloads\labels"

# Load file labels
labels = torch.load(label_path)

# In thông tin
print("Shape of labels:", labels.shape)  # Xem kích thước, số lượng nhãn
print("First 10 labels:", labels[:10])  # In thử 10 label đầu tiên
