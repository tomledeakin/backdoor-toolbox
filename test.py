import torch

file_path = "C:/Users/Admin/Downloads/imgs"  # đường dẫn đầy đủ đến file

data = torch.load(file_path)
print(type(data))

# Nếu data là một Tensor:
if isinstance(data, torch.Tensor):
    print("Tensor shape:", data.shape)

# Nếu data là dictionary hoặc list:
elif isinstance(data, dict):
    print("Keys:", data.keys())
elif isinstance(data, list):
    print("List length:", len(data))
else:
    print("Dạng dữ liệu khác:", type(data))
