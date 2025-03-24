import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm


# Dummy dataset cho ví dụ (giả sử input là ảnh 224x224 và num_classes là 200)
class DummyDataset(Dataset):
    def __init__(self, num_samples=1000, num_classes=200):
        self.num_samples = num_samples
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Tạo ảnh ngẫu nhiên với kích thước (3, 224, 224)
        img = torch.randn(3, 224, 224)
        # Tạo label ngẫu nhiên từ 0 đến num_classes-1
        label = torch.randint(0, self.num_classes, (1,)).item()
        return img, label


# Cấu hình dataset và dataloader
batch_size = 32
num_classes = 200
dataset = DummyDataset(num_samples=1000, num_classes=num_classes)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Khởi tạo model từ timm, đảm bảo rằng hàm entrypoint đã được chỉnh sửa để nhận num_classes
# Lưu ý: Nếu bạn có code custom và đã đăng ký model tên 'deit_base_patch16_224', thì nó sẽ ghi đè model gốc.
model = timm.create_model('deit_base_patch16_224', pretrained=False, num_classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Thiết lập optimizer và loss function
optimizer = optim.AdamW(model.parameters(), lr=0.001)  # Bạn có thể điều chỉnh LR cho phù hợp
criterion = nn.CrossEntropyLoss()

# Vòng lặp training đơn giản cho 5 epochs
num_epochs = 5
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        # Nếu model trả về tuple (cho model distilled) thì lấy trung bình của các head
        if isinstance(outputs, tuple):
            outputs = sum(outputs) / len(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}")
            running_loss = 0.0

print("Training completed!")
