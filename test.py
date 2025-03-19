from PIL import Image
import numpy as np

# Đường dẫn đến ảnh cần kiểm tra (CẬP NHẬT LẠI CHO ĐÚNG)
image_path = "C:\\Users\\Admin\\Downloads\\99.png"

# Mở ảnh
img = Image.open(image_path)

# Chuyển ảnh thành numpy array để kiểm tra shape
img_array = np.array(img)

# In thông tin về ảnh
print(f"Ảnh: {image_path}")
print(f"Mode: {img.mode}")  # Kiểm tra kiểu ảnh (RGB, L, RGBA, v.v.)
print(f"Shape: {img_array.shape}")  # In kích thước ảnh dưới dạng numpy array

# Kiểm tra số kênh ảnh
if len(img_array.shape) == 2:
    print("✅ Ảnh này là grayscale (1 kênh).")
elif len(img_array.shape) == 3:
    print(f"⚠️ Ảnh này có {img_array.shape[2]} kênh (Có thể là RGB hoặc RGBA).")
