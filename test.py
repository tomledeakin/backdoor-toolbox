import cv2
import os


def convert_to_grayscale(image_path):
    # Đọc ảnh màu
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Không thể đọc ảnh. Hãy kiểm tra đường dẫn.")

    # Chuyển đổi sang ảnh xám
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Tạo tên file mới
    base, ext = os.path.splitext(image_path)
    new_image_path = f"{base}_gray{ext}"

    # Lưu ảnh mới
    cv2.imwrite(new_image_path, gray_image)
    print(f"Ảnh đã được lưu tại: {new_image_path}")

    return new_image_path

# Ví dụ sử dụng:
convert_to_grayscale("E:\\backdoor-toolbox\\triggers\\phoenix_corner_28.png")
