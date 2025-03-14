import os
from PIL import Image
from tqdm import tqdm
import argparse

def resize_images(input_dir, output_dir, target_size=(256, 256)):
    """
    Resize all images in input_dir to target_size and save to output_dir.

    Args:
        input_dir (str): Path to the directory containing original images (224x224).
        output_dir (str): Path to the directory where resized images (256x256) will be saved.
        target_size (tuple): Target size to resize images (default: (256, 256)).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[INFO] Created output directory: {output_dir}")

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_name in tqdm(image_files, desc='Resizing Images'):
        input_path = os.path.join(input_dir, image_name)
        output_path = os.path.join(output_dir, image_name)

        with Image.open(input_path) as img:
            img_resized = img.resize(target_size, Image.BICUBIC)  # You can also try Image.LANCZOS for better quality
            img_resized.save(output_path)

    print(f"[INFO] Resized {len(image_files)} images and saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize images from 224x224 to 256x256.")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing 224x224 triggers.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save resized 256x256 triggers.")
    args = parser.parse_args()

    resize_images(args.input_dir, args.output_dir, target_size=(256, 256))
