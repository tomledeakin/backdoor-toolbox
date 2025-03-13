# from datasets import load_dataset
#
# dataset = load_dataset('slegroux/tiny-imagenet-200-clean')
#
# print(dataset)



import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# Load dataset
dataset = load_dataset('slegroux/tiny-imagenet-200-clean')

# Output directory
output_dir = 'data/imagenet-200'
os.makedirs(output_dir, exist_ok=True)

# Convert each split
def save_split(split_name):
    split = dataset[split_name]
    for idx, sample in tqdm(enumerate(split), total=len(split), desc=f"Processing {split_name}"):
        label = sample['label']
        label_dir = os.path.join(output_dir, split_name, str(label))
        os.makedirs(label_dir, exist_ok=True)

        # Save image
        image = sample['image']
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image.save(os.path.join(label_dir, f'{idx}.jpg'))

# Save all splits
save_split('train')
save_split('validation')
save_split('test')
