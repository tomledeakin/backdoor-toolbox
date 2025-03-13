from datasets import load_dataset

dataset = load_dataset('parquet', data_files={
    'train': 'data/tiny-imagenet-200-clean/data/train-00000-of-00001.parquet',
    'validation': 'data/tiny-imagenet-200-clean/data/validation-00000-of-00001.parquet',
    'test': 'data/tiny-imagenet-200-clean/data/test-00000-of-00001.parquet',
})

print(dataset)
