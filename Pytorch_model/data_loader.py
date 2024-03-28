import torch
import torch.nn as nn

from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def download_dataset(dataset_name: str):
    datasets = load_dataset(dataset_name)
    return datasets


def split_data(TEST_SIZE: float, dataset):
    datasets = dataset["train"].train_test_split(test_size=TEST_SIZE)
    return datasets


def create_dataloader(datasets, IMG_SIZE: int):
    img_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    return img_transform

if __name__ == "__main__":
    DATASET_NAME = 'cats_vs_dogs'
    download_dataset(DATASET_NAME)
    TEST_SIZE = 0.2
    datasets = download_dataset(TEST_SIZE, DATASET_NAME)
    data_loader = create_dataloader(datasets, 128)