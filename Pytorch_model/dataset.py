import torch
import torch.nn as nn
from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CatDogDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        images = self.data[idx]["image"]
        labels = self.data[idx]["label"]
        
        if self.transform:
            images = self.transform(images)

        labels = torch.tensor(labels, dtype=torch.long)
        
        return images, labels