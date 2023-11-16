# coding: utf-8
# NAME: preprocess.py
"""
AUTHOR: Ian Chavez

Unpublished-rights reserved under the copyright laws of the United States.

This data and information is proprietary to, and a valuable trade secret of Ian Chavez. It is given in confidence by
Ian Chavez. Its use, duplication, or disclosure is subject to the restrictions set forth in the License Agreement under which it has been
distributed.

Unpublished Copyright © 2023 Ian Chavez

All Rights Reserved
"""
"""
========================== MODIFICATION HISTORY ==============================
11/15/23:
MOD: Creation of file and initial function
AUTHOR: Ian Chavez
COMMENT: n/a

11/16/23:
MOD: Fix transformation and dataset loading
AUTHOR: Ian Chavez
COMMENT: n/a
====================== END OF MODIFICATION HISTORY ============================
"""
print("---- Starting Preprocessing ----")
print("Importing libraries...")
import os
import pandas as pd
import torch
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image

print("Setting paths...")
# Set the paths
data_dir = "./dataset/"
csv_dir = ".image_class_map.csv"
processed_datasets_dir = "./processed_datasets/"
train_dir = "./processed_datasets/train/"
val_dir = "./processed_datasets/validation/"
test_dir = "./processed_datasets/test/"

print("Creating directories...")
# Create directories for train, validation, and test sets
os.makedirs(processed_datasets_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Read csv file
df = pd.read_csv(csv_dir)


# Define custom dataset class
class CustomDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 1])
        image = Image.open(img_name).convert("RGB")
        label = int(self.dataframe.iloc[idx, 2])  # Assuming class is in the third column

        if self.transform:
            image = self.transform(image)

        return image, label

print("Composing transformations...")
# Transformations to be applied to the dataset
transform = v2.Compose(
    [
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        v2.RandomHorizontalFlip(p=0.5),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

print("Creating dataset...")
dataset = CustomDataset(dataframe=df, root_dir=data_dir, transform=transform)

print("Calculating dataset sizes...")
# Calculate dataset sizes
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

print("Splitting dataset...")
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

print("Creating dataloaders...")
# Split dataset into train, validation, and test sets
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
valid_loader = DataLoader(val_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=True)

print("Saving datasets...")
# Save datasets
torch.save(train_set, train_dir)
torch.save(val_set, val_dir)
torch.save(test_set, test_dir)

print("---- Finished Preprocessing ----")