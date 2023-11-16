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
import Custom_Dataset as cd
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image

print("Setting paths...")
# Set the paths
data_dir = "./dataset/"
csv_dir = "image_class_map.csv"
processed_datasets_dir = "./processed_datasets/"

print("Creating directories...")
# Create directories for train, validation, and test sets
os.makedirs(processed_datasets_dir, exist_ok=True)

print("Reading csv file...")
# Read csv file
df = pd.read_csv(csv_dir)

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
dataset = cd.CustomDataset(dataframe=df, root_dir=data_dir, transform=transform)

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
torch.save(train_set, os.path.join(processed_datasets_dir, 'train_dataset.pth'))
torch.save(val_set, os.path.join(processed_datasets_dir, 'val_dataset.pth'))
torch.save(test_set, os.path.join(processed_datasets_dir, 'test_dataset.pth'))

print("---- Finished Preprocessing ----")