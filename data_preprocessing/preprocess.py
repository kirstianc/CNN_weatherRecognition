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
====================== END OF MODIFICATION HISTORY ============================
"""
import os
import shutil
import random
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Set the paths
data_dir = "CNN_weatherRecognition/dataset/"
train_dir = "CNN_weatherRecognition/data_preprocessing/train/"
val_dir = "CNN_weatherRecognition/data_preprocessing/validation/"
test_dir = "CNN_weatherRecognition/data_preprocessing/test/"

# Create directories for train, validation, and test sets
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Define the transformation
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

# Load the dataset
dataset = ImageFolder(root=data_dir, transform=transform)

# Calculate dataset sizes
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

# Split the dataset
train_set, val_set, test_set = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size]
)

# Create data loaders
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)


# Save the split datasets to respective directories
def save_dataset(loader, directory):
    for i, (data, labels) in enumerate(loader):
        for j in range(len(data)):
            img_path = os.path.join(directory, f"{i * len(data) + j}.jpg")
            shutil.copy(dataset.imgs[i * len(data) + j][0], img_path)


# Save datasets
save_dataset(train_loader, train_dir)
save_dataset(val_loader, val_dir)
save_dataset(test_loader, test_dir)
