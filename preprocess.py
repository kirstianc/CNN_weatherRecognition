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
import os
import pandas as pd
import torch
import Custom_Dataset as cd
from torchvision.transforms import ToTensor
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import Lambda

from PIL import Image


# convert grayscale to rgb (resolves error with resnet)
def convert_to_rgb(image):
    return image.convert("RGB")


def main():
    print("---- Starting Preprocessing ----")

    # set paths
    data_dir = "./dataset/"
    csv_dir = "image_class_map.csv"
    processed_datasets_dir = "./processed_datasets/"

    print("Creating processed_datasets directory...")
    os.makedirs(processed_datasets_dir, exist_ok=True)

    df = pd.read_csv(csv_dir)

    print("Composing transformations...")
    transform = v2.Compose(
        [
            Lambda(convert_to_rgb),
            v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
            v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            ToTensor(),
        ]
    )

    dataset = cd.CustomDataset(dataframe=df, root_dir=data_dir, transform=transform)

    # calculate dataset sizes
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    print("Splitting datasets...")
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    # create dataloaders
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    valid_loader = DataLoader(val_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True)

    print("Saving datasets...")
    torch.save(train_set, os.path.join(processed_datasets_dir, "train_dataset.pth"))
    torch.save(val_set, os.path.join(processed_datasets_dir, "val_dataset.pth"))
    torch.save(test_set, os.path.join(processed_datasets_dir, "test_dataset.pth"))

    print("---- Finished Preprocessing ----")
    return train_loader, valid_loader, test_loader
