# coding: utf-8
# NAME: Custom_Dataset.py
"""
AUTHOR: Ian Chavez

Unpublished-rights reserved under the copyright laws of the United States.

This data and information is proprietary to, and a valuable trade secret of Ian Chavez. It is given in confidence by
Ian Chavez. Its use, duplication, or disclosure is subject to the restrictions set forth in the License Agreement under which it has been
distributed.

Unpublished Copyright © 2023 Ian Chavez

All Rights Reserved
"""
import torch
from torch.utils.data import Dataset
import os
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

        # Create a mapping of class names to integers
        self.class_to_num = {
            class_name: i
            for i, class_name in enumerate(self.dataframe["Class"].unique())
        }

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 1])
        image = Image.open(img_name)
        label = self.class_to_num[self.dataframe.iloc[idx, 2]]

        if self.transform:
            image = self.transform(image)

        return image, label
