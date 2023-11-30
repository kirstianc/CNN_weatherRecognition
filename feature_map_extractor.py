# coding: utf-8
# NAME: feature_map_extractor.py
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
from torchvision.models import resnet18
from torch.nn import Linear
from torchvision.transforms import transforms
from PIL import Image
import pandas as pd
from torch.utils.tensorboard import SummaryWriter


class FeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = torch.nn.Sequential(
            *list(model.children())[:-1]
        )  # Adjust this depending on your model

    def forward(self, x):
        return self.features(x)


# Load image paths and classes
df = pd.read_csv("image_class_map.csv")

# Define the CNN model
model = resnet18(pretrained=False)

# Load the trained model
model.load_state_dict(torch.load("model_18res_5epoch.pth"))
num_ftrs = model.fc.in_features
model.fc = Linear(num_ftrs, len(df["Class"].unique()))

# Wrap the model with the feature extractor
model = FeatureExtractor(model)

# Load an image
image = Image.open("/dataset/dew/2208.jpg")
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
image = transform(image).unsqueeze(0)  # Add batch dimension

# Create a SummaryWriter
writer = SummaryWriter("runs/feature_maps")

# Extract feature maps
with torch.no_grad():
    feature_maps = model(image)

# Normalize feature maps and add to TensorBoard
feature_maps = feature_maps - feature_maps.min()
feature_maps = feature_maps / feature_maps.max()
writer.add_images("Feature_Maps", feature_maps, 0)
writer.close()
