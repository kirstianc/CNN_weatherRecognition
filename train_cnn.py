# coding: utf-8
# NAME: train_cnn.py
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
import pandas as pd
from torchvision.models import resnet34
from torch.nn import Linear
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main(train_loader, valid_loader):
    print("---- Starting Training ----")

    # create a SummaryWriter
    writer = SummaryWriter()

    print("Loading image class map...")
    # Load image paths and classes
    df = pd.read_csv("image_class_map.csv")

    print("Creating model...")
    # Define the CNN model
    model = resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = Linear(num_ftrs, len(df["Class"].unique()))

    print("Defining loss function and optimizer...")
    # Define loss function and optimizer
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

    print("Training model...")
    # Training loop
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        all_labels = []
        all_predictions = []

        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.tolist())
            all_predictions.extend(predicted.tolist())

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        accuracy = (
            torch.tensor(all_labels) == torch.tensor(all_predictions)
        ).sum().item() / len(all_labels)
        precision = precision_score(all_labels, all_predictions, average="macro")
        recall = recall_score(all_labels, all_predictions, average="macro")
        f1 = f1_score(all_labels, all_predictions, average="macro")

        # log metrics to TensorBoard
        writer.add_scalar("Accuracy/train", accuracy, epoch)
        writer.add_scalar("Precision/train", precision, epoch)
        writer.add_scalar("Recall/train", recall, epoch)
        writer.add_scalar("F1/train", f1, epoch)
        # Save the model
        torch.save(model.state_dict(), "model.pth")

    print("Finished Training")


if __name__ == "__main__":
    main()
