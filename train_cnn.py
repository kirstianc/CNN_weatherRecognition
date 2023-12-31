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
from torchvision.models import resnet18
from torch.nn import Linear
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main(train_loader, valid_loader):
    print("---- Starting Training ----")

    writer = SummaryWriter()
    df = pd.read_csv("image_class_map.csv")

    # setup
    model = resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = Linear(num_ftrs, len(df["Class"].unique()))
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

    print("Training model...")
    for epoch in range(10):
        # initialize
        running_loss = 0.0
        train_labels = []
        train_predictions = []
        valid_labels = []
        valid_predictions = []

        # train model
        model.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            train_labels.extend(labels.tolist())
            train_predictions.extend(predicted.tolist())

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        accuracy = (
            torch.tensor(train_labels) == torch.tensor(train_predictions)
        ).sum().item() / len(train_labels)
        precision = precision_score(train_labels, train_predictions, average="macro")
        recall = recall_score(train_labels, train_predictions, average="macro")
        f1 = f1_score(train_labels, train_predictions, average="macro")

        # log metrics to TensorBoard
        writer.add_scalar("Accuracy/train", accuracy, epoch)
        writer.add_scalar("Precision/train", precision, epoch)
        writer.add_scalar("Recall/train", recall, epoch)
        writer.add_scalar("F1/train", f1, epoch)

        # validation
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(valid_loader, 0):
                inputs, labels = data

                # forward pass
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                valid_labels.extend(labels.tolist())
                valid_predictions.extend(predicted.tolist())

            accuracy = (
                torch.tensor(valid_labels) == torch.tensor(valid_predictions)
            ).sum().item() / len(valid_labels)
            precision = precision_score(
                valid_labels, valid_predictions, average="macro"
            )
            recall = recall_score(valid_labels, valid_predictions, average="macro")
            f1 = f1_score(valid_labels, valid_predictions, average="macro")

            # log metrics to TensorBoard
            writer.add_scalar("Accuracy/valid", accuracy, epoch)
            writer.add_scalar("Precision/valid", precision, epoch)
            writer.add_scalar("Recall/valid", recall, epoch)
            writer.add_scalar("F1/valid", f1, epoch)

        # save model
        torch.save(model.state_dict(), "model.pth")

    print("Finished Training")


if __name__ == "__main__":
    main()
