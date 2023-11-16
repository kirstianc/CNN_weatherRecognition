# coding: utf-8
# NAME: train_model.py
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
11/16/23:
MOD: Creation of file and initial function
AUTHOR: Ian Chavez
COMMENT: n/a
====================== END OF MODIFICATION HISTORY ============================
"""
# train CNN model using pth dataset file from directory (./processed_datasets/train_dataset.pth)
# save model in directory (./model/cnn_model.pth)
# save training loss and accuracy in directory (./model/cnn_model_training.csv)
# save validation loss and accuracy in directory (./model/cnn_model_validation.csv)
# save training and validation loss and accuracy plots in directory (./model/cnn_model_training.png)
print("---- Starting Training CNN ----")

print("Importing libraries...")
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '../')
import Custom_Dataset as cd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 3 input channels, 6 output channels, 5x5 kernel
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 max pooling
        self.conv2 = nn.Conv2d(6, 16, 5)  # 6 input channels, 16 output channels, 5x5 kernel
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # fully connected layer
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

print("Loading dataset...")
dataset = torch.load("../processed_datasets/train_dataset.pth")

print("Creating instance of CNN model...")
model = CNNModel()

print("Defining loss function and optimizer...")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print("Training model...")
num_epochs = 10  
batch_size = 64 
print("Number of epochs: ", num_epochs)
print("Batch size: ", batch_size)

for epoch in range(num_epochs):
    for inputs, labels in DataLoader(dataset, batch_size=batch_size, shuffle=True):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print("Saving trained model...")
torch.save(model, "./model/cnn_model.pth")

# TODO: save training and validation loss and accuracy

# TODO: save plots of training loss and accuracy

plt.plot(training_loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.savefig("./model/cnn_model_training_loss.png")
plt.close()

plt.plot(training_accuracy_history)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Over Time')
plt.savefig("./model/cnn_model_training_accuracy.png")
plt.close()

print("---- Finished Training CNN ----")