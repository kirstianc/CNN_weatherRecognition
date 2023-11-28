import pandas as pd
import torch
from torchvision.models import resnet18
from torch.nn import Linear
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter


def test_cnn(test_loader):
    # Load image paths and classes
    df = pd.read_csv("image_class_map.csv")

    # Define the CNN model
    model = resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = Linear(num_ftrs, len(df["Class"].unique()))

    # Load the trained model
    model.load_state_dict(torch.load("model.pth"))

    # Test the model
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    # create a SummaryWriter
    writer = SummaryWriter()

    with torch.no_grad():
        import pandas as pd


import torch
from torchvision.models import resnet18
from torch.nn import Linear
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter


def test_cnn(test_loader):
    # Load image paths and classes
    df = pd.read_csv("image_class_map.csv")

    # Define the CNN model
    model = resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = Linear(num_ftrs, len(df["Class"].unique()))

    # Load the trained model
    model.load_state_dict(torch.load("model.pth"))

    # Test the model
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    # create a SummaryWriter
    writer = SummaryWriter()

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            images, labels = data
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)
            if len(images.shape) == 3:
                images = images.unsqueeze(1)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.tolist())
            all_predictions.extend(predicted.tolist())

            accuracy = (
                torch.tensor(all_labels) == torch.tensor(all_predictions)
            ).sum().item() / len(all_labels)
            precision = precision_score(all_labels, all_predictions, average="macro")
            recall = recall_score(all_labels, all_predictions, average="macro")
            f1 = f1_score(all_labels, all_predictions, average="macro")

            # log metrics to TensorBoard
            writer.add_scalar("Accuracy/test", accuracy, i)
            writer.add_scalar("Precision/test", precision, i)
            writer.add_scalar("Recall/test", recall, i)
            writer.add_scalar("F1/test", f1, i)

        print(
            "Accuracy of the network on the test images: %d %%"
            % (100 * correct / total)
        )

    # close the writer after testing
    writer.close()
