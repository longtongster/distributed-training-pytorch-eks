from typing import Any, Callable

import matplotlib.pyplot as plt
import torch
import boto3
from io import BytesIO
import os

from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x: torch.Tensor) -> Any:
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def test(dataloader: DataLoader, model: nn.Module, loss_fn: torch.optim.Optimizer) -> None:
    # set the model to evaluation
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss: float = 0
    correct: float = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")  # noqa: E501


training_data = datasets.MNIST(root=".", train=True, download=True, transform=ToTensor())
test_data = datasets.MNIST(root=".", train=False, download=True, transform=ToTensor())

loaded_train = DataLoader(training_data, batch_size=128, shuffle=True)
loaded_test = DataLoader(test_data, batch_size=128, shuffle=True)

model = NeuralNetwork()
print(model)

# load the saved state dictionary
state_dict = torch.load("model.pth", weights_only=True)
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

# Load the modified state dict
model.load_state_dict(new_state_dict)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

test(loaded_test, model, loss_function)
print("Done!")

