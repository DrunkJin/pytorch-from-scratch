"""Convolutional Neural Network from Scratch — MNIST

Conv → ReLU → Pool → Conv → ReLU → Pool → FC → Softmax.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from common.utils import set_seed, plot_loss, get_device


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # [B, 32, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))   # [B, 64, 7, 7]
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def train(num_epochs=5, batch_size=128, lr=1e-3, seed=42):
    set_seed(seed)
    device = get_device()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST("data", train=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000)

    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X.size(0)

        avg_loss = epoch_loss / len(train_set)
        losses.append(avg_loss)

        model.eval()
        correct = 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X).argmax(dim=1)
                correct += (preds == y).sum().item()
        acc = correct / len(test_set)
        print(f"  [CNN] Epoch {epoch+1}/{num_epochs}  Loss: {avg_loss:.4f}  Test Acc: {acc:.4f}")

    plot_loss(losses, "CNN — MNIST", "assets/04_cnn.png")
    return losses


if __name__ == "__main__":
    train()
