"""Multi-Layer Perceptron from Scratch — MNIST

Handwritten digit classification with a 2-hidden-layer MLP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from common.utils import set_seed, plot_loss, get_device


class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden=256, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train(num_epochs=10, batch_size=128, lr=1e-3, seed=42):
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

    model = MLP().to(device)
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

        # Evaluate
        model.eval()
        correct = 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X).argmax(dim=1)
                correct += (preds == y).sum().item()
        acc = correct / len(test_set)
        print(f"  [MLP] Epoch {epoch+1}/{num_epochs}  Loss: {avg_loss:.4f}  Test Acc: {acc:.4f}")

    plot_loss(losses, "MLP — MNIST", "assets/03_mlp.png")
    return losses


if __name__ == "__main__":
    train()
