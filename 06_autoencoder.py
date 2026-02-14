"""Autoencoder from Scratch — MNIST

Learns a compressed latent representation and reconstructs digit images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from common.utils import set_seed, get_device
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        # Encoder
        self.enc1 = nn.Linear(784, 256)
        self.enc2 = nn.Linear(256, latent_dim)
        # Decoder
        self.dec1 = nn.Linear(latent_dim, 256)
        self.dec2 = nn.Linear(256, 784)

    def encode(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.enc1(x))
        return self.enc2(x)

    def decode(self, z):
        z = F.relu(self.dec1(z))
        return torch.sigmoid(self.dec2(z))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


def train(num_epochs=15, batch_size=128, lr=1e-3, seed=42):
    set_seed(seed)
    device = get_device()

    transform = transforms.ToTensor()
    train_set = datasets.MNIST("data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    model = Autoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for X, _ in train_loader:
            X = X.to(device)
            recon = model(X)
            loss = F.mse_loss(recon, X.view(X.size(0), -1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X.size(0)

        avg_loss = epoch_loss / len(train_set)
        losses.append(avg_loss)
        print(f"  [AE] Epoch {epoch+1}/{num_epochs}  Recon Loss: {avg_loss:.6f}")

    # Plot: loss + original vs reconstructed
    model.eval()
    test_set = datasets.MNIST("data", train=False, transform=transform)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=True)
    X_sample, _ = next(iter(test_loader))
    X_sample = X_sample.to(device)
    with torch.no_grad():
        recon = model(X_sample).view(-1, 1, 28, 28).cpu()
    X_sample = X_sample.cpu()

    fig, axes = plt.subplots(3, 8, figsize=(12, 5))
    fig.suptitle("Autoencoder — MNIST", fontsize=14)
    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=10)
    axes[2, 0].set_ylabel("Difference", fontsize=10)
    for i in range(8):
        axes[0, i].imshow(X_sample[i, 0], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i, 0], cmap="gray")
        axes[1, i].axis("off")
        diff = (X_sample[i, 0] - recon[i, 0]).abs()
        axes[2, i].imshow(diff, cmap="hot")
        axes[2, i].axis("off")
    fig.tight_layout()
    fig.savefig("assets/06_autoencoder.png", dpi=150)
    plt.close(fig)
    print("  Plot saved → assets/06_autoencoder.png")
    return losses


if __name__ == "__main__":
    train()
