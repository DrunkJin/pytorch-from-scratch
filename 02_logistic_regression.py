"""Logistic Regression from Scratch in PyTorch

Binary classification on the moons dataset.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common.utils import set_seed


def make_moons(n_samples=500, noise=0.2, seed=42):
    """Generate 2D moons dataset."""
    rng = np.random.RandomState(seed)
    n = n_samples // 2
    outer = np.linspace(0, np.pi, n)
    inner = np.linspace(0, np.pi, n)
    x_outer = np.cos(outer) + rng.normal(0, noise, n)
    y_outer = np.sin(outer) + rng.normal(0, noise, n)
    x_inner = 1 - np.cos(inner) + rng.normal(0, noise, n)
    y_inner = 1 - np.sin(inner) - 0.5 + rng.normal(0, noise, n)
    X = np.vstack([np.column_stack([x_outer, y_outer]),
                   np.column_stack([x_inner, y_inner])]).astype(np.float32)
    y = np.hstack([np.zeros(n), np.ones(n)]).astype(np.float32)
    return torch.from_numpy(X), torch.from_numpy(y)


def train(num_epochs=200, seed=42):
    set_seed(seed)

    X, y = make_moons(500, noise=0.2, seed=seed)

    # Parameters
    w = torch.randn(2, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    lr = 0.5

    losses = []
    for epoch in range(num_epochs):
        logits = X @ w + b
        loss = F.binary_cross_entropy_with_logits(logits, y)

        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()

        losses.append(loss.item())
        if (epoch + 1) % 50 == 0:
            preds = (logits > 0).float()
            acc = (preds == y).float().mean().item()
            print(f"  [Logistic] Epoch {epoch+1}/{num_epochs}  Loss: {loss.item():.4f}  Acc: {acc:.3f}")

    # Final accuracy
    with torch.no_grad():
        preds = (X @ w + b > 0).float()
        acc = (preds == y).float().mean().item()
    print(f"  Final accuracy: {acc:.3f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(losses, color="steelblue", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)

    # Decision boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    with torch.no_grad():
        zz = torch.sigmoid(grid @ w + b).numpy().reshape(xx.shape)
    ax2.contourf(xx, yy, zz, levels=50, cmap="RdBu_r", alpha=0.6)
    ax2.scatter(X[y == 0, 0], X[y == 0, 1], c="blue", s=10, alpha=0.6, label="Class 0")
    ax2.scatter(X[y == 1, 0], X[y == 1, 1], c="red", s=10, alpha=0.6, label="Class 1")
    ax2.set_title("Decision Boundary")
    ax2.legend()

    fig.tight_layout()
    fig.savefig("assets/02_logistic_regression.png", dpi=150)
    plt.close(fig)
    print("  Plot saved → assets/02_logistic_regression.png")
    return losses


if __name__ == "__main__":
    train()
