"""Linear Regression from Scratch in PyTorch

Learns y = 2x + 1 with noise using manual gradient computation,
then compares with autograd.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common.utils import set_seed


def train(num_epochs=100, n_samples=200, seed=42):
    set_seed(seed)

    # Generate synthetic data: y = 2x + 1 + noise
    X = torch.randn(n_samples, 1)
    y = 2 * X + 1 + 0.3 * torch.randn(n_samples, 1)

    # Parameters (from scratch)
    w = torch.randn(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    lr = 0.1

    losses = []
    for epoch in range(num_epochs):
        y_pred = X * w + b
        loss = ((y_pred - y) ** 2).mean()

        loss.backward()

        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()

        losses.append(loss.item())
        if (epoch + 1) % 20 == 0:
            print(f"  [Linear] Epoch {epoch+1}/{num_epochs}  Loss: {loss.item():.4f}  w: {w.item():.3f}  b: {b.item():.3f}")

    print(f"  Final: w = {w.item():.4f} (true: 2.0), b = {b.item():.4f} (true: 1.0)")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(losses, color="steelblue", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)

    ax2.scatter(X.numpy(), y.numpy(), alpha=0.4, s=15, label="Data")
    x_line = torch.linspace(X.min(), X.max(), 100).unsqueeze(1)
    with torch.no_grad():
        y_line = x_line * w + b
    ax2.plot(x_line.numpy(), y_line.numpy(), color="red", linewidth=2, label=f"y = {w.item():.2f}x + {b.item():.2f}")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Linear Regression Fit")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("assets/01_linear_regression.png", dpi=150)
    plt.close(fig)
    print("  Plot saved → assets/01_linear_regression.png")
    return losses


if __name__ == "__main__":
    train()
