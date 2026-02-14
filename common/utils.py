import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def plot_loss(losses, title: str, save_path: str, xlabel="Epoch", ylabel="Loss"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses, color="steelblue", linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved → {save_path}")


def plot_samples(images, title: str, save_path: str, nrow=8):
    """Plot a grid of images (numpy or tensor, shape [N, H, W] or [N, 1, H, W])."""
    n = min(len(images), nrow * nrow)
    fig, axes = plt.subplots(nrow, nrow, figsize=(8, 8))
    fig.suptitle(title, fontsize=14)
    for i, ax in enumerate(axes.flat):
        if i < n:
            img = images[i]
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = img.transpose(1, 2, 0)
            if img.ndim == 3 and img.shape[2] == 1:
                img = img.squeeze(-1)
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved → {save_path}")
