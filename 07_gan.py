"""GAN (Generative Adversarial Network) from Scratch — MNIST

Generator creates fake digits, Discriminator tries to tell real from fake.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from common.utils import set_seed, get_device, plot_samples
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


def train(num_epochs=30, batch_size=128, latent_dim=64, lr=2e-4, seed=42):
    set_seed(seed)
    device = get_device()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_set = datasets.MNIST("data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    G = Generator(latent_dim).to(device)
    D = Discriminator().to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    g_losses, d_losses = [], []
    fixed_z = torch.randn(64, latent_dim, device=device)

    for epoch in range(num_epochs):
        g_loss_sum = d_loss_sum = 0
        count = 0
        for real, _ in train_loader:
            real = real.to(device)
            bs = real.size(0)

            # --- Discriminator ---
            z = torch.randn(bs, latent_dim, device=device)
            fake = G(z).detach()
            d_real = D(real)
            d_fake = D(fake)
            d_loss = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) + \
                     F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            # --- Generator ---
            z = torch.randn(bs, latent_dim, device=device)
            fake = G(z)
            g_out = D(fake)
            g_loss = F.binary_cross_entropy_with_logits(g_out, torch.ones_like(g_out))
            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            g_loss_sum += g_loss.item()
            d_loss_sum += d_loss.item()
            count += 1

        g_losses.append(g_loss_sum / count)
        d_losses.append(d_loss_sum / count)
        print(f"  [GAN] Epoch {epoch+1}/{num_epochs}  D_loss: {d_losses[-1]:.4f}  G_loss: {g_losses[-1]:.4f}")

    # Generate samples
    G.eval()
    with torch.no_grad():
        samples = G(fixed_z).cpu()
        samples = (samples + 1) / 2  # denormalize to [0, 1]

    # Plot: loss curves + generated samples
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(g_losses, label="Generator", color="steelblue")
    ax1.plot(d_losses, label="Discriminator", color="coral")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("GAN Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    grid = torch.cat([samples[i, 0] for i in range(min(64, len(samples)))]).view(8, 8, 28, 28)
    grid = grid.permute(0, 2, 1, 3).reshape(8 * 28, 8 * 28)
    ax2.imshow(grid.numpy(), cmap="gray")
    ax2.set_title("Generated Digits")
    ax2.axis("off")

    fig.tight_layout()
    fig.savefig("assets/07_gan.png", dpi=150)
    plt.close(fig)
    print("  Plot saved → assets/07_gan.png")
    return g_losses


if __name__ == "__main__":
    train()
