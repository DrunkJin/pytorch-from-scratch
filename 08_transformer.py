"""Transformer from Scratch — Sequence Classification (MNIST)

Treats each MNIST image as a sequence of 28 rows (28 timesteps, 28 features).
Classifies digits using multi-head self-attention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from common.utils import set_seed, plot_loss, get_device


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, T, -1)
        return self.W_o(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=28, seq_len=28, d_model=64, num_heads=4,
                 ff_dim=128, num_layers=2, num_classes=10, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [B, 1, 28, 28] → [B, 28, 28] (sequence of rows)
        x = x.squeeze(1)
        x = self.proj(x) + self.pos_embed
        x = self.blocks(x)
        x = self.ln(x)
        x = x.mean(dim=1)  # global average pooling
        return self.head(x)


def train(num_epochs=10, batch_size=128, lr=3e-4, seed=42):
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

    model = TransformerClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Transformer params: {num_params:,}")

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
        print(f"  [Transformer] Epoch {epoch+1}/{num_epochs}  Loss: {avg_loss:.4f}  Test Acc: {acc:.4f}")

    plot_loss(losses, "Transformer — MNIST (Sequence Classification)",
              "assets/08_transformer.png")
    return losses


if __name__ == "__main__":
    train()
