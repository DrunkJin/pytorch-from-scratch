"""RNN / LSTM from Scratch — Character-Level Text Generation

Trains on a small text corpus and generates new text character by character.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.utils import set_seed, plot_loss, get_device

# Training corpus
CORPUS = """To be or not to be that is the question.
Whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune.
Or to take arms against a sea of troubles and by opposing end them.
To die to sleep no more and by a sleep to say we end the heartache.
The thousand natural shocks that flesh is heir to.
Tis a consummation devoutly to be wished.
To die to sleep to sleep perchance to dream.
Ay there is the rub for in that sleep of death what dreams may come.
"""


class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden


def train(num_epochs=200, seq_len=50, lr=3e-3, seed=42):
    set_seed(seed)
    device = get_device()

    # Build vocabulary
    chars = sorted(set(CORPUS))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    vocab_size = len(chars)
    data = torch.LongTensor([char2idx[c] for c in CORPUS])
    print(f"  Corpus: {len(CORPUS)} chars, vocab: {vocab_size} unique chars")

    model = CharLSTM(vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        count = 0
        for i in range(0, len(data) - seq_len - 1, seq_len):
            x = data[i:i + seq_len].unsqueeze(0).to(device)
            y = data[i + 1:i + seq_len + 1].unsqueeze(0).to(device)

            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            count += 1

        avg_loss = epoch_loss / max(count, 1)
        losses.append(avg_loss)

        if (epoch + 1) % 50 == 0:
            print(f"  [RNN] Epoch {epoch+1}/{num_epochs}  Loss: {avg_loss:.4f}")

    # Generate sample text
    model.eval()
    generated = "To "
    x = torch.LongTensor([[char2idx[c] for c in generated]]).to(device)
    hidden = None
    with torch.no_grad():
        for _ in range(150):
            logits, hidden = model(x, hidden)
            probs = F.softmax(logits[0, -1] / 0.8, dim=0)
            idx = torch.multinomial(probs, 1).item()
            generated += idx2char[idx]
            x = torch.LongTensor([[idx]]).to(device)
    print(f"  Generated: {generated}")

    plot_loss(losses, "Char-LSTM — Text Generation", "assets/05_rnn.png")
    return losses


if __name__ == "__main__":
    train()
