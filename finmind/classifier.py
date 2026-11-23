"""Phase 1: a neural network from first principles in PyTorch.

Classifies invoice descriptions into expense categories. Shows tensors, autograd,
loss, optimiser, regularisation (dropout + weight decay), validation split and
early stopping -- and is compared against a majority-class baseline."""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import CATEGORIES, make_invoices
from .nlp import Vocab, normalize


class MLP(nn.Module):
    def __init__(self, n_in: int, n_hidden: int, n_out: int, p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, n_hidden), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(n_hidden, n_out))

    def forward(self, x):
        return self.net(x)


def train_classifier(n: int = 600, epochs: int = 60, lr: float = 5e-3, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    rows = make_invoices(n, seed=11)
    random.Random(seed).shuffle(rows)
    docs = [normalize(r["description"]) for r in rows]
    vocab = Vocab([t for d in docs for t in d])
    X = torch.tensor(np.stack([vocab.bag_of_words(d) for d in docs]))
    y = torch.tensor([CATEGORIES.index(r["category"]) for r in rows])
    split = int(0.8 * n)
    Xtr, ytr, Xva, yva = X[:split], y[:split], X[split:], y[split:]

    model = MLP(len(vocab), 32, len(CATEGORIES))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    best, best_state, history = 1e9, None, []
    for ep in range(epochs):
        model.train()
        loss = F.cross_entropy(model(Xtr), ytr)
        opt.zero_grad()
        loss.backward()  # autograd computes d(loss)/d(every weight)
        opt.step()       # gradient descent update
        model.eval()
        with torch.no_grad():
            val = F.cross_entropy(model(Xva), yva).item()
        history.append((loss.item(), val))
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    with torch.no_grad():
        acc = (model(Xva).argmax(-1) == yva).float().mean().item()
    majority = (yva == torch.mode(ytr).values).float().mean().item()
    return {"val_accuracy": round(acc, 3), "majority_baseline": round(majority, 3),
            "final_train_loss": round(history[-1][0], 4), "best_val_loss": round(best, 4),
            "vocab_size": len(vocab)}
