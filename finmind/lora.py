"""Phase 8: LoRA, SFT and DPO implemented from scratch.

LoRA freezes the pretrained weight W and learns a low-rank update:
    y = x W^T + (alpha / r) * x A^T B^T      with A: (r, in), B: (out, r), B starts at 0
Only A and B are trained (a few % of parameters). After training they can be merged
back into W so inference costs nothing extra.
"""
from __future__ import annotations

import copy
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.A.T @ self.B.T) * self.scale

    def merged(self) -> nn.Linear:
        lin = copy.deepcopy(self.base)
        lin.weight.data += (self.B @ self.A) * self.scale
        for p in lin.parameters():
            p.requires_grad = False
        return lin


def apply_lora(model, targets=("qkv", "proj"), r: int = 8, alpha: int = 16) -> dict:
    for p in model.parameters():
        p.requires_grad = False
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, r, alpha))
    return param_report(model)


def merge_lora(model) -> None:
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, name, child.merged())


def param_report(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": train, "trainable_pct": round(100 * train / total, 2)}


# ------------------------------------------------------------------ SFT
def build_example(tok, prompt: str, response: str, block: int):
    p = tok.encode(f"<|user|>{prompt}<|assistant|>")
    r = tok.encode(response) + [tok.eos_id]
    ids = (p + r)[:block]
    labels = ([-100] * len(p) + r)[:block]          # loss only on the assistant's answer
    return ids, labels


def _batch(examples, idx, pad):
    L = max(len(examples[i][0]) for i in idx)
    x = torch.full((len(idx), L), pad)
    y = torch.full((len(idx), L), -100)
    for row, i in enumerate(idx):
        ids, lab = examples[i]
        x[row, :len(ids)], y[row, :len(lab)] = torch.tensor(ids), torch.tensor(lab)
    return x, y[:, 1:], x[:, :-1]


@torch.no_grad()
def sft_loss(model, examples, pad) -> float:
    model.eval()
    tot = 0.0
    for k in range(0, len(examples), 16):
        x, y, xin = _batch(examples, list(range(k, min(k + 16, len(examples)))), pad)
        tot += model(xin, y)[1].item() * xin.size(0)
    return tot / len(examples)


def train_sft(model, examples, pad, epochs: int = 4, lr: float = 3e-3, bs: int = 8, seed: int = 0,
              log=print) -> list[float]:
    rng = random.Random(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    losses = []
    for ep in range(epochs):
        model.train()
        order = list(range(len(examples)))
        rng.shuffle(order)
        run = []
        for k in range(0, len(order), bs):
            x, y, xin = _batch(examples, order[k:k + bs], pad)
            loss = model(xin, y)[1]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            run.append(loss.item())
        losses.append(sum(run) / len(run))
        log(f"  sft epoch {ep + 1}/{epochs} loss {losses[-1]:.3f}")
    return losses


# ------------------------------------------------------------------ DPO
def seq_logp(model, ids: list[int], labels: list[int]) -> torch.Tensor:
    x = torch.tensor([ids[:-1]])
    y = torch.tensor(labels[1:])
    logp = F.log_softmax(model(x)[0][0], dim=-1)
    mask = y != -100
    return logp[torch.arange(len(y)), y.clamp(min=0)][mask].sum()


def dpo_loss(pol_c, pol_r, ref_c, ref_r, beta: float = 0.1):
    """-log sigmoid(beta * [(log pi(c)/pi_ref(c)) - (log pi(r)/pi_ref(r))])"""
    margin = beta * ((pol_c - ref_c) - (pol_r - ref_r))
    return -F.logsigmoid(margin), margin


def train_dpo(model, tok, pairs, block: int, epochs: int = 3, lr: float = 5e-4, beta: float = 0.1,
              log=print) -> dict:
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    data = [(build_example(tok, q, c, block), build_example(tok, q, r, block)) for q, c, r in pairs]
    acc = 0.0
    for ep in range(epochs):
        model.train()
        wins = 0
        for (cx, cl), (rx, rl) in data:
            with torch.no_grad():
                rc, rr = seq_logp(ref, cx, cl), seq_logp(ref, rx, rl)
            loss, margin = dpo_loss(seq_logp(model, cx, cl), seq_logp(model, rx, rl), rc, rr, beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            wins += int(margin.item() > 0)
        acc = wins / len(data)
        log(f"  dpo epoch {ep + 1}/{epochs} preference-accuracy {acc:.2f}")
    return {"preference_accuracy": round(acc, 3)}
