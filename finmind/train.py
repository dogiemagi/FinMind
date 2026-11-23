"""Phase 4/8: pretraining loop (next-token prediction) and shared training helpers."""
from __future__ import annotations

import math
import time

import torch


def get_batch(data: torch.Tensor, bs: int, block: int, g: torch.Generator):
    ix = torch.randint(0, len(data) - block - 1, (bs,), generator=g)
    x = torch.stack([data[i:i + block] for i in ix])
    y = torch.stack([data[i + 1:i + block + 1] for i in ix])      # target = input shifted by one
    return x, y


@torch.no_grad()
def eval_loss(model, data: torch.Tensor, bs: int = 16, block: int = 128, iters: int = 10) -> float:
    model.eval()
    g = torch.Generator().manual_seed(123)
    total = 0.0
    for _ in range(iters):
        x, y = get_batch(data, bs, block, g)
        total += model(x, y)[1].item()
    return total / iters


def pretrain(model, train_ids: list[int], val_ids: list[int], steps: int = 600, bs: int = 16,
             block: int = 128, lr: float = 3e-3, seed: int = 0, log=print) -> dict:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    tr, va = torch.tensor(train_ids), torch.tensor(val_ids)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    hist, t0 = [], time.time()
    for step in range(steps):
        model.train()
        x, y = get_batch(tr, bs, block, g)
        loss = model(x, y)[1]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == steps - 1:
            v = eval_loss(model, va, bs, block)
            hist.append({"step": step, "train": round(loss.item(), 3), "val": round(v, 3)})
            log(f"  step {step:4d}  train {loss.item():.3f}  val {v:.3f}  ({time.time() - t0:.0f}s)")
    final = hist[-1]["val"]
    return {"history": hist, "val_loss": final, "val_perplexity": round(math.exp(final), 2)}
