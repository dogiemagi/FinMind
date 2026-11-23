"""Phase 11: quantization + benchmarking, written from scratch.

INT8 (per-output-channel symmetric) and INT4 (group-wise) weight quantization with
dequantize-on-the-fly Linear layers. Production stacks (AWQ/GPTQ + vLLM) do the same
idea with fused GPU kernels; here we measure the memory/quality trade-off honestly."""
from __future__ import annotations

import copy
import json
import statistics
import threading
import time
import urllib.request

import torch
import torch.nn as nn


def quantize_tensor(w: torch.Tensor, bits: int = 8, group: int = 32):
    qmax = 2 ** (bits - 1) - 1
    if bits == 8:                                   # one scale per output row
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
        q = torch.round(w / scale).clamp(-qmax - 1, qmax).to(torch.int8)
        return q, scale
    out, inp = w.shape                              # int4: one scale per group of `group` inputs
    pad = (-inp) % group
    wp = torch.nn.functional.pad(w, (0, pad)).view(out, -1, group)
    scale = wp.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / qmax
    q = torch.round(wp / scale).clamp(-qmax - 1, qmax).to(torch.int8)
    return q, scale


class QuantLinear(nn.Module):
    def __init__(self, lin: nn.Linear, bits: int = 8, group: int = 32):
        super().__init__()
        self.bits, self.group = bits, group
        self.in_f, self.out_f = lin.in_features, lin.out_features
        q, s = quantize_tensor(lin.weight.data, bits, group)
        self.register_buffer("q", q)
        self.register_buffer("scale", s)
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.data.clone(), requires_grad=False)

    def dequant(self) -> torch.Tensor:
        w = self.q.float() * self.scale
        return w.reshape(self.out_f, -1)[:, :self.in_f]

    def forward(self, x):
        return nn.functional.linear(x, self.dequant(), self.bias)

    def nbytes(self) -> float:
        bits = self.bits
        return self.q.numel() * bits / 8 + self.scale.numel() * 4


def quantize_model(model, bits: int = 8):
    m = copy.deepcopy(model)
    for module in list(m.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child is not getattr(m, "head", None):
                setattr(module, name, QuantLinear(child, bits))
    return m


def model_bytes(model) -> float:
    total, seen = 0.0, set()
    for mod in model.modules():
        if isinstance(mod, QuantLinear):
            total += mod.nbytes() + (mod.bias.numel() * 4 if mod.bias is not None else 0)
            seen.update(id(p) for p in mod.parameters())
    for p in model.parameters():
        if id(p) not in seen:
            total += p.numel() * 4
    return total


def benchmark_generation(model, ids: list[int], new_tokens: int = 40, runs: int = 3) -> dict:
    from .sampling import generate
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        generate(model, ids, max_new=new_tokens, greedy=True)
        times.append(time.perf_counter() - t)
    med = statistics.median(times)
    return {"median_s": round(med, 4), "tokens_per_s": round(new_tokens / med, 1)}


def load_test(url: str, payload: dict, n: int = 40, concurrency: int = 8) -> dict:
    """Throughput / latency benchmark for any HTTP endpoint (FastAPI app or vLLM server)."""
    lat, errors, lock = [], 0, threading.Lock()
    todo = list(range(n))

    def worker():
        nonlocal errors
        while True:
            with lock:
                if not todo:
                    return
                todo.pop()
            t = time.perf_counter()
            try:
                req = urllib.request.Request(url, json.dumps(payload).encode(), {"content-type": "application/json"})
                urllib.request.urlopen(req, timeout=60).read()
                dt = time.perf_counter() - t
                with lock:
                    lat.append(dt)
            except Exception:  # noqa: BLE001
                with lock:
                    errors += 1

    t0 = time.perf_counter()
    ts = [threading.Thread(target=worker) for _ in range(concurrency)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    wall = time.perf_counter() - t0
    lat.sort()
    q = lambda p: round(lat[min(len(lat) - 1, int(p * len(lat)))] * 1000, 1) if lat else None
    return {"requests": n, "errors": errors, "rps": round(len(lat) / wall, 1), "p50_ms": q(0.5), "p95_ms": q(0.95)}
