"""Phase 12: research techniques, implemented small enough to read.

* tiled_attention  - FlashAttention's online-softmax trick (never builds the full TxT matrix)
* distill_loss     - knowledge distillation (student matches teacher's softened distribution)
* self_consistency - test-time compute: sample N answers, majority vote
(MoE lives in gpt.py, speculative decoding in sampling.py, synthetic data in data.py.)"""
from __future__ import annotations

import math
from collections import Counter

import torch
import torch.nn.functional as F


def tiled_attention(q, k, v, block: int = 16, causal: bool = True):
    """q,k,v: (T, d). Processes keys/values in tiles, keeping a running max and
    running denominator so the result equals softmax(QK^T/sqrt(d))V exactly."""
    T, d = q.shape
    out = torch.zeros_like(q)
    m = torch.full((T, 1), float("-inf"))
    l = torch.zeros(T, 1)
    for s in range(0, T, block):
        ks, vs = k[s:s + block], v[s:s + block]
        scores = q @ ks.T / math.sqrt(d)
        if causal:
            qi = torch.arange(T).unsqueeze(1)
            kj = torch.arange(s, s + ks.size(0)).unsqueeze(0)
            scores = scores.masked_fill(kj > qi, float("-inf"))
        m_new = torch.maximum(m, scores.amax(dim=1, keepdim=True))
        p = torch.exp(scores - m_new)
        scale = torch.exp(m - m_new).nan_to_num(0.0)
        l = l * scale + p.sum(dim=1, keepdim=True)
        out = out * scale + p @ vs
        m = m_new
    return out / l


def distill_loss(student_logits, teacher_logits, labels=None, T: float = 2.0, alpha: float = 0.5):
    soft = F.kl_div(F.log_softmax(student_logits / T, -1), F.softmax(teacher_logits / T, -1),
                    reduction="batchmean") * T * T
    if labels is None:
        return soft
    hard = F.cross_entropy(student_logits.reshape(-1, student_logits.size(-1)), labels.reshape(-1), ignore_index=-100)
    return alpha * soft + (1 - alpha) * hard


def self_consistency(sample_fn, n: int = 5) -> tuple[str, float]:
    votes = Counter(sample_fn() for _ in range(n))
    ans, c = votes.most_common(1)[0]
    return ans, c / n
