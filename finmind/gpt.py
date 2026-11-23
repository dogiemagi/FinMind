"""Phases 3-4 & 12: a GPT written from scratch.

Every component from the roadmap is explicit here:
token + positional embeddings, causal multi-head self-attention
    Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V
layer norm, residual connections, feed-forward network, weight tying, KV cache,
and an optional Mixture-of-Experts feed-forward layer (Phase 12).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 512
    block_size: int = 160      # max context window
    n_layer: int = 3
    n_head: int = 4
    n_embd: int = 96
    dropout: float = 0.1
    n_experts: int = 0         # >0 switches the FFN to Mixture-of-Experts
    top_k_experts: int = 2


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head, self.d = cfg.n_head, cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)   # Q, K, V projections in one matmul
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, past=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, self.d).transpose(1, 2) for t in (q, k, v))  # B,h,T,d
        if past is not None:                       # KV cache: reuse keys/values of earlier tokens
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        present = (k, v)
        S = k.size(2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d)          # QK^T / sqrt(d_k)
        causal = torch.ones(S, S, dtype=torch.bool, device=x.device).tril()[S - T:]
        scores = scores.masked_fill(~causal, float("-inf"))           # causal mask: no peeking ahead
        att = self.drop(F.softmax(scores, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), present


class MLPBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.fc2 = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class MoE(nn.Module):
    """Sparse Mixture of Experts: a router picks the top-k experts per token."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.router = nn.Linear(cfg.n_embd, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(MLPBlock(cfg) for _ in range(cfg.n_experts))
        self.k = min(cfg.top_k_experts, cfg.n_experts)
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x):
        B, T, C = x.shape
        flat = x.reshape(-1, C)
        logits = self.router(flat)
        top_w, top_i = logits.topk(self.k, dim=-1)
        top_w = F.softmax(top_w, dim=-1)
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            sel = top_i == e
            rows = sel.any(-1).nonzero(as_tuple=True)[0]
            if rows.numel():
                w = (top_w * sel)[rows].sum(-1, keepdim=True)
                out = out.index_add(0, rows, w * expert(flat[rows]))
        probs = F.softmax(logits, dim=-1).mean(0)                    # load-balancing loss
        load = F.one_hot(top_i, len(self.experts)).float().sum(1).mean(0) / self.k
        self.aux_loss = len(self.experts) * (probs * load).sum()
        return out.view(B, T, C)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.n_embd), nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ffn = MoE(cfg) if cfg.n_experts > 0 else MLPBlock(cfg)

    def forward(self, x, past=None):
        a, present = self.attn(self.ln1(x), past)
        x = x + a                       # residual connection
        x = x + self.ffn(self.ln2(x))   # residual connection
        return x, present


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)     # tokens -> vectors
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)     # learned positions
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight                      # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, past=None):
        B, T = idx.shape
        offset = past[0][0].size(2) if past else 0
        assert offset + T <= self.cfg.block_size, "context window exceeded"
        pos = torch.arange(offset, offset + T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        presents = []
        for i, blk in enumerate(self.blocks):
            x, pr = blk(x, past[i] if past else None)
            presents.append(pr)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
            if self.cfg.n_experts > 0:
                loss = loss + 0.01 * sum(b.ffn.aux_loss for b in self.blocks)
        return logits, loss, presents

    def config_dict(self) -> dict:
        return asdict(self.cfg)
