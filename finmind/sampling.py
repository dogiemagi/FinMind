"""Phase 4 & 12: decoding. Greedy, temperature, top-k, top-p, beam search,
KV-cached generation and speculative decoding -- all written by hand."""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F


def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None,
                top_p: float | None = None, greedy: bool = False,
                generator: torch.Generator | None = None) -> int:
    """logits: (vocab,) -> sampled token id."""
    if greedy or temperature <= 0:
        return int(logits.argmax())
    logits = logits / temperature                       # <1 sharper, >1 flatter
    if top_k:                                           # keep only the k most likely tokens
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p and top_p < 1.0:                           # nucleus: smallest set with mass >= p
        sorted_logits, order = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(-1)
        remove = cum - probs > top_p                    # drop tokens once mass already exceeds p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(0, order, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))


@torch.no_grad()
def generate(model, ids: list[int], max_new: int = 50, use_cache: bool = True, eos: int | None = None,
             stats: dict | None = None, **sampling) -> list[int]:
    """Autoregressive generation. With use_cache the model only processes the newest
    token each step and reuses cached keys/values (O(n) instead of O(n^2) per step)."""
    model.eval()
    block = model.cfg.block_size
    ctx, past, past_len = list(ids), None, 0
    t0 = time.perf_counter()
    for _ in range(max_new):
        if use_cache and past is not None and past_len < block:
            inp = torch.tensor([[ctx[-1]]])
        else:
            inp, past = torch.tensor([ctx[-block:]]), None
        logits, _, presents = model(inp, past=past)
        past = presents if use_cache else None
        past_len = presents[0][0].size(2)
        nxt = sample_next(logits[0, -1], **sampling)
        ctx.append(nxt)
        if eos is not None and nxt == eos:
            break
    if stats is not None:
        n = len(ctx) - len(ids)
        stats.update(new_tokens=n, seconds=time.perf_counter() - t0,
                     tokens_per_sec=n / max(time.perf_counter() - t0, 1e-9))
    return ctx


@torch.no_grad()
def beam_search(model, ids: list[int], beams: int = 3, max_new: int = 30, eos: int | None = None) -> list[int]:
    model.eval()
    block = model.cfg.block_size
    hyps = [(0.0, list(ids))]
    for _ in range(max_new):
        cand = []
        for score, seq in hyps:
            if eos is not None and seq[-1] == eos:
                cand.append((score, seq))
                continue
            logits, _, _ = model(torch.tensor([seq[-block:]]))
            logp = F.log_softmax(logits[0, -1], dim=-1)
            top = torch.topk(logp, beams)
            for lp, tok in zip(top.values.tolist(), top.indices.tolist()):
                cand.append((score + lp, seq + [tok]))
        hyps = sorted(cand, key=lambda h: h[0] / (len(h[1]) - len(ids) + 1), reverse=True)[:beams]
        if eos is not None and all(h[1][-1] == eos for h in hyps):
            break
    return hyps[0][1]


@torch.no_grad()
def speculative_decode(target, draft, ids: list[int], max_new: int = 40, k: int = 4,
                       stats: dict | None = None) -> list[int]:
    """Greedy speculative decoding: a cheap draft model proposes k tokens, the target
    model verifies them all in ONE forward pass. Output is identical to greedy decoding
    with the target alone, but needs fewer target passes when the draft is often right."""
    target.eval()
    draft.eval()
    block = min(target.cfg.block_size, draft.cfg.block_size)
    ctx = list(ids)
    goal = len(ids) + max_new
    passes = accepted = proposed = 0
    while len(ctx) < goal and len(ctx) + k + 1 <= block:
        props = []
        for _ in range(k):                                      # draft proposes greedily
            lg, _, _ = draft(torch.tensor([ctx + props]))
            props.append(int(lg[0, -1].argmax()))
        lg, _, _ = target(torch.tensor([ctx + props]))          # one target pass verifies all
        passes += 1
        preds = lg[0, len(ctx) - 1:].argmax(-1).tolist()        # target's choice at each position
        n_ok = 0
        while n_ok < k and props[n_ok] == preds[n_ok]:
            n_ok += 1
        ctx += props[:n_ok] + [preds[n_ok]]                     # accepted tokens + 1 free token
        proposed += k
        accepted += n_ok
    if stats is not None:
        stats.update(target_passes=passes, accepted=accepted, proposed=proposed,
                     acceptance_rate=accepted / max(proposed, 1))
    return ctx[:goal]
