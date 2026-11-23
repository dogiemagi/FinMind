"""Byte-level BPE tokenizer written from scratch (Phase 2).

Why tokens? A model needs a fixed vocabulary of integer ids. Words are too sparse
(every new invoice number would be unknown), bytes are too long. BPE starts from
256 bytes and repeatedly merges the most frequent adjacent pair into a new token,
so common strings ("payment", "Invoice") become single tokens while rare strings
("INV-2025-0042") fall back to small pieces. No text is ever out-of-vocabulary.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

SPECIALS = ["<|eos|>", "<|user|>", "<|assistant|>"]


def _pair_counts(ids: list[int]) -> Counter:
    return Counter(zip(ids, ids[1:]))


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special: dict[str, int] = {}

    # ------------------------------------------------------------ training
    def train(self, text: str, vocab_size: int = 512) -> "BPETokenizer":
        ids = list(text.encode("utf-8"))
        n_merges = vocab_size - 256 - len(SPECIALS)
        for i in range(n_merges):
            counts = _pair_counts(ids)
            if not counts:
                break
            pair, freq = counts.most_common(1)[0]
            if freq < 2:
                break
            new_id = 256 + i
            ids = _merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
        base = len(self.vocab)
        self.special = {s: base + j for j, s in enumerate(SPECIALS)}
        return self

    # ------------------------------------------------------------ encode/decode
    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special)

    @property
    def eos_id(self) -> int:
        return self.special["<|eos|>"]

    def _encode_chunk(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            counts = _pair_counts(ids)
            pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text: str) -> list[int]:
        pattern = "(" + "|".join(re.escape(s) for s in self.special) + ")"
        out: list[int] = []
        for part in re.split(pattern, text):
            if part in self.special:
                out.append(self.special[part])
            elif part:
                out.extend(self._encode_chunk(part))
        return out

    def decode(self, ids: list[int]) -> str:
        inv_special = {v: k for k, v in self.special.items()}
        buf = b""
        out = ""
        for i in ids:
            if i in inv_special:
                out += buf.decode("utf-8", errors="replace") + inv_special[i]
                buf = b""
            else:
                buf += self.vocab[i]
        return out + buf.decode("utf-8", errors="replace")

    def tokens(self, text: str) -> list[str]:
        """Human-readable pieces, for inspecting how a sentence is split."""
        return [self.vocab[i].decode("utf-8", errors="replace") if i in self.vocab else f"<{i}>"
                for i in self.encode(text)]

    # ------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        data = {"merges": [[a, b, i] for (a, b), i in self.merges.items()], "special": self.special}
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls()
        for a, b, i in data["merges"]:
            tok.merges[(a, b)] = i
            tok.vocab[i] = tok.vocab[a] + tok.vocab[b]
        tok.special = data["special"]
        return tok
