"""NLP fundamentals (Phases 1-2): tokenisation, stop words, stemming, vocab,
one-hot, Word2Vec (skip-gram with negative sampling) and cosine similarity."""
from __future__ import annotations

import math
import random
import re

import numpy as np

STOP_WORDS = set(
    "a an the and or of to in on for with by from at as is are was were be been it its this that "
    "these those which who what when where how do does did can may must will shall than then".split()
)


def sent_tokenize(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def word_tokenize(text: str) -> list[str]:
    return re.findall(r"\$?\d[\d,]*\.?\d*%?|[a-z]+(?:'[a-z]+)?", text.lower())


def stem(word: str) -> str:
    """Tiny suffix-stripping stemmer (Porter-lite): enough for retrieval."""
    for suf in ("ations", "ation", "ings", "ing", "ies", "ied", "es", "ed", "ly", "al", "er", "s", "e"):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)] + ("y" if suf in ("ies", "ied") else "")
    return word


def normalize(text: str, drop_stop: bool = True) -> list[str]:
    return [stem(w) for w in word_tokenize(text) if not (drop_stop and w in STOP_WORDS)]


class Vocab:
    def __init__(self, tokens: list[str], min_freq: int = 1):
        freq: dict[str, int] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        self.itos = ["<unk>"] + sorted(t for t, c in freq.items() if c >= min_freq)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(t, 0) for t in tokens]

    def one_hot(self, token: str) -> np.ndarray:
        v = np.zeros(len(self), dtype=np.float32)
        v[self.stoi.get(token, 0)] = 1.0
        return v

    def bag_of_words(self, tokens: list[str]) -> np.ndarray:
        v = np.zeros(len(self), dtype=np.float32)
        for t in tokens:
            v[self.stoi.get(t, 0)] += 1.0
        return v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d else 0.0


def train_word2vec(sentences: list[list[str]], dim: int = 32, window: int = 3, epochs: int = 40,
                   neg: int = 5, lr: float = 0.05, seed: int = 0):
    """Skip-gram with negative sampling in PyTorch. Words used in similar contexts
    end up with similar vectors -> semantic similarity is just geometry."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    rng = random.Random(seed)
    vocab = Vocab([w for s in sentences for w in s])
    pairs = []
    for s in sentences:
        ids = vocab.encode(s)
        for i, c in enumerate(ids):
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if j != i:
                    pairs.append((c, ids[j]))
    centers = torch.tensor([p[0] for p in pairs])
    contexts = torch.tensor([p[1] for p in pairs])
    emb_in, emb_out = nn.Embedding(len(vocab), dim), nn.Embedding(len(vocab), dim)
    opt = torch.optim.Adam(list(emb_in.parameters()) + list(emb_out.parameters()), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(pairs))
        for k in range(0, len(pairs), 512):
            b = perm[k:k + 512]
            c, o = emb_in(centers[b]), emb_out(contexts[b])
            n = emb_out(torch.randint(1, len(vocab), (len(b), neg)))
            pos = torch.nn.functional.logsigmoid((c * o).sum(-1))
            negs = torch.nn.functional.logsigmoid(-(n @ c.unsqueeze(-1)).squeeze(-1)).sum(-1)
            loss = -(pos + negs).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return vocab, emb_in.weight.detach().numpy()


def most_similar(word: str, vocab: Vocab, vectors: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
    if word not in vocab.stoi:
        return []
    v = vectors[vocab.stoi[word]]
    sims = [(w, cosine(v, vectors[i])) for w, i in vocab.stoi.items() if w not in (word, "<unk>")]
    return sorted(sims, key=lambda x: -x[1])[:k]


def tfidf_idf(docs: list[list[str]]) -> dict[str, float]:
    n = len(docs)
    df: dict[str, int] = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    return {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}
