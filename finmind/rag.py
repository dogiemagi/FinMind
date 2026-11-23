"""Phase 7: Retrieval-Augmented Generation, built from parts.

ingest -> chunk (with overlap) -> embed -> vector store (+ BM25) -> hybrid retrieval
(RRF fusion) -> rerank -> context compression -> grounded, cited answer (or abstain).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .nlp import normalize, sent_tokenize, tfidf_idf


# --------------------------------------------------------------------- ingest
@dataclass
class Chunk:
    id: int
    doc: str
    text: str
    meta: dict = field(default_factory=dict)


def load_documents(folder: Path) -> list[tuple[str, str, dict]]:
    """txt / md / csv / pdf (pdf needs `pypdf`) -> (name, text, metadata)."""
    out = []
    for p in sorted(Path(folder).glob("*")):
        kind = p.suffix.lower()
        if kind in (".txt", ".md"):
            out.append((p.name, p.read_text(encoding="utf-8"), {"type": "policy" if "policy" in p.name or "terms" in p.name or "records" in p.name else "contract"}))
        elif kind == ".csv":
            with open(p, newline="", encoding="utf-8") as f:
                rows = ["; ".join(f"{k} is {v}" for k, v in r.items()) + "." for r in csv.DictReader(f)]
            out.append((p.name, "\n".join(rows), {"type": "table"}))
        elif kind == ".pdf":
            try:
                from pypdf import PdfReader
                out.append((p.name, "\n".join(pg.extract_text() or "" for pg in PdfReader(str(p)).pages), {"type": "pdf"}))
            except ImportError:
                pass
    return out


def chunk_text(text: str, max_words: int = 60, overlap_sents: int = 1) -> list[str]:
    """Sentence-aware chunking: pack sentences up to max_words, carry the last
    `overlap_sents` sentences into the next chunk so facts at boundaries are not lost."""
    sents, chunks, cur = sent_tokenize(text), [], []
    for s in sents:
        if cur and sum(len(x.split()) for x in cur) + len(s.split()) > max_words:
            chunks.append(" ".join(cur))
            cur = cur[-overlap_sents:] if overlap_sents else []
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


# ----------------------------------------------------------------- embeddings
class HashingEmbedder:
    """Deterministic TF-IDF embedder (hashing trick, unigrams + bigrams) that needs no
    model download. Swap in `SentenceTransformerEmbedder` for true semantic vectors."""

    def __init__(self, dim: int = 1024):
        self.dim, self.idf = dim, {}

    def fit(self, texts: list[str]):
        self.idf = tfidf_idf([normalize(t) for t in texts])
        return self

    def _feats(self, text: str):
        toks = normalize(text)
        yield from toks
        yield from (f"{a}_{b}" for a, b in zip(toks, toks[1:]))

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for f, c in Counter(self._feats(text)).items():
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            sign = 1.0 if (h >> 64) & 1 else -1.0
            v[h % self.dim] += sign * (1 + math.log(c)) * self.idf.get(f.split("_")[0], 1.0)
        n = np.linalg.norm(v)
        return v / n if n else v


class SentenceTransformerEmbedder:
    def __init__(self, name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(name)

    def fit(self, texts):
        return self

    def embed(self, text: str) -> np.ndarray:
        return self.m.encode(text, normalize_embeddings=True).astype(np.float32)


# ------------------------------------------------------------------- indexes
class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.avg = sum(map(len, docs)) / max(len(docs), 1)
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for i, (tf, d) in enumerate(zip(self.tf, self.docs)):
            for t in query:
                if t in tf:
                    f = tf[t]
                    out[i] += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avg))
        return out


class LexicalReranker:
    """Second-stage scorer over the shortlist: idf-weighted query coverage + bigram
    (phrase) bonus. A cross-encoder (e.g. bge-reranker) is the drop-in upgrade."""

    def __init__(self, idf: dict[str, float]):
        self.idf = idf

    def score(self, query: str, text: str) -> float:
        q, d = normalize(query), normalize(text)
        if not q:
            return 0.0
        dset = set(d)
        cover = sum(self.idf.get(t, 1.0) for t in q if t in dset) / sum(self.idf.get(t, 1.0) for t in q)
        qb, db = set(zip(q, q[1:])), set(zip(d, d[1:]))
        phrase = len(qb & db) / max(len(qb), 1)
        return cover + 0.5 * phrase


def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal rank fusion: robust way to merge dense and keyword rankings."""
    fused: dict[int, float] = {}
    for r in rankings:
        for rank, i in enumerate(r):
            fused[i] = fused.get(i, 0.0) + 1.0 / (k + rank + 1)
    return fused


# ----------------------------------------------------------------- the index
@dataclass
class Hit:
    chunk: Chunk
    score: float


class KnowledgeBase:
    def __init__(self, embedder=None):
        self.embedder = embedder or HashingEmbedder()
        self.chunks: list[Chunk] = []

    def build(self, docs: list[tuple[str, str, dict]], max_words: int = 60, overlap_sents: int = 1):
        self.chunks = []
        for name, text, meta in docs:
            for t in chunk_text(text, max_words, overlap_sents):
                self.chunks.append(Chunk(len(self.chunks), name, t, dict(meta)))
        texts = [c.text for c in self.chunks]
        self.embedder.fit(texts)
        self.matrix = np.stack([self.embedder.embed(t) for t in texts])
        norm = [normalize(t) for t in texts]
        self.bm25 = BM25(norm)
        self.reranker = LexicalReranker(tfidf_idf(norm))
        return self

    def _eligible(self, filters: dict | None) -> np.ndarray:
        if not filters:
            return np.ones(len(self.chunks), dtype=bool)
        return np.array([all(c.meta.get(k) == v or (k == "doc" and c.doc == v) for k, v in filters.items())
                         for c in self.chunks])

    def retrieve(self, query: str, k: int = 3, mode: str = "hybrid", rerank: bool = True,
                 filters: dict | None = None, pool: int = 8) -> list[Hit]:
        ok = self._eligible(filters)
        dense = self.matrix @ self.embedder.embed(query)
        sparse = self.bm25.scores(normalize(query))
        dense[~ok], sparse[~ok] = -1e9, -1e9
        d_rank = [int(i) for i in np.argsort(-dense)[:pool] if ok[i]]
        s_rank = [int(i) for i in np.argsort(-sparse)[:pool] if ok[i] and sparse[i] > 0]
        if mode == "dense":
            cand = {i: float(dense[i]) for i in d_rank}
        elif mode == "bm25":
            cand = {i: float(sparse[i]) for i in s_rank}
        else:
            cand = rrf([d_rank, s_rank])
        ids = sorted(cand, key=cand.get, reverse=True)[:pool]
        if rerank:
            scored = [(i, self.reranker.score(query, self.chunks[i].text)) for i in ids]
            ids = [i for i, _ in sorted(scored, key=lambda x: -x[1])]
            cand = dict(scored)
        return [Hit(self.chunks[i], float(cand[i])) for i in ids[:k]]

    def save(self, path: Path):
        Path(path).write_text(json.dumps([{"id": c.id, "doc": c.doc, "text": c.text, "meta": c.meta}
                                          for c in self.chunks]), encoding="utf-8")


def compress(query: str, hits: list[Hit], max_sents: int = 4) -> list[tuple[Hit, str]]:
    """Context compression: keep only the sentences that overlap most with the query."""
    q = set(normalize(query))
    scored = []
    for h in hits:
        for s in sent_tokenize(h.chunk.text):
            scored.append((len(q & set(normalize(s))), h, s))
    best = sorted(scored, key=lambda x: -x[0])[:max_sents]
    return [(h, s) for sc, h, s in best if sc > 0] or [(hits[0], hits[0].chunk.text)] if hits else []


# ------------------------------------------------------------------- answer
@dataclass
class RAGAnswer:
    question: str
    text: str
    citations: list[dict]
    contexts: list[str]
    grounded: bool
    abstained: bool = False


RAG_SYSTEM = ("You answer finance-policy questions using ONLY the numbered context. Cite sources like [1]. "
              "If the context does not contain the answer, reply exactly: Not found in the documents.")


class RAGPipeline:
    def __init__(self, kb: KnowledgeBase, router, min_score: float = 0.35):
        self.kb, self.router, self.min_score = kb, router, min_score

    def answer(self, question: str, k: int = 3, mode: str = "hybrid", rerank: bool = True,
               filters: dict | None = None, model_hint: str | None = None) -> RAGAnswer:
        hits = self.kb.retrieve(question, k=k, mode=mode, rerank=rerank, filters=filters)
        rr = self.kb.reranker
        best = max((rr.score(question, h.chunk.text) for h in hits), default=0.0)
        if not hits or best < self.min_score:         # abstain instead of guessing
            return RAGAnswer(question, "Not found in the documents.", [], [], True, True)
        picked = compress(question, hits)
        contexts = [f"[{i + 1}] ({h.chunk.doc}) {s}" for i, (h, s) in enumerate(picked)]
        prompt = "Context:\n" + "\n".join(contexts) + f"\n\nQuestion: {question}"
        resp = self.router.complete([{"role": "user", "content": prompt}], system=RAG_SYSTEM)
        cites = [{"n": i + 1, "doc": h.chunk.doc, "text": s} for i, (h, s) in enumerate(picked)]
        grounded = bool(re.search(r"\[\d+\]", resp.text)) or "not found" in resp.text.lower()
        return RAGAnswer(question, resp.text, cites, [c["text"] for c in cites], grounded)
