"""Phase 10: evaluation & reliability.

Answer metrics (exact match, token F1, ROUGE-L, BLEU-1, semantic similarity), RAG metrics
(context precision / recall, hit-rate, MRR, faithfulness, answer relevancy, abstention),
LLM-as-a-judge, config comparison and a regression gate for CI."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .data import QA, UNANSWERABLE
from .nlp import normalize, sent_tokenize, word_tokenize


# ------------------------------------------------------------ answer metrics
def exact_match(pred: str, ref: str) -> float:
    return float(" ".join(word_tokenize(pred)) == " ".join(word_tokenize(ref)))


def precision_recall_f1(pred: str, ref: str) -> tuple[float, float, float]:
    p, r = Counter(word_tokenize(pred)), Counter(word_tokenize(ref))
    overlap = sum((p & r).values())
    if not overlap:
        return 0.0, 0.0, 0.0
    prec, rec = overlap / sum(p.values()), overlap / sum(r.values())
    return prec, rec, 2 * prec * rec / (prec + rec)


def rouge_l(pred: str, ref: str) -> float:
    a, b = word_tokenize(pred), word_tokenize(ref)
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[-1][-1]
    if not lcs:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r)


def bleu1(pred: str, ref: str) -> float:
    p, r = word_tokenize(pred), Counter(word_tokenize(ref))
    if not p:
        return 0.0
    clipped = sum(min(c, r[w]) for w, c in Counter(p).items())
    bp = 1.0 if len(p) >= len(r) else math.exp(1 - sum(r.values()) / len(p))
    return bp * clipped / len(p)


def semantic_similarity(embedder, a: str, b: str) -> float:
    return float(embedder.embed(a) @ embedder.embed(b))


def perplexity(loss: float) -> float:
    return math.exp(loss)


# --------------------------------------------------------------- RAG metrics
def context_recall(contexts: list[str], gold_keywords: list[str]) -> float:
    blob = " ".join(contexts).lower()
    return sum(k.lower() in blob for k in gold_keywords) / len(gold_keywords)


def context_precision(contexts: list[str], gold_keywords: list[str]) -> float:
    if not contexts:
        return 0.0
    return sum(any(k.lower() in c.lower() for k in gold_keywords) for c in contexts) / len(contexts)


def faithfulness(answer: str, contexts: list[str]) -> float:
    """Share of answer sentences whose content words are supported by the context."""
    ctx = set(normalize(" ".join(contexts)))
    sents = [re.sub(r"\[\d+\]", "", s) for s in sent_tokenize(answer)]
    scores = []
    for s in sents:
        w = normalize(s)
        if w:
            scores.append(sum(t in ctx for t in w) / len(w))
    return sum(scores) / len(scores) if scores else 0.0


def answer_relevancy(embedder, question: str, answer: str) -> float:
    return max(0.0, semantic_similarity(embedder, question, answer))


def llm_judge(router, question: str, answer: str, reference: str) -> dict:
    """LLM-as-a-judge: 1-5 score against a reference with a rubric. With a real model the
    judge should be *stronger than* the model being judged."""
    sys = ('Rate the answer 1-5 for correctness vs the reference and grounding. '
           'Reply JSON {"score": int, "reason": str}')
    user = f"Question: {question}\nReference: {reference}\nAnswer: {answer}"
    try:
        out = json.loads(re.search(r"\{.*\}", router.complete([{"role": "user", "content": user}], sys).text, re.S).group(0))
        return {"score": int(out["score"]), "reason": out.get("reason", "")}
    except Exception:  # noqa: BLE001
        return {"score": 0, "reason": "judge failed"}


# ------------------------------------------------------------- run an eval
def evaluate_rag(pipeline, judge_router=None, **cfg) -> dict:
    rows = []
    for q, ref, gold in QA:
        a = pipeline.answer(q, **cfg)
        hits = pipeline.kb.retrieve(q, k=cfg.get("k", 3), mode=cfg.get("mode", "hybrid"), rerank=cfg.get("rerank", True))
        ctx = [h.chunk.text for h in hits]
        rank = next((i + 1 for i, h in enumerate(hits) if any(k.lower() in h.chunk.text.lower() for k in gold)), None)
        rows.append({
            "em": exact_match(a.text, ref), "f1": precision_recall_f1(a.text, ref)[2], "rouge_l": rouge_l(a.text, ref),
            "ctx_recall": context_recall(ctx, gold), "ctx_precision": context_precision(ctx, gold),
            "hit": float(rank is not None), "mrr": 1 / rank if rank else 0.0,
            "faithfulness": faithfulness(a.text, a.contexts or ctx),
            "relevancy": answer_relevancy(pipeline.kb.embedder, q, a.text),
            "correct": float(all(k.lower() in a.text.lower() for k in gold)),
            "judge": llm_judge(judge_router, q, a.text, ref)["score"] if judge_router else 0,
        })
    abstain = np.mean([pipeline.answer(q, **cfg).abstained for q in UNANSWERABLE])
    out = {k: round(float(np.mean([r[k] for r in rows])), 3) for k in rows[0]}
    out["abstain_on_unanswerable"] = float(abstain)
    return out


def compare_configs(pipeline, judge_router=None) -> dict:
    configs = {
        "bm25": dict(mode="bm25", rerank=False), "dense": dict(mode="dense", rerank=False),
        "hybrid": dict(mode="hybrid", rerank=False), "hybrid+rerank": dict(mode="hybrid", rerank=True),
    }
    return {name: evaluate_rag(pipeline, judge_router, **cfg) for name, cfg in configs.items()}


def regression_gate(current: dict, baseline_path: Path, tol: float = 0.05,
                    keys=("correct", "ctx_recall", "faithfulness")) -> tuple[bool, list[str]]:
    """Fail CI if a key metric drops by more than `tol` versus the stored baseline."""
    baseline_path = Path(baseline_path)
    if not baseline_path.exists():
        baseline_path.write_text(json.dumps(current, indent=2))
        return True, ["baseline created"]
    base = json.loads(baseline_path.read_text())
    problems = [f"{k}: {base[k]} -> {current[k]}" for k in keys if current[k] < base[k] - tol]
    return not problems, problems


def to_markdown(table: dict) -> str:
    cols = ["correct", "ctx_recall", "ctx_precision", "mrr", "faithfulness", "relevancy", "f1", "judge"]
    lines = ["| config | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for name, m in table.items():
        lines.append(f"| {name} | " + " | ".join(str(m[c]) for c in cols) + " |")
    return "\n".join(lines)
