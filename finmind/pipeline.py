"""Orchestration: train everything end-to-end (`train_all`) and load the running system
(`load_system`). Each step maps to one phase of the roadmap and records a metric."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from . import data as D
from .agent import FinanceAgent, Ledger
from .classifier import train_classifier
from .gpt import GPT, GPTConfig
from .llm import LocalGPTProvider, Router
from .lora import apply_lora, build_example, merge_lora, param_report, sft_loss, train_dpo, train_sft
from .nlp import normalize, most_similar, train_word2vec, sent_tokenize
from .quant import benchmark_generation, model_bytes, quantize_model
from .rag import KnowledgeBase, RAGPipeline, load_documents
from .sampling import generate, speculative_decode
from .tokenizer import BPETokenizer
from .train import eval_loss, pretrain

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"


def _ckpt(model, path):
    torch.save({"cfg": model.config_dict(), "state": model.state_dict()}, path)


def _load_model(path) -> GPT:
    c = torch.load(path, weights_only=False)
    m = GPT(GPTConfig(**c["cfg"]))
    m.load_state_dict(c["state"])
    return m.eval()


def _distill(student, teacher, ids, steps=400, bs=16, block=96, lr=3e-3):
    """Train a small draft model to match the target's output distribution (Phase 12)."""
    from .research import distill_loss
    from .train import get_batch
    teacher.eval()
    g = torch.Generator().manual_seed(0)
    data = torch.tensor(ids)
    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    for _ in range(steps):
        x, y = get_batch(data, bs, block, g)
        with torch.no_grad():
            t_logits = teacher(x)[0]
        loss = distill_loss(student(x)[0], t_logits, T=1.0, alpha=1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
    student.eval()


def train_all(art: Path = ART, pretrain_steps: int = 250, log=print) -> dict:
    art.mkdir(parents=True, exist_ok=True)
    M: dict = {}
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    log("[Phase 1] data + neural network from scratch")
    docs_dir = D.write_documents(art / "data")
    invoices = D.create_ledger(art / "ledger.db")
    M["phase1_classifier"] = train_classifier()
    log("  ", M["phase1_classifier"])

    log("[Phase 2] NLP: BPE tokenizer + Word2Vec")
    corpus = D.pretrain_corpus(invoices)
    tok = BPETokenizer().train(corpus, vocab_size=420)
    tok.save(art / "tokenizer.json")
    sample = "Invoice INV-2025-0042 from Acme Cloud for $1,234.50"
    M["phase2_tokenizer"] = {"vocab_size": tok.vocab_size, "bytes_per_token": round(len(corpus.encode()) / len(tok.encode(corpus)), 2),
                             "sample_tokens": tok.tokens(sample), "roundtrip_ok": tok.decode(tok.encode(sample)) == sample}
    sents = [normalize(s, drop_stop=False) for s in sent_tokenize(corpus)]
    vocab, vecs = train_word2vec(sents[:400], dim=24, epochs=15)
    M["phase2_word2vec"] = {"similar_to_invoice": most_similar("invoice", vocab, vecs, 3)}
    log("  ", M["phase2_tokenizer"]["bytes_per_token"], "bytes/token;", M["phase2_tokenizer"]["sample_tokens"])

    log("[Phase 3-4] pretrain mini-GPT (next-token prediction)")
    ids = tok.encode(corpus)
    cut = int(len(ids) * 0.9)
    cfg = GPTConfig(vocab_size=tok.vocab_size)
    model = GPT(cfg)
    M["phase3_model"] = {"params": model.n_params(), **model.config_dict()}
    M["phase4_pretrain"] = pretrain(model, ids[:cut], ids[cut:], steps=pretrain_steps, log=log)

    log("[Phase 4] decoding + KV cache")
    prompt = tok.encode("Payment Approval Policy.")
    s_c, s_n = {}, {}
    generate(model, prompt, 100, use_cache=True, greedy=True, stats=s_c)
    generate(model, prompt, 100, use_cache=False, greedy=True, stats=s_n)
    M["phase4_kv_cache"] = {"with_cache_tok_s": round(s_c["tokens_per_sec"], 1), "no_cache_tok_s": round(s_n["tokens_per_sec"], 1)}
    log("  ", M["phase4_kv_cache"])

    log("[Phase 8] SFT with LoRA, then DPO")
    train_ex, held_ex = D.sft_examples(invoices)
    enc = lambda xs: [build_example(tok, p, r, cfg.block_size) for p, r in xs]
    tr, ho = enc(train_ex), enc(held_ex)
    pad = tok.eos_id
    base_loss = sft_loss(model, ho, pad)
    M["phase8_lora"] = apply_lora(model, r=16, alpha=32)
    train_sft(model, tr, pad, epochs=25, lr=5e-3, log=lambda m: log(m) if "epoch 25" in m or "epoch 1/" in m else None)
    tuned_loss = sft_loss(model, ho, pad)
    M["phase8_sft"] = {"heldout_loss_base": round(base_loss, 3), "heldout_loss_tuned": round(tuned_loss, 3)}
    M["phase8_dpo"] = train_dpo(model, tok, D.dpo_pairs(), cfg.block_size, epochs=3, log=log)
    probe = tok.encode("<|user|>" + held_ex[0][0] + "<|assistant|>")
    before = model(torch.tensor([probe]))[0]
    merge_lora(model)                                  # fold adapters into the base weights
    M["phase8_merge"] = {"max_logit_diff": float((model(torch.tensor([probe]))[0] - before).abs().max().detach()), **param_report(model)}
    M["phase8_sample_output"] = tok.decode(generate(model, probe, 60, greedy=True, eos=tok.eos_id)[len(probe):])
    _ckpt(model, art / "finmind_gpt.pt")
    log("  ", M["phase8_sft"], M["phase8_dpo"])

    log("[Phase 11] quantization")
    val = torch.tensor(ids[cut:])
    fp = eval_loss(model, val)
    q = {}
    for bits in (8, 4):
        qm = quantize_model(model, bits)
        q[f"int{bits}"] = {"val_loss": round(eval_loss(qm, val), 4), "mb": round(model_bytes(qm) / 1e6, 3),
                           **benchmark_generation(qm, prompt, 30, 2)}
    M["phase11_quant"] = {"fp32": {"val_loss": round(fp, 4), "mb": round(model_bytes(model) / 1e6, 3),
                                   **benchmark_generation(model, prompt, 30, 2)}, **q}
    log("  ", {k: (v["val_loss"], v["mb"]) for k, v in M["phase11_quant"].items()})

    log("[Phase 12] speculative decoding with a smaller draft model")
    draft = GPT(GPTConfig(vocab_size=tok.vocab_size, n_layer=1, n_embd=48, n_head=2, dropout=0.0))
    _distill(draft, model, ids[:cut], steps=400)       # knowledge distillation from the target
    st = {}
    spec = speculative_decode(model, draft, prompt, 40, k=4, stats=st)
    ref = generate(model, prompt, 40, use_cache=False, greedy=True)
    M["phase12_speculative"] = {**{k: round(v, 3) for k, v in st.items()}, "identical_to_greedy": spec == ref[:len(spec)]}
    log("  ", M["phase12_speculative"])

    (art / "metrics.json").write_text(json.dumps(M, indent=2, default=str))
    return M


# ------------------------------------------------------------------ runtime
@dataclass
class System:
    tok: BPETokenizer
    model: GPT
    kb: KnowledgeBase
    router: Router
    rag: RAGPipeline
    ledger: Ledger
    agent: FinanceAgent
    local: LocalGPTProvider


def load_system(art: Path = ART) -> System:
    tok = BPETokenizer.load(art / "tokenizer.json")
    model = _load_model(art / "finmind_gpt.pt")
    kb = KnowledgeBase().build(load_documents(art / "data" / "docs"))
    local = LocalGPTProvider(model, tok, greedy=True)
    router = Router.from_env()
    rag = RAGPipeline(kb, router)
    ledger = Ledger(art / "ledger.db")
    return System(tok, model, kb, router, rag, ledger, FinanceAgent(router, rag, ledger), local)
