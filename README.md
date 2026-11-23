# FinMind — one project, every LLM-engineering topic

A finance-operations assistant that answers policy questions with citations, queries a ledger,
extracts invoice data, and executes payments **only after human approval** — running on a
GPT we wrote, pretrained, LoRA-tuned, DPO-aligned and quantized ourselves, behind a multi-provider
router (Claude / GPT / Gemini / local / offline), with an evaluation gate, API, Docker and K8s.

```
python -m finmind train     # builds data, tokenizer, trains+tunes+quantizes the GPT  (~3 min CPU)
python -m finmind demo      # RAG, structured output, tools, guardrails, local model, agent + approval
python -m finmind eval      # compares 4 retrieval configs, runs the CI regression gate
python -m finmind serve     # FastAPI on :8000   (then: python -m finmind bench)
python -m pytest tests -q   # 21 tests
```

## Roadmap phase → where it lives

| Phase | Topic | File | What you can inspect |
|---|---|---|---|
| 1 Foundations | tensors, autograd, loss, optimiser, dropout, early stopping | `classifier.py` | MLP invoice classifier vs majority baseline |
| 2 NLP | tokenise, stop-words, stemming, vocab, one-hot, Word2Vec, **BPE** | `nlp.py`, `tokenizer.py` | `tok.tokens("INV-2025-0042")` shows subword splits |
| 3 Transformer | Q/K/V, scaled dot-product, causal mask, multi-head, residual, LayerNorm, GPT | `gpt.py` | `CausalSelfAttention.forward` |
| 4 LLM fundamentals | pretraining, perplexity, greedy / temperature / top-k / top-p / beam, **KV cache** | `train.py`, `sampling.py` | `metrics.json → phase4_kv_cache` |
| 5 Ecosystem | Claude, OpenAI, Gemini, local — cost/latency per token | `llm.py` (`Router`, `cost_report`) | set `ANTHROPIC_API_KEY` etc. |
| 6 App engineering | structured JSON (Pydantic + repair), function calling, ReAct loop, streaming (SSE), retries/fallback, guardrails | `llm.py`, `guardrails.py`, `api.py` | `/ask`, `/extract`, injection + PII tests |
| 7 RAG | loaders, sentence chunking + overlap, embeddings, BM25, **hybrid RRF**, reranker, metadata filter, context compression, citations, abstention | `rag.py` | `retrieve(mode=…, rerank=…, filters=…)` |
| 8 Fine-tuning | **LoRA from scratch**, SFT with loss masking, **DPO**, adapter merge, held-out eval; real QLoRA with TRL/PEFT | `lora.py`, `scripts/finetune_hf.py` | `phase8_*` metrics |
| 9 Agents | state graph, planner/supervisor, tools, short + long-term memory, **human-in-the-loop interrupt/resume**, policy enforcement, audit log, safe SQL gate | `agent.py` | `demo`, `/agent/run` → `/agent/approve` |
| 10 Evaluation | EM, P/R/F1, ROUGE-L, BLEU, perplexity, context precision/recall, MRR, faithfulness, relevancy, LLM-as-judge, config comparison, **regression gate** | `evals.py` | `python -m finmind eval` |
| 11 Deployment | INT8/INT4 quantization, load test (p50/p95/RPS), Docker, Compose, K8s + HPA, Prometheus, CI, vLLM script | `quant.py`, `deploy/`, `scripts/serve_vllm.sh`, `.github/` | `phase11_quant` |
| 12 Research | **MoE**, FlashAttention-style tiled attention, speculative decoding, distillation, self-consistency | `gpt.py`, `research.py`, `sampling.py` | `phase12_speculative` |

## Measured results (this machine, CPU)

| Measurement | Result |
|---|---|
| Invoice classifier (Phase 1) | 100% val accuracy vs 14% majority baseline (synthetic data is easy) |
| BPE tokenizer | ~3 bytes per token; `INV-2025-0042` splits into digits/pieces |
| KV cache | ~1.7× faster generation (233 vs 133 tok/s) |
| SFT + LoRA | held-out loss 6.86 → 1.54; output is now valid invoice JSON |
| DPO | preference accuracy 1.0 on the training pairs |
| Quantization | INT8: 63% smaller, same loss. INT4: 72% smaller, small loss increase |
| Speculative decoding | output identical to greedy (tested); acceptance 13.5% with a distilled 1-layer draft |
| RAG (12 gold questions) | correct 1.0, context recall 1.0, MRR 0.96–1.0 across configs |

## Be honest about what is a stand-in

* **The mini-GPT is ~400k parameters.** It learns the *format* of invoice JSON but invents the
  values (see `phase8_sample_output`). Its job is to make every mechanism real and inspectable,
  not to be useful. For a useful tuned model run `scripts/finetune_hf.py` on a GPU.
* **The offline provider is not an LLM.** It is an extractive stand-in so everything runs with
  no keys: it quotes the best-matching sentences with citations and cannot reason (it can't
  tell that $7,500 is "above $5,000" — retrieval surfaces the CFO rule, a real LLM concludes).
  Set `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` or `GEMINI_API_KEY` and the router prefers it,
  falling back to offline on errors. The remote providers are written but **untested live**.
* **Embeddings are hashed TF-IDF**, so "dense" retrieval is lexical here. Install
  `sentence-transformers` and pass `SentenceTransformerEmbedder` to `KnowledgeBase` for semantics.
* **The eval set is 12 hand-written questions on 7 tiny docs** — a pipeline demo, not a benchmark.
  Several metrics saturate at 1.0; grow the set before trusting differences between configs.
* Not run here: Docker build, Kubernetes, vLLM (needs a Linux GPU), Prometheus/Grafana, real QLoRA.

## Suggested way to learn from it

Read in phase order. For each module: read it, break something on purpose (remove the causal mask,
set LoRA `r=1`, set `overlap_sents=0`, lower `min_score`), re-run `train`/`eval`/`pytest`, and
explain the change. Then replace one stand-in: first the embedder, then a real provider, then QLoRA.
