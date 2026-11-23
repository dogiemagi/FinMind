#!/usr/bin/env bash
# Phase 11: serve an open-weight model with vLLM (Linux + NVIDIA GPU).
# vLLM gives continuous batching + PagedAttention KV-cache; it exposes an OpenAI-compatible API.
set -euo pipefail
MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
pip install vllm
python -m vllm.entrypoints.openai.api_server --model "$MODEL" --port 8001 --max-model-len 4096 --gpu-memory-utilization 0.9 &
sleep 60
# benchmark the same load test used for the FastAPI gateway:
python - <<'PY'
from finmind.quant import load_test
print(load_test("http://127.0.0.1:8001/v1/chat/completions",
      {"model": "Qwen/Qwen2.5-1.5B-Instruct", "messages": [{"role": "user", "content": "Summarise net 30 terms."}], "max_tokens": 64},
      n=100, concurrency=16))
PY
# To use it from FinMind, point an OpenAI-compatible provider at http://localhost:8001/v1.
wait
