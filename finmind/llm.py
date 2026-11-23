"""Phases 5-6: multi-provider LLM router + application engineering.

* Providers: Anthropic, OpenAI, Gemini (plain HTTPS, key from env), the local
  from-scratch GPT, and an Offline extractive provider so everything runs with no keys.
* Router: ordered fallback, retries with backoff, cost + latency accounting.
* Structured output: JSON -> Pydantic validation with automatic repair retries.
* Tool calling: provider-agnostic JSON protocol + ReAct-style loop.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Iterator

from pydantic import BaseModel, ValidationError

# $ per 1M tokens (input, output) -- approximate list prices, used for cost accounting
PRICES = {"anthropic": (3.0, 15.0), "openai": (0.15, 0.60), "gemini": (0.10, 0.40),
          "local": (0.0, 0.0), "offline": (0.0, 0.0)}


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


def _approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)


def _post(url: str, payload: dict, headers: dict, timeout: float = 60) -> dict:
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"content-type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class Provider:
    name = "base"
    model = "n/a"

    def complete(self, messages: list[dict], system: str = "", max_tokens: int = 512) -> str:
        raise NotImplementedError


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5-5"):
        self.model, self.key = model, os.environ["ANTHROPIC_API_KEY"]

    def complete(self, messages, system="", max_tokens=512):
        d = _post("https://api.anthropic.com/v1/messages",
                  {"model": self.model, "max_tokens": max_tokens, "system": system, "messages": messages},
                  {"x-api-key": self.key, "anthropic-version": "2023-06-01"})
        return "".join(b.get("text", "") for b in d["content"])


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model, self.key = model, os.environ["OPENAI_API_KEY"]

    def complete(self, messages, system="", max_tokens=512):
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        d = _post("https://api.openai.com/v1/chat/completions",
                  {"model": self.model, "messages": msgs, "max_tokens": max_tokens},
                  {"authorization": f"Bearer {self.key}"})
        return d["choices"][0]["message"]["content"]


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, model: str = "gemini-2.0-flash"):
        self.model, self.key = model, os.environ["GEMINI_API_KEY"]

    def complete(self, messages, system="", max_tokens=512):
        contents = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]} for m in messages]
        d = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                  {"contents": contents, "systemInstruction": {"parts": [{"text": system}]},
                   "generationConfig": {"maxOutputTokens": max_tokens}},
                  {"x-goog-api-key": self.key})
        return d["candidates"][0]["content"]["parts"][0]["text"]


class LocalGPTProvider(Provider):
    """Serves the model we pretrained + fine-tuned ourselves."""
    name, model = "local", "finmind-mini-gpt"

    def __init__(self, model, tok, **sampling):
        from .sampling import generate
        self._gen, self.m, self.tok = generate, model, tok
        self.sampling = sampling or {"greedy": True}

    def complete(self, messages, system="", max_tokens=80):
        q = messages[-1]["content"]
        ids = self.tok.encode(f"<|user|>{q}<|assistant|>")
        out = self._gen(self.m, ids, max_new=max_tokens, eos=self.tok.eos_id, **self.sampling)
        return self.tok.decode(out[len(ids):]).replace("<|eos|>", "").strip()


class OfflineProvider(Provider):
    """Deterministic stand-in LLM so the whole system runs without API keys or a GPU.
    It is *extractive*: it answers by quoting the best-matching context sentence with a
    citation, routes tool calls by keyword and extracts JSON fields by regex.
    It is NOT a language model -- it exists to exercise the pipeline and tests."""
    name, model = "offline", "extractive-v1"

    def complete(self, messages, system="", max_tokens=512):
        user = messages[-1]["content"]
        if "TOOLS:" in system:
            return self._tool_choice(user, messages)
        if "JSON" in system and "invoice_id" in system:
            return self._extract_invoice(user)
        if "JSON" in system and "plan" in system:
            return json.dumps({"worker": self._worker(user)})
        if "Context:" in user:
            return self._extractive_answer(user)
        if "Rate the answer" in system:
            return json.dumps({"score": 5 if "[" in user else 3, "reason": "offline heuristic"})
        return "I am the offline extractive model; configure an API key for open-ended chat."

    @staticmethod
    def _worker(task: str) -> str:
        t = task.lower().strip()
        if re.search(r"\b(how many|total|sum of|average|top|count|spend|largest|list)\b", t):
            return "sql"                      # read-only data questions first
        if re.match(r"(please )?(pay|settle|process payments?)\b", t):
            return "payments"                 # only imperative payment commands reach the write path
        return "rag"

    def _tool_choice(self, user, messages):
        if any(m["content"].startswith("TOOL_RESULT") for m in messages):
            last = [m for m in messages if m["content"].startswith("TOOL_RESULT")][-1]["content"]
            return json.dumps({"final": last.split(":", 1)[1].strip()})
        return json.dumps({"tool": "search_policy", "args": {"question": user}})

    @staticmethod
    def _extract_invoice(text):
        g = lambda p: (re.search(p, text) or [None, None])[1]
        amt = g(r"\$([\d,]+\.?\d*)")
        return json.dumps({"invoice_id": g(r"(INV-\d{4}-\d{4})"), "vendor": g(r"from ([A-Z][\w ]+?) dated"),
                           "amount": float(amt.replace(",", "")) if amt else None})

    @staticmethod
    def _extractive_answer(prompt):
        ctx, q = prompt.split("\n\nQuestion:", 1)
        from .nlp import normalize
        qset = set(normalize(q))
        lines = [l for l in ctx.splitlines() if l.startswith("[")]
        if not lines:
            return "Not found in the documents."
        scored = sorted(lines, key=lambda l: -len(qset & set(normalize(l))))
        top = [l for l in scored[:4] if qset & set(normalize(l))] or scored[:1]
        out = []
        for l in top:                       # quote the best sentences, each with its citation
            n = re.match(r"\[(\d+)\]", l).group(1)
            out.append(f"{re.sub(r'^\[\d+\] \([^)]*\) ', '', l)} [{n}]")
        return " ".join(out)


@dataclass
class Router:
    providers: list[Provider]
    retries: int = 2
    backoff: float = 0.5
    ledger: list[LLMResponse] = field(default_factory=list)

    @classmethod
    def from_env(cls, local: Provider | None = None) -> "Router":
        ps: list[Provider] = []
        for env, cls_ in (("ANTHROPIC_API_KEY", AnthropicProvider), ("OPENAI_API_KEY", OpenAIProvider),
                          ("GEMINI_API_KEY", GeminiProvider)):
            if os.environ.get(env):
                ps.append(cls_())
        if local:
            ps.append(local)
        ps.append(OfflineProvider())
        return cls(ps)

    def complete(self, messages, system: str = "", max_tokens: int = 512) -> LLMResponse:
        last_err = None
        for p in self.providers:                       # fallback chain
            for attempt in range(self.retries + 1):    # retry with exponential backoff
                t0 = time.perf_counter()
                try:
                    text = p.complete(messages, system, max_tokens)
                except Exception as e:                 # noqa: BLE001 - provider errors vary widely
                    last_err = e
                    time.sleep(self.backoff * 2 ** attempt if p.name != "offline" else 0)
                    continue
                tin = _approx_tokens(system + "".join(m["content"] for m in messages))
                tout = _approx_tokens(text)
                pi, po = PRICES.get(p.name, (0, 0))
                r = LLMResponse(text, p.name, p.model, time.perf_counter() - t0, tin, tout,
                                (tin * pi + tout * po) / 1e6)
                self.ledger.append(r)
                return r
        raise RuntimeError(f"all providers failed: {last_err}")

    def stream(self, messages, system: str = "") -> Iterator[str]:
        """Streaming interface (SSE-ready). Remote providers return whole text here;
        swap in their native streaming endpoints for token-level latency."""
        text = self.complete(messages, system).text
        for w in re.findall(r"\S+\s*", text):
            yield w

    def cost_report(self) -> dict:
        by: dict[str, dict] = {}
        for r in self.ledger:
            d = by.setdefault(r.provider, {"calls": 0, "cost_usd": 0.0, "avg_latency_s": 0.0})
            d["calls"] += 1
            d["cost_usd"] += r.cost_usd
            d["avg_latency_s"] += r.latency_s
        for d in by.values():
            d["avg_latency_s"] = round(d["avg_latency_s"] / d["calls"], 4)
            d["cost_usd"] = round(d["cost_usd"], 6)
        return by


# ----------------------------------------------------------- structured output
def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object found")
    return json.loads(m.group(0))


def structured(router: Router, prompt: str, schema: type[BaseModel], system: str = "", retries: int = 3):
    """Ask for JSON, validate with Pydantic, feed validation errors back and retry."""
    sys = f"{system}\nReply with ONLY JSON matching this schema:\n{json.dumps(schema.model_json_schema())}"
    msgs = [{"role": "user", "content": prompt}]
    err = ""
    for _ in range(retries):
        text = router.complete(msgs, sys).text
        try:
            return schema.model_validate(_extract_json(text))
        except (ValueError, ValidationError) as e:
            err = str(e)
            msgs += [{"role": "assistant", "content": text},
                     {"role": "user", "content": f"Invalid output: {err[:300]}. Return corrected JSON only."}]
    raise ValueError(f"structured output failed: {err}")


class InvoiceExtract(BaseModel):
    invoice_id: str
    vendor: str
    amount: float


# ---------------------------------------------------------------- tool calling
@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., str]
    schema: type[BaseModel]


def run_tools(router: Router, task: str, tools: list[Tool], max_steps: int = 4) -> tuple[str, list[dict]]:
    """ReAct-style loop: model chooses a tool -> we validate args -> run -> feed result back."""
    spec = "\n".join(f"- {t.name}: {t.description} args={json.dumps(t.schema.model_json_schema()['properties'])}" for t in tools)
    system = ('TOOLS:\n' + spec + '\nRespond with JSON: {"tool": name, "args": {...}} to call a tool, '
              'or {"final": answer} when done.')
    msgs, trace = [{"role": "user", "content": task}], []
    by_name = {t.name: t for t in tools}
    for _ in range(max_steps):
        out = _extract_json(router.complete(msgs, system).text)
        if "final" in out:
            return str(out["final"]), trace
        tool = by_name.get(out.get("tool", ""))
        if not tool:
            msgs.append({"role": "user", "content": f"TOOL_ERROR: unknown tool {out.get('tool')}"})
            continue
        try:
            args = tool.schema.model_validate(out.get("args", {}))
            result = tool.fn(**args.model_dump())
        except Exception as e:  # noqa: BLE001
            result = f"error: {e}"
        trace.append({"tool": tool.name, "args": out.get("args"), "result": result})
        msgs += [{"role": "assistant", "content": json.dumps(out)}, {"role": "user", "content": f"TOOL_RESULT: {result}"}]
    return "stopped: step limit reached", trace
