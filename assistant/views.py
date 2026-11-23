"""REST API for chat, extraction, the model lab and reports."""
from __future__ import annotations

import json
import re

from django.conf import settings
from django.http import HttpResponse, StreamingHttpResponse
from rest_framework.response import Response
from rest_framework.views import APIView

from finmind.guardrails import check_input, check_output
from finmind.llm import InvoiceExtract, structured
from finmind.sampling import generate
from finmind.smalltalk import smalltalk

from .serializers import AskSerializer, GenerateSerializer, TextSerializer
from .services import get_system, require_model
from .telemetry import METRICS


class HealthView(APIView):
    def get(self, request):
        s = get_system()
        return Response({"ok": True, "chunks": len(s.kb.chunks), "providers": [p.name for p in s.router.providers],
                         "model_loaded": s.model is not None})


class AskView(APIView):
    """Grounded Q&A. Greetings are answered directly; everything else goes through hybrid RAG
    with citations, or an explicit 'not found'. `stream=true` returns server-sent events."""

    def post(self, request):
        data = AskSerializer(data=request.data)
        data.is_valid(raise_exception=True)
        d = data.validated_data
        s = get_system()
        question, _ = check_input(d["question"])          # PII masking + prompt-injection guard
        chat = smalltalk(question)
        if chat is not None:
            text, cites, kind, abstained = chat, [], "chat", False
        else:
            ans = s.rag.answer(question, mode=d["mode"], rerank=d["rerank"])
            text, cites, abstained = ans.text, ans.citations, ans.abstained
            kind = "declined" if abstained else "answer"
        if d["stream"]:
            def events():
                for w in re.findall(r"\S+\s*", text):
                    yield f"data: {json.dumps({'token': w})}\n\n"
                yield f"data: {json.dumps({'done': True, 'citations': cites, 'kind': kind})}\n\n"
            resp = StreamingHttpResponse(events(), content_type="text/event-stream")
            resp["Cache-Control"] = "no-cache"
            resp["X-Accel-Buffering"] = "no"
            return resp
        return Response({"answer": text, "citations": cites, "abstained": abstained, "kind": kind,
                         "checks": check_output(text, cites)})


class ExtractView(APIView):
    def post(self, request):
        data = TextSerializer(data=request.data)
        data.is_valid(raise_exception=True)
        text, _ = check_input(data.validated_data["text"])
        try:
            inv = structured(get_system().router, text, InvoiceExtract, system="Extract invoice_id, vendor, amount as JSON.")
        except ValueError as e:
            return Response({"detail": str(e)}, status=422)
        return Response(inv.model_dump())


class TokenizeView(APIView):
    def post(self, request):
        data = TextSerializer(data=request.data)
        data.is_valid(raise_exception=True)
        s = get_system()
        require_model(s)
        text = data.validated_data["text"][:500]
        ids = s.tok.encode(text)
        return Response({"ids": ids, "chars": len(text), "bytes": len(text.encode()),
                         "tokens": [s.tok.vocab[i].decode("utf-8", "replace") if i in s.tok.vocab else "<special>" for i in ids]})


class GenerateView(APIView):
    def post(self, request):
        data = GenerateSerializer(data=request.data)
        data.is_valid(raise_exception=True)
        d = data.validated_data
        s = get_system()
        require_model(s)
        prompt, _ = check_input(d["prompt"])
        ids = s.tok.encode(f"<|user|>{prompt}<|assistant|>")
        stats: dict = {}
        out = generate(s.model, ids, max_new=d["max_new"], eos=s.tok.eos_id, stats=stats, greedy=d["greedy"],
                       temperature=d["temperature"], top_k=d["top_k"] or None, top_p=d["top_p"] if d["top_p"] < 1 else None)
        return Response({"text": s.tok.decode(out[len(ids):]).replace("<|eos|>", ""),
                         "tokens_per_sec": round(stats["tokens_per_sec"], 1)})


class ReportView(APIView):
    """Training metrics (`train_model`) and retrieval evaluation (`python -m finmind eval`)."""

    def get(self, request):
        out = {}
        for name in ("metrics", "eval_report"):
            p = settings.FINMIND_ARTIFACTS / f"{name}.json"
            out[name] = json.loads(p.read_text()) if p.exists() else None
        return Response(out)


class MetricsView(APIView):
    """Prometheus text format."""
    authentication_classes: list = []

    def get(self, request):
        lines = [f"finmind_{k} {v}" for k, v in METRICS.items()]
        try:
            lines += [f'finmind_llm_cost_usd{{provider="{p}"}} {d["cost_usd"]}' for p, d in get_system().router.cost_report().items()]
        except Exception:  # noqa: BLE001 - metrics must never fail the scrape
            pass
        return HttpResponse("\n".join(lines) + "\n", content_type="text/plain; version=0.0.4")
