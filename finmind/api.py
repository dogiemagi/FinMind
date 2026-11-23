"""Phase 6/9/11: FastAPI gateway -- streaming, structured output, guardrails, agent with
human approval, Prometheus-style metrics and a simple rate limiter."""
from __future__ import annotations

import json
import re
import time
import uuid
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from pathlib import Path

from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from .guardrails import GuardrailViolation, check_input, check_output
from .llm import InvoiceExtract, structured
from .pipeline import ART, load_system
from .smalltalk import smalltalk

app = FastAPI(title="FinMind", version="1.0")
S = None
PENDING: dict[str, dict] = {}
METRICS = defaultdict(float)
HITS: dict[str, deque] = defaultdict(deque)


def system():
    global S
    if S is None:
        S = load_system(ART)
    return S


@app.middleware("http")
async def telemetry(request: Request, call_next):
    ip, now = request.client.host if request.client else "x", time.time()
    q = HITS[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= 600:                                   # 600 req/min/IP
        METRICS["rate_limited_total"] += 1
        return PlainTextResponse("rate limit exceeded", status_code=429)
    q.append(now)
    t = time.perf_counter()
    resp = await call_next(request)
    METRICS["requests_total"] += 1
    METRICS["latency_seconds_sum"] += time.perf_counter() - t
    return resp


class AskReq(BaseModel):
    question: str
    mode: str = "hybrid"
    rerank: bool = True
    stream: bool = False


class ExtractReq(BaseModel):
    text: str


class AgentReq(BaseModel):
    task: str


class ApproveReq(BaseModel):
    run_id: str
    approved: bool
    approver: str
    role: str = "finance_manager"


def _guard(text: str) -> str:
    try:
        clean, _ = check_input(text)
        return clean
    except GuardrailViolation as e:
        METRICS["guardrail_blocked_total"] += 1
        raise HTTPException(400, str(e))


UI = Path(__file__).with_name("ui.html")


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(UI)


class TokReq(BaseModel):
    text: str


class GenReq(BaseModel):
    prompt: str
    max_new: int = 60
    temperature: float = 0.8
    top_k: int = 0
    top_p: float = 1.0
    greedy: bool = False


@app.post("/tokenize")
def tokenize(req: TokReq):
    t = system().tok
    ids = t.encode(req.text[:500])
    return {"ids": ids, "tokens": [t.vocab[i].decode("utf-8", "replace") if i in t.vocab else "<special>" for i in ids],
            "chars": len(req.text[:500]), "bytes": len(req.text[:500].encode())}


@app.post("/generate")
def generate_text(req: GenReq):
    from .sampling import generate
    s = system()
    ids = s.tok.encode(f"<|user|>{_guard(req.prompt)}<|assistant|>")
    stats: dict = {}
    out = generate(s.model, ids, max_new=min(req.max_new, 120), eos=s.tok.eos_id, stats=stats, greedy=req.greedy,
                   temperature=req.temperature, top_k=req.top_k or None, top_p=req.top_p if req.top_p < 1 else None)
    return {"text": s.tok.decode(out[len(ids):]).replace("<|eos|>", ""), "tokens_per_sec": round(stats["tokens_per_sec"], 1)}


@app.get("/ledger/summary")
def ledger_summary():
    L = system().ledger
    one = lambda q: L.safe_select(q)[0]
    return {
        "invoices": one("SELECT COUNT(*) FROM invoices")[0],
        "paid": one("SELECT COUNT(*) FROM invoices WHERE status='paid'")[0],
        "overdue": one("SELECT COUNT(*), ROUND(COALESCE(SUM(amount),0),2) FROM invoices WHERE status='pending' AND due_date < '2025-06-30'"),
        "by_vendor": L.safe_select("SELECT vendor, ROUND(SUM(amount),2) FROM invoices GROUP BY vendor ORDER BY 2 DESC"),
        "payments": L.safe_select("SELECT invoice_id, amount, approved_by FROM payments ORDER BY id DESC", 8),
        "audit": [list(r) for r in _audit(L)],
    }


@app.get("/ledger/overdue")
def ledger_overdue():
    from .agent import required_approver
    rows = system().ledger.overdue()[:12]
    return [{"id": r["id"], "vendor": r["vendor"], "amount": r["amount"], "due": r["due_date"],
             "approver": required_approver(r["amount"])} for r in rows]


def _audit(L):
    with L._con() as c:
        return c.execute("SELECT ts, actor, action, detail FROM audit_log ORDER BY id DESC LIMIT 8").fetchall()


@app.get("/report")
def report():
    out = {}
    for name in ("metrics", "eval_report"):
        p = ART / f"{name}.json"
        out[name] = json.loads(p.read_text()) if p.exists() else None
    return out


@app.get("/health")
def health():
    s = system()
    return {"ok": True, "chunks": len(s.kb.chunks), "providers": [p.name for p in s.router.providers]}


@app.post("/ask")
def ask(req: AskReq):
    s = system()
    q = _guard(req.question)
    chat = smalltalk(q)                       # greetings / thanks / help never hit retrieval
    if chat is not None:
        text, cites, kind, abstained = chat, [], "chat", False
    else:
        ans = s.rag.answer(q, mode=req.mode, rerank=req.rerank)
        text, cites, abstained = ans.text, ans.citations, ans.abstained
        kind = "declined" if abstained else "answer"
    if req.stream:
        def gen():
            for w in re.findall(r"\S+\s*", text):
                yield f"data: {json.dumps({'token': w})}\n\n"
            yield f"data: {json.dumps({'done': True, 'citations': cites, 'kind': kind})}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    return {"answer": text, "citations": cites, "abstained": abstained, "kind": kind,
            "checks": check_output(text, cites)}


@app.post("/extract")
def extract(req: ExtractReq):
    s = system()
    try:
        return structured(s.router, _guard(req.text), InvoiceExtract, system="Extract invoice_id, vendor, amount as JSON.").model_dump()
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.post("/agent/run")
def agent_run(req: AgentReq):
    s = system()
    task = _guard(req.task)
    chat = smalltalk(task)
    if chat is not None:
        return {"status": "done", "answer": chat, "log": [], "kind": "chat"}
    state = s.agent.start(task)
    if state["status"] == "awaiting_approval":
        run_id = uuid.uuid4().hex[:8]
        PENDING[run_id] = state
        return {"run_id": run_id, "status": state["status"], "needs_approval": state["__interrupt__"], "log": state["log"]}
    return {"status": state["status"], "answer": state.get("answer"), "log": state["log"]}


@app.post("/agent/approve")
def agent_approve(req: ApproveReq):
    s = system()
    state = PENDING.pop(req.run_id, None)
    if state is None:
        raise HTTPException(404, "unknown run_id")
    out = s.agent.approve(state, req.approved, req.approver, req.role)
    return {"status": out["status"], "answer": out.get("answer"), "executed": out.get("executed"), "skipped": out.get("skipped")}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    s = system()
    lines = [f"finmind_{k} {v}" for k, v in METRICS.items()]
    lines += [f'finmind_llm_cost_usd{{provider="{p}"}} {d["cost_usd"]}' for p, d in s.router.cost_report().items()]
    return "\n".join(lines) + "\n"
