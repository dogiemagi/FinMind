"""Phase 9: agentic workflow engine.

A tiny LangGraph-style StateGraph (nodes, conditional edges, checkpointed interrupts for
human-in-the-loop) plus a finance-operations supervisor with RAG / SQL / payments workers,
short-term + long-term memory, policy checks and a full audit trail.
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from .data import AS_OF
from .llm import _extract_json
from .guardrails import mask_pii

END = "__end__"
INTERRUPT = "__interrupt__"


class StateGraph:
    """nodes: name -> fn(state) -> dict of updates. edges: name -> next name or router fn.
    A node may return {INTERRUPT: payload} to pause; resume() continues after the decision."""

    def __init__(self):
        self.nodes: dict[str, Callable] = {}
        self.edges: dict[str, str | Callable] = {}
        self.entry = ""

    def add_node(self, name, fn):
        self.nodes[name] = fn
        self.entry = self.entry or name
        return self

    def add_edge(self, a, b):
        self.edges[a] = b
        return self

    def run(self, state: dict, start: str | None = None) -> dict:
        cur = start or self.entry
        while cur != END:
            upd = self.nodes[cur](state) or {}
            if INTERRUPT in upd:
                state["__paused_at__"] = self.edges[cur] if isinstance(self.edges[cur], str) else cur
                state["__interrupt__"] = upd[INTERRUPT]
                state["status"] = "awaiting_approval"
                return state
            state.update(upd)
            nxt = self.edges.get(cur, END)
            cur = nxt(state) if callable(nxt) else nxt
        state["status"] = "done"
        return state

    def resume(self, state: dict, decision: dict) -> dict:
        start = state.pop("__paused_at__")
        state.pop("__interrupt__", None)
        state["decision"] = decision
        state["status"] = "running"
        return self.run(state, start)


# --------------------------------------------------------------------- tools
class Ledger:
    """Safe data access for agents: read-only SQL gate + audited writes."""
    ALLOWED_TABLES = {"invoices", "payments", "vendors"}

    def __init__(self, db_path):
        self.db = str(db_path)

    def _con(self):
        return sqlite3.connect(self.db)

    def audit(self, actor, action, detail):
        with self._con() as c:
            c.execute("INSERT INTO audit_log(ts,actor,action,detail) VALUES(?,?,?,?)",
                      (datetime.now().isoformat(timespec="seconds"), actor, action, json.dumps(detail)))

    def safe_select(self, sql: str, limit: int = 20) -> list[tuple]:
        return self.select_named(sql, limit)[1]

    def select_named(self, sql: str, limit: int = 20) -> tuple[list[str], list[tuple]]:
        s = sql.strip().rstrip(";")
        if ";" in s or not re.match(r"(?is)^select\b", s):
            raise PermissionError("only a single SELECT statement is allowed")
        if re.search(r"(?i)\b(insert|update|delete|drop|alter|attach|pragma|create)\b", s):
            raise PermissionError("write/DDL keywords are not allowed")
        tables = set(re.findall(r"(?i)\b(?:from|join)\s+(\w+)", s))
        if not tables <= self.ALLOWED_TABLES:
            raise PermissionError(f"table not allowed: {tables - self.ALLOWED_TABLES}")
        with self._con() as c:
            cur = c.execute(f"SELECT * FROM ({s}) LIMIT {limit}")
            return [d[0] for d in cur.description], cur.fetchall()

    def overdue(self) -> list[dict]:
        with self._con() as c:
            c.row_factory = sqlite3.Row
            rows = c.execute("SELECT * FROM invoices WHERE status='pending' AND due_date < ? ORDER BY amount DESC", (AS_OF,)).fetchall()
        return [dict(r) for r in rows]

    def pay(self, invoice_id: str, approver: str) -> None:
        with self._con() as c:
            row = c.execute("SELECT amount,status FROM invoices WHERE id=?", (invoice_id,)).fetchone()
            if not row or row[1] != "pending":
                raise ValueError(f"{invoice_id} is not payable")
            c.execute("UPDATE invoices SET status='paid' WHERE id=?", (invoice_id,))
            c.execute("INSERT INTO payments(invoice_id,amount,paid_at,approved_by) VALUES(?,?,?,?)",
                      (invoice_id, row[0], AS_OF, approver))
        self.audit(approver, "payment_executed", {"invoice": invoice_id, "amount": row[0]})


def required_approver(amount: float) -> str:
    """Encodes approval_policy.txt."""
    return "clerk" if amount < 1000 else "finance_manager" if amount <= 5000 else "cfo"


def _cell(col: str, v):
    if isinstance(v, float) or (isinstance(v, int) and any(k in col for k in ("total", "amount"))):
        return f"${v:,.2f}" if any(k in col for k in ("total", "amount", "spend")) else f"{v:,.2f}"
    return f"{v:,}" if isinstance(v, int) else str(v)


def format_rows(cols: list[str], rows: list[tuple]) -> str:
    """Readable text for a query result (used by the agent's SQL worker)."""
    if not rows:
        return "No rows matched."
    if len(rows) == 1 and len(cols) == 1:
        return f"{cols[0].replace('_', ' ').capitalize()}: {_cell(cols[0], rows[0][0])}"
    return "\n".join(f"• {_cell(cols[0], r[0])} — " + ", ".join(_cell(c, v) for c, v in zip(cols[1:], r[1:])) for r in rows)


SQL_TEMPLATES = [
    (r"how many.*overdue", "SELECT COUNT(*) AS overdue_invoices FROM invoices WHERE status='pending' AND due_date < '%s'" % AS_OF),
    (r"total.*overdue", "SELECT ROUND(SUM(amount),2) AS overdue_total FROM invoices WHERE status='pending' AND due_date < '%s'" % AS_OF),
    (r"(top|largest).*vendor|spend.*vendor|by vendor", "SELECT vendor, ROUND(SUM(amount),2) AS total FROM invoices GROUP BY vendor ORDER BY total DESC"),
    (r"how many.*invoice", "SELECT COUNT(*) AS invoices FROM invoices"),
    (r"total.*paid|paid.*total", "SELECT ROUND(SUM(amount),2) AS paid_total FROM invoices WHERE status='paid'"),
]


# --------------------------------------------------------------- the supervisor
@dataclass
class FinanceAgent:
    router: object
    rag: object
    ledger: Ledger
    short_term: list = field(default_factory=list)       # conversation turns this session

    # ---- long-term memory (persisted in the DB)
    def remember(self, key, value):
        with self.ledger._con() as c:
            c.execute("INSERT OR REPLACE INTO memory VALUES(?,?)", (key, value))

    def recall(self, query):
        with self.ledger._con() as c:
            rows = c.execute("SELECT key,value FROM memory").fetchall()
        q = set(query.lower().split())
        return [f"{k}: {v}" for k, v in rows if q & set(k.lower().split("_")) | q & set(v.lower().split())]

    # ---- nodes
    def _plan(self, s):
        resp = self.router.complete([{"role": "user", "content": s["task"]}],
                                    'Choose a worker. Reply with JSON like {"worker": "rag|sql|payments"} (plan).')
        try:
            worker = _extract_json(resp.text).get("worker", "rag")
        except Exception:  # noqa: BLE001
            worker = "rag"
        s["log"].append(f"plan -> worker={worker}")
        return {"worker": worker if worker in ("rag", "sql", "payments") else "rag"}

    def _rag(self, s):
        a = self.rag.answer(s["task"])
        s["log"].append(f"rag -> {len(a.citations)} citations")
        return {"answer": a.text, "citations": a.citations}

    def _sql(self, s):
        sql = next((q for pat, q in SQL_TEMPLATES if re.search(pat, s["task"].lower())), None)
        if not sql:
            return {"answer": "I could not map that question to a safe query."}
        cols, rows = self.ledger.select_named(sql)
        s["log"].append(f"sql -> {sql}")
        return {"answer": format_rows(cols, rows), "sql": sql}

    def _propose(self, s):
        rows = self.ledger.overdue()
        limit = s.get("max_payments", 5)
        batch = [{"id": r["id"], "vendor": r["vendor"], "amount": r["amount"],
                  "approver_needed": required_approver(r["amount"])} for r in rows[-limit:]]
        s["log"].append(f"propose -> {len(batch)} of {len(rows)} overdue invoices")
        if not batch:
            return {"answer": "No overdue invoices.", "proposal": []}
        return {"proposal": batch}

    def _approval(self, s):
        return {INTERRUPT: {"message": "Approve these payments?", "proposal": s["proposal"]}}

    def _execute(self, s):
        d = s["decision"]
        if not d.get("approved"):
            self.ledger.audit(d.get("approver", "unknown"), "payments_rejected", {"n": len(s["proposal"])})
            return {"answer": "Payments rejected; nothing was executed.", "executed": []}
        order = {"clerk": 0, "finance_manager": 1, "cfo": 2}
        role = d.get("approver_role", "clerk")
        done, skipped = [], []
        for p in s["proposal"]:
            if order[role] < order[p["approver_needed"]]:      # policy enforcement, not just trust
                skipped.append(p["id"])
                continue
            self.ledger.pay(p["id"], f"{d.get('approver', 'unknown')}({role})")
            done.append(p["id"])
        return {"executed": done, "skipped": skipped}

    def _report(self, s):
        if "executed" in s:
            ans = (f"Paid {len(s['executed'])} invoices: {', '.join(s['executed']) or 'none'}. "
                   f"Skipped (needs higher approval): {', '.join(s.get('skipped', [])) or 'none'}.")
            s["answer"] = s.get("answer") or ans
        self.short_term.append({"task": s["task"], "answer": s["answer"]})
        return {}

    def graph(self) -> StateGraph:
        g = StateGraph()
        for name, fn in (("plan", self._plan), ("rag", self._rag), ("sql", self._sql), ("propose", self._propose),
                         ("approval", self._approval), ("execute", self._execute), ("report", self._report)):
            g.add_node(name, fn)
        g.add_edge("plan", lambda s: {"rag": "rag", "sql": "sql", "payments": "propose"}[s["worker"]])
        g.add_edge("rag", "report").add_edge("sql", "report")
        g.add_edge("propose", lambda s: "approval" if s.get("proposal") else "report")
        g.add_edge("approval", "execute").add_edge("execute", "report").add_edge("report", END)
        return g

    # ---- public API
    def start(self, task: str, **kw) -> dict:
        clean, _ = mask_pii(task)
        state = {"task": clean, "log": [], "memory": self.recall(clean), **kw}
        self.ledger.audit("agent", "task_started", {"task": clean})
        return self.graph().run(state)

    def approve(self, state: dict, approved: bool, approver: str, role: str) -> dict:
        return self.graph().resume(state, {"approved": approved, "approver": approver, "approver_role": role})
