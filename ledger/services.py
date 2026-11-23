"""ORM-backed ledger for the agent. Subclasses the framework-free `finmind.agent.Ledger`:
reads/writes go through Django models and transactions, while the read-only SQL gate
(`_validate`) is inherited unchanged and then executed on Django's connection."""
from __future__ import annotations

import json
from datetime import datetime

from django.db import connection, transaction

from finmind.agent import Ledger
from finmind.data import AS_OF

from .models import AuditLog, Invoice, Memory, Payment


class DjangoLedger(Ledger):
    def __init__(self):          # no file path: Django owns the connection
        self.db = "django"

    def audit(self, actor, action, detail):
        AuditLog.objects.create(ts=datetime.now().replace(microsecond=0), actor=actor, action=action,
                                detail=json.dumps(detail))

    def select_named(self, sql: str, limit: int = 20):
        s = self._validate(sql)
        with connection.cursor() as cur:
            cur.execute(f"SELECT * FROM ({s}) LIMIT {int(limit)}")
            return [c[0] for c in cur.description], cur.fetchall()

    def overdue(self) -> list[dict]:
        qs = Invoice.objects.filter(status=Invoice.Status.PENDING, due_date__lt=AS_OF).order_by("-amount")
        return [{"id": i.id, "vendor": i.vendor, "category": i.category, "description": i.description,
                 "amount": i.amount, "invoice_date": i.invoice_date.isoformat(), "due_date": i.due_date.isoformat(),
                 "status": i.status} for i in qs]

    def pay(self, invoice_id: str, approver: str) -> None:
        with transaction.atomic():
            inv = Invoice.objects.select_for_update().filter(pk=invoice_id).first()
            if inv is None or inv.status != Invoice.Status.PENDING:
                raise ValueError(f"{invoice_id} is not payable")
            inv.status = Invoice.Status.PAID
            inv.save(update_fields=["status"])
            Payment.objects.create(invoice=inv, amount=inv.amount, paid_at=AS_OF, approved_by=approver)
        self.audit(approver, "payment_executed", {"invoice": invoice_id, "amount": inv.amount})

    def remember(self, key: str, value: str) -> None:
        Memory.objects.update_or_create(key=key, defaults={"value": value})

    def memory_rows(self) -> list[tuple]:
        return list(Memory.objects.values_list("key", "value"))
