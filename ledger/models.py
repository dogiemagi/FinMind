"""The finance ledger. Table names match the SQL the agent's read-only gate and the
framework-free `finmind.agent.Ledger` expect (`invoices`, `payments`, `audit_log`, `memory`)."""
from django.db import models


class Invoice(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending"
        PAID = "paid"

    id = models.CharField(max_length=20, primary_key=True)
    vendor = models.CharField(max_length=80, db_index=True)
    category = models.CharField(max_length=40, db_index=True)
    description = models.CharField(max_length=200)
    amount = models.FloatField()
    invoice_date = models.DateField()
    due_date = models.DateField(db_index=True)
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.PENDING, db_index=True)

    class Meta:
        db_table = "invoices"
        ordering = ["id"]

    def __str__(self):
        return f"{self.id} · {self.vendor} · ${self.amount:,.2f}"


class Payment(models.Model):
    invoice = models.ForeignKey(Invoice, on_delete=models.PROTECT, db_column="invoice_id", related_name="payments")
    amount = models.FloatField()
    paid_at = models.DateField()
    approved_by = models.CharField(max_length=100)

    class Meta:
        db_table = "payments"
        ordering = ["-id"]

    def __str__(self):
        return f"{self.invoice_id} paid by {self.approved_by}"


class AuditLog(models.Model):
    ts = models.DateTimeField()
    actor = models.CharField(max_length=100)
    action = models.CharField(max_length=60, db_index=True)
    detail = models.TextField(blank=True)

    class Meta:
        db_table = "audit_log"
        ordering = ["-id"]

    def __str__(self):
        return f"{self.ts:%Y-%m-%d %H:%M} {self.actor} {self.action}"


class Memory(models.Model):
    """Agent long-term memory (key → fact)."""
    key = models.CharField(max_length=100, primary_key=True)
    value = models.TextField()

    class Meta:
        db_table = "memory"
        verbose_name_plural = "memory"

    def __str__(self):
        return self.key
