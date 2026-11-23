from django.test import TestCase
from rest_framework.test import APIClient

from finmind.data import make_invoices

from .models import AuditLog, Invoice, Payment
from .services import DjangoLedger


def seed(n=60):
    Invoice.objects.bulk_create([Invoice(**r) for r in make_invoices(n)])


class LedgerServiceTests(TestCase):
    def setUp(self):
        seed()
        self.ledger = DjangoLedger()

    def test_sql_gate_allows_select_and_blocks_everything_else(self):
        cols, rows = self.ledger.select_named("SELECT COUNT(*) AS n FROM invoices")
        self.assertEqual((cols, rows[0][0]), (["n"], 60))
        for bad in ("DELETE FROM invoices", "SELECT 1; DROP TABLE invoices", "SELECT * FROM audit_log",
                    "UPDATE invoices SET amount=0", "PRAGMA table_info(invoices)"):
            with self.assertRaises(PermissionError, msg=bad):
                self.ledger.select_named(bad)
        self.assertEqual(Invoice.objects.count(), 60)

    def test_overdue_is_pending_and_past_due_largest_first(self):
        rows = self.ledger.overdue()
        self.assertTrue(rows)
        self.assertEqual([r["amount"] for r in rows], sorted((r["amount"] for r in rows), reverse=True))
        self.assertTrue(all(r["status"] == "pending" and r["due_date"] < "2025-06-30" for r in rows))

    def test_pay_updates_invoice_creates_payment_and_audit_row(self):
        inv = self.ledger.overdue()[-1]
        self.ledger.pay(inv["id"], "Priya(finance_manager)")
        self.assertEqual(Invoice.objects.get(pk=inv["id"]).status, "paid")
        self.assertEqual(Payment.objects.get(invoice_id=inv["id"]).approved_by, "Priya(finance_manager)")
        self.assertTrue(AuditLog.objects.filter(action="payment_executed").exists())
        with self.assertRaises(ValueError):          # cannot pay twice
            self.ledger.pay(inv["id"], "Priya")

    def test_memory_roundtrip(self):
        self.ledger.remember("vendor_pref", "Acme Cloud prefers Tuesday payments")
        self.ledger.remember("vendor_pref", "Acme Cloud prefers Friday payments")   # upsert
        self.assertEqual(self.ledger.memory_rows(), [("vendor_pref", "Acme Cloud prefers Friday payments")])


class LedgerApiTests(TestCase):
    def setUp(self):
        seed()
        self.client = APIClient()

    def test_summary_shape(self):
        d = self.client.get("/api/ledger/summary").json()
        self.assertEqual(d["invoices"], 60)
        self.assertEqual(len(d["overdue"]), 2)
        self.assertTrue(all(len(v) == 2 for v in d["by_vendor"]))

    def test_overdue_includes_required_approver(self):
        rows = self.client.get("/api/ledger/overdue").json()
        self.assertTrue(rows and {"id", "vendor", "amount", "due", "approver"} <= set(rows[0]))
        for r in rows:
            expected = "clerk" if r["amount"] < 1000 else "finance_manager" if r["amount"] <= 5000 else "cfo"
            self.assertEqual(r["approver"], expected)
