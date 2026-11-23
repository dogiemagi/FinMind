import shutil
import tempfile
from pathlib import Path

from django.test import TestCase
from rest_framework.test import APIClient

from assistant import services
from finmind import data as D
from finmind.pipeline import load_system
from ledger.models import AuditLog, Invoice, Payment
from ledger.services import DjangoLedger

from .models import AgentRun


class AgentApiTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tmp = Path(tempfile.mkdtemp())
        D.write_documents(cls.tmp / "data")
        services.set_system(load_system(cls.tmp, ledger=DjangoLedger(), with_model=False))

    @classmethod
    def tearDownClass(cls):
        services.set_system(None)
        shutil.rmtree(cls.tmp, ignore_errors=True)
        super().tearDownClass()

    def setUp(self):
        Invoice.objects.bulk_create([Invoice(**r) for r in D.make_invoices(120)])
        self.client = APIClient()

    def run_agent(self, task):
        return self.client.post("/api/agent/run", {"task": task}, format="json").json()

    def decide(self, run_id, approved, role="finance_manager", who="Priya"):
        return self.client.post("/api/agent/approve", {"run_id": run_id, "approved": approved, "approver": who, "role": role}, format="json")

    def test_read_only_question_finishes_without_approval(self):
        d = self.run_agent("How many overdue invoices do we have?")
        self.assertEqual(d["status"], "done")
        self.assertIn("Overdue invoices:", d["answer"])
        self.assertFalse(AgentRun.objects.exists())

    def test_overdue_question_never_routes_to_payments(self):
        d = self.run_agent("How many overdue invoices do we have?")
        self.assertIn("worker=sql", " ".join(d["log"]))

    def test_greeting(self):
        self.assertEqual(self.run_agent("hi")["kind"], "chat")

    def test_payment_task_pauses_and_persists_run(self):
        d = self.run_agent("Pay the overdue invoices")
        self.assertEqual(d["status"], "awaiting_approval")
        self.assertEqual(AgentRun.objects.get(run_id=d["run_id"]).status, "awaiting_approval")
        self.assertEqual(Payment.objects.count(), 0)                      # nothing paid yet

    def test_approval_pays_within_role_and_skips_above(self):
        d = self.run_agent("Pay the overdue invoices")
        proposal = d["needs_approval"]["proposal"]
        r = self.decide(d["run_id"], True, role="clerk").json()
        affordable = [p["id"] for p in proposal if p["approver_needed"] == "clerk"]
        self.assertEqual(sorted(r["executed"]), sorted(affordable))
        self.assertEqual(Payment.objects.count(), len(affordable))
        self.assertEqual(Invoice.objects.filter(status="paid", pk__in=affordable).count(), len(affordable))
        self.assertTrue(AuditLog.objects.filter(action="payment_executed").count() == len(affordable))

    def test_rejection_executes_nothing_and_is_audited(self):
        d = self.run_agent("Pay the overdue invoices")
        r = self.decide(d["run_id"], False, role="cfo").json()
        self.assertEqual(r["executed"], [])
        self.assertEqual(Payment.objects.count(), 0)
        self.assertTrue(AuditLog.objects.filter(action="payments_rejected").exists())
        self.assertEqual(AgentRun.objects.get(run_id=d["run_id"]).status, "rejected")

    def test_run_can_only_be_decided_once(self):
        d = self.run_agent("Pay the overdue invoices")
        self.assertEqual(self.decide(d["run_id"], True).status_code, 200)
        self.assertEqual(self.decide(d["run_id"], True).status_code, 404)    # no double payment

    def test_unknown_run_and_bad_role(self):
        self.assertEqual(self.decide("nope", True).status_code, 404)
        d = self.run_agent("Pay the overdue invoices")
        self.assertEqual(self.decide(d["run_id"], True, role="intern").status_code, 400)
