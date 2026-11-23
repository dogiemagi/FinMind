from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import transaction

from finmind import data as D
from ledger.models import AuditLog, Invoice, Memory, Payment


class Command(BaseCommand):
    help = "Load the synthetic invoice ledger and write the policy / contract documents used by RAG."

    def add_arguments(self, parser):
        parser.add_argument("--count", type=int, default=300, help="number of invoices")
        parser.add_argument("--reset", action="store_true", help="delete existing ledger rows first")

    @transaction.atomic
    def handle(self, *args, **opts):
        if Invoice.objects.exists() and not opts["reset"]:
            self.stdout.write(self.style.WARNING("Ledger already has data — use --reset to reload it."))
        else:
            for model in (Payment, AuditLog, Memory, Invoice):
                model.objects.all().delete()
            rows = D.make_invoices(opts["count"])
            Invoice.objects.bulk_create([Invoice(**r) for r in rows])
            self.stdout.write(self.style.SUCCESS(f"Loaded {len(rows)} invoices."))
        docs = D.write_documents(settings.FINMIND_ARTIFACTS / "data")
        self.stdout.write(self.style.SUCCESS(f"Wrote policy documents to {docs}"))
