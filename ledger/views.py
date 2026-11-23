from django.db.models import Count, Sum
from rest_framework.response import Response
from rest_framework.views import APIView

from finmind.agent import required_approver
from finmind.data import AS_OF

from .models import AuditLog, Invoice, Payment
from .services import DjangoLedger


class SummaryView(APIView):
    """Dashboard numbers for the Ledger page (single source of truth: the ORM)."""

    def get(self, request):
        overdue = Invoice.objects.filter(status="pending", due_date__lt=AS_OF).aggregate(n=Count("id"), total=Sum("amount"))
        by_vendor = Invoice.objects.values("vendor").annotate(total=Sum("amount")).order_by("-total")
        return Response({
            "invoices": Invoice.objects.count(),
            "paid": Invoice.objects.filter(status="paid").count(),
            "overdue": [overdue["n"], round(overdue["total"] or 0, 2)],
            "by_vendor": [[v["vendor"], round(v["total"], 2)] for v in by_vendor],
            "payments": [[p.invoice_id, p.amount, p.approved_by] for p in Payment.objects.all()[:8]],
            "audit": [[a.ts.isoformat(timespec="seconds"), a.actor, a.action, a.detail] for a in AuditLog.objects.all()[:8]],
        })


class OverdueView(APIView):
    def get(self, request):
        rows = DjangoLedger().overdue()[:12]
        return Response([{"id": r["id"], "vendor": r["vendor"], "amount": r["amount"], "due": r["due_date"],
                          "approver": required_approver(r["amount"])} for r in rows])
