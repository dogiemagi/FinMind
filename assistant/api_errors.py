"""Map domain errors to clean JSON responses."""
from rest_framework.response import Response
from rest_framework.views import exception_handler

from finmind.guardrails import GuardrailViolation

from .services import SystemNotReady
from .telemetry import METRICS


def handler(exc, context):
    if isinstance(exc, GuardrailViolation):
        METRICS["guardrail_blocked_total"] += 1
        return Response({"detail": str(exc)}, status=400)
    if isinstance(exc, SystemNotReady):
        return Response({"detail": str(exc)}, status=503)
    return exception_handler(exc, context)
