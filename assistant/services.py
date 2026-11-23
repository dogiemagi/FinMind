"""Lazy, process-wide handle on the FinMind runtime (knowledge base, router, agent, optional GPT)."""
from __future__ import annotations

from django.conf import settings

from finmind.pipeline import System, load_system
from ledger.services import DjangoLedger


class SystemNotReady(Exception):
    """Raised when required artifacts are missing; the API turns this into a 503 with the fix."""


_system: System | None = None


def get_system() -> System:
    global _system
    if _system is None:
        art = settings.FINMIND_ARTIFACTS
        if not (art / "data" / "docs").exists():
            raise SystemNotReady("Policy documents are missing. Run: python manage.py seed_ledger")
        has_model = (art / "finmind_gpt.pt").exists() and (art / "tokenizer.json").exists()
        _system = load_system(art, ledger=DjangoLedger(), with_model=has_model)
    return _system


def set_system(system: System | None) -> None:
    """Swap the runtime (used by tests to avoid training a model)."""
    global _system
    _system = system


def require_model(system: System) -> None:
    if system.model is None:
        raise SystemNotReady("The GPT is not trained yet. Run: python manage.py train_model")
