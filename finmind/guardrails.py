"""Phase 6: guardrails -- PII masking, prompt-injection screening, output grounding check."""
from __future__ import annotations

import re

PII_PATTERNS = {
    "EMAIL": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "PHONE": re.compile(r"\b\+?\d{1,2}[ -]?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}\b"),
}
INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|rules)",
    r"disregard (the )?(system|previous) (prompt|instructions)",
    r"reveal (your )?(system prompt|hidden instructions)",
    r"you are now (?:in )?(developer|dan|jailbreak)",
    r"pretend (that )?you have no (rules|restrictions)",
]


def mask_pii(text: str) -> tuple[str, dict]:
    found: dict[str, int] = {}
    for label, pat in PII_PATTERNS.items():
        text, n = pat.subn(f"[{label}]", text)
        if n:
            found[label] = n
    return text, found


def injection_score(text: str) -> list[str]:
    low = text.lower()
    return [p for p in INJECTION_PATTERNS if re.search(p, low)]


class GuardrailViolation(Exception):
    pass


def check_input(text: str, max_chars: int = 4000) -> tuple[str, dict]:
    if len(text) > max_chars:
        raise GuardrailViolation("input too long")
    hits = injection_score(text)
    if hits:
        raise GuardrailViolation("possible prompt injection blocked")
    clean, pii = mask_pii(text)
    return clean, {"pii_masked": pii}


def check_output(answer: str, citations: list) -> dict:
    """An answer that makes claims without citations is flagged as ungrounded."""
    has_cite = bool(re.search(r"\[\d+\]", answer)) or "not found" in answer.lower()
    return {"has_citation": has_cite, "n_citations": len(citations)}
