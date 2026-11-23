"""Synthetic finance data: policy documents, vendor contracts, a SQLite ledger,
a labelled QA set (for RAG evaluation) and SFT / DPO datasets.

Everything is generated deterministically so results are reproducible.
"""
from __future__ import annotations

import json
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

AS_OF = "2025-06-30"  # "today" for the ledger, keeps overdue logic deterministic

VENDORS = [
    ("Acme Cloud", "software"),
    ("Globex Travel", "travel"),
    ("Initech Office", "office"),
    ("Umbrella Legal", "legal"),
    ("Stark Hardware", "hardware"),
    ("Wayne Consulting", "consulting"),
]
CATEGORIES = sorted({c for _, c in VENDORS})

DESCRIPTIONS = {
    "software": ["annual SaaS subscription renewal", "cloud hosting usage for the month",
                 "API platform license seats", "monitoring tool premium plan"],
    "travel": ["flight tickets for the sales offsite", "hotel booking for client visit",
               "airport transfer and car rental", "conference travel package"],
    "office": ["printer paper and toner supplies", "ergonomic chairs for new hires",
               "pantry snacks and coffee", "office stationery bulk order"],
    "legal": ["contract review retainer", "trademark filing fees",
              "employment law advisory hours", "compliance audit support"],
    "hardware": ["laptop purchase batch", "network switches and cabling",
                 "external monitors for the design team", "server rack replacement parts"],
    "consulting": ["quarterly strategy workshop", "process automation assessment",
                   "data migration consulting hours", "finance transformation advisory"],
}

POLICIES = {
    "approval_policy.txt": (
        "Payment Approval Policy. Version 3.2, effective January 2025.\n"
        "Section 1. Invoices under $1,000 may be approved by the accounts payable clerk.\n"
        "Section 2. Invoices from $1,000 up to and including $5,000 require approval from the finance manager.\n"
        "Section 3. Invoices above $5,000 require approval from the CFO.\n"
        "Section 4. No employee may approve an invoice for a vendor managed by their own team without a second approver.\n"
        "Section 5. All approvals must be recorded in the audit log with the approver name and timestamp."
    ),
    "payment_terms.txt": (
        "Vendor Payment Terms. Standard payment terms are net 30 days from the invoice date. "
        "An early payment discount of 2 percent applies if Acme Cloud is paid within 10 days. "
        "A late fee of 1.5 percent per month applies to overdue balances for Globex Travel. "
        "Invoices more than 60 days overdue must be escalated to the controller. "
        "Payments are processed every Tuesday and Friday."
    ),
    "travel_policy.txt": (
        "Travel and Expense Policy. Employees must book economy class for flights under 6 hours. "
        "Hotel stays are capped at $220 per night in major cities and $150 elsewhere. "
        "Meals are reimbursed up to $60 per day. "
        "Travel expenses must be submitted within 14 days of return. "
        "Receipts are required for any expense above $25."
    ),
    "records_and_privacy.txt": (
        "Records and Privacy Policy. Financial records must be retained for 7 years. "
        "Invoices and contracts are stored in the document management system. "
        "Personal data must be masked in AI-assisted tools. "
        "Corporate cards have a monthly limit of $10,000 per cardholder."
    ),
    "acme_cloud_contract.txt": (
        "Master Services Agreement with Acme Cloud. Term is 24 months starting March 2024. "
        "Monthly fee is $4,200 with a 3 percent annual price escalation. "
        "Termination requires 90 days written notice. "
        "Service level uptime is 99.9 percent; credits of 10 percent of the monthly fee apply "
        "if uptime falls below 99.5 percent."
    ),
    "globex_travel_contract.txt": (
        "Travel Services Agreement with Globex Travel. Term is 12 months. "
        "The booking fee is $25 per ticket. Cancellation within 24 hours is free. "
        "A corporate discount of 8 percent applies to published fares."
    ),
    "wayne_consulting_contract.txt": (
        "Consulting engagement with Wayne Consulting. The rate is $180 per hour, "
        "capped at 400 hours per quarter. Deliverables are accepted within 10 business days. "
        "Intellectual property transfers to the company upon final payment."
    ),
}

VENDOR_CSV = (
    "vendor,category,payment_terms,contact\n"
    "Acme Cloud,software,net 30,billing@acme.example\n"
    "Globex Travel,travel,net 15,ap@globex.example\n"
    "Initech Office,office,net 30,orders@initech.example\n"
    "Umbrella Legal,legal,net 45,finance@umbrella.example\n"
    "Stark Hardware,hardware,net 30,sales@stark.example\n"
    "Wayne Consulting,consulting,net 30,pm@wayne.example\n"
)

# question, short reference answer, gold keywords that a correct context/answer must contain
QA = [
    ("Who must approve an invoice of $7,500?", "The CFO must approve invoices above $5,000.", ["CFO"]),
    ("What are the standard vendor payment terms?", "Standard payment terms are net 30 days.", ["net 30"]),
    ("What is the hotel cap per night in major cities?", "Hotels are capped at $220 per night in major cities.", ["$220"]),
    ("What notice is needed to terminate the Acme Cloud agreement?", "Termination requires 90 days written notice.", ["90 days"]),
    ("How long must financial records be retained?", "Financial records must be retained for 7 years.", ["7 years"]),
    ("What late fee applies to Globex Travel?", "A late fee of 1.5 percent per month applies.", ["1.5 percent"]),
    ("What is the hourly rate for Wayne Consulting?", "The rate is $180 per hour.", ["$180"]),
    ("On which days are payments processed?", "Payments are processed every Tuesday and Friday.", ["Tuesday", "Friday"]),
    ("What is the uptime SLA for Acme Cloud?", "Service level uptime is 99.9 percent.", ["99.9"]),
    ("Within how many days must travel expenses be submitted?", "Within 14 days of return.", ["14 days"]),
    ("What early payment discount does Acme Cloud offer?", "2 percent if paid within 10 days.", ["2 percent"]),
    ("Who approves invoices between $1,000 and $5,000?", "The finance manager approves them.", ["finance manager"]),
]
UNANSWERABLE = ["What is the CEO's salary?", "Which city hosts the company holiday party?"]


# ------------------------------------------------------------------ documents
def write_documents(root: Path) -> Path:
    docs = Path(root) / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    for name, text in POLICIES.items():
        (docs / name).write_text(text, encoding="utf-8")
    (docs / "vendors.csv").write_text(VENDOR_CSV, encoding="utf-8")
    return docs


# --------------------------------------------------------------------- ledger
def make_invoices(n: int = 300, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    base = date(2025, 1, 1)
    rows = []
    for i in range(n):
        vendor, cat = rng.choice(VENDORS)
        amount = round(rng.choice([rng.uniform(80, 900), rng.uniform(900, 5200), rng.uniform(5200, 18000)]), 2)
        inv_date = base + timedelta(days=rng.randint(0, 175))
        due = inv_date + timedelta(days=30)
        rows.append({
            "id": f"INV-2025-{i + 1:04d}", "vendor": vendor, "category": cat,
            "description": rng.choice(DESCRIPTIONS[cat]), "amount": amount,
            "invoice_date": inv_date.isoformat(), "due_date": due.isoformat(),
            "status": "paid" if rng.random() < 0.5 else "pending",
        })
    return rows


def invoice_text(r: dict) -> str:
    return (f"Invoice {r['id']} from {r['vendor']} dated {r['invoice_date']} for "
            f"${r['amount']:,.2f}: {r['description']}. Due {r['due_date']}.")


def create_ledger(db_path: Path, n: int = 300) -> list[dict]:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    rows = make_invoices(n)
    con = sqlite3.connect(db_path)
    con.executescript(
        """
        CREATE TABLE invoices(id TEXT PRIMARY KEY, vendor TEXT, category TEXT, description TEXT,
            amount REAL, invoice_date TEXT, due_date TEXT, status TEXT);
        CREATE TABLE payments(id INTEGER PRIMARY KEY AUTOINCREMENT, invoice_id TEXT, amount REAL,
            paid_at TEXT, approved_by TEXT);
        CREATE TABLE audit_log(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, actor TEXT,
            action TEXT, detail TEXT);
        CREATE TABLE memory(key TEXT PRIMARY KEY, value TEXT);
        """
    )
    con.executemany(
        "INSERT INTO invoices VALUES(:id,:vendor,:category,:description,:amount,:invoice_date,:due_date,:status)",
        rows,
    )
    con.commit()
    con.close()
    return rows


# ------------------------------------------------------- LM / SFT / DPO data
def pretrain_corpus(invoices: list[dict]) -> str:
    parts = list(POLICIES.values()) + [VENDOR_CSV]
    parts += [invoice_text(r) for r in invoices[:200]]
    return "\n".join(parts)


def sft_examples(invoices: list[dict], seed: int = 3) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(prompt, response) pairs -> (train, held-out). Two task types:
    domain QA, and invoice-text -> JSON extraction (the held-out invoices are unseen)."""
    rng = random.Random(seed)
    qa = [(f"Question: {q}", a) for q, a, _ in QA]

    def ext(rows):
        return [(f"Extract fields: {invoice_text(r)}",
                 json.dumps({"invoice_id": r["id"], "vendor": r["vendor"], "amount": r["amount"]})) for r in rows]

    # held-out invoices (270+) never appear in pretraining text (first 200) or SFT train (first 150)
    held = ext(invoices[270:])[:30] + qa[-2:]
    train = ext(invoices[:150]) + qa[:-2] * 3
    rng.shuffle(train)
    return train, held


def dpo_pairs() -> list[tuple[str, str, str]]:
    """(prompt, chosen, rejected): rejected answers are plausible-but-wrong (hallucinations)."""
    wrong = ["I am not sure, but it is probably 45 days.", "The CEO decides everything.",
             "That is not covered anywhere.", "It depends on the weather."]
    pairs = []
    for i, (q, a, _) in enumerate(QA):
        pairs.append((f"Question: {q}", a, wrong[i % len(wrong)]))
    return pairs
