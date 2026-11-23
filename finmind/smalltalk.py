"""Conversational layer: greetings, thanks, 'who are you', help -- answered directly instead of
being sent to document retrieval (which would only say 'not found'). Only short messages that
match whole-phrase patterns are intercepted, so real questions always reach the RAG pipeline."""
from __future__ import annotations

import random
import re

CAPABILITIES = (
    "I can help you with:\n"
    "• Policy and contract questions, with citations (e.g. “Who must approve an invoice of $7,500?”)\n"
    "• Data questions on the ledger (open the Payment agent for “How many invoices are overdue?”)\n"
    "• Paying overdue invoices, with your approval first\n"
    "• Turning invoice text into structured data (Invoice extraction)"
)

_GREET = r"(hi+|hello+|hey+|hiya|howdy|yo|sup|greetings|good (morning|afternoon|evening)|namaste|vanakkam)"
_RULES = [
    (rf"^{_GREET}( there| team| finmind| assistant| bot)?[\s!.,?]*$",
     ["Hi there! 👋 I'm FinMind, your finance assistant. What can I help you with today?",
      "Hello! I'm FinMind. Ask me about approval limits, vendor contracts, payment terms or travel policy."]),
    (r"^how are you( doing| today)?[\s!.?]*$|^how('s| is) it going[\s!.?]*$|^what'?s up[\s!.?]*$",
     ["I'm doing well, thanks for asking! Ready to dig into your finance policies whenever you are."]),
    (r"^(thanks|thank you|thx|ty|thanks a lot|thank you so much|cheers)[\s!.]*$",
     ["You're welcome! Let me know if there's anything else."]),
    (r"^(bye|goodbye|see you|see ya|cya|good night)[\s!.]*$", ["Goodbye! Come back any time."]),
    (r"^(ok|okay|cool|great|nice|got it|alright|sure)[\s!.]*$", ["👍 Anything else I can look up for you?"]),
    (r"^(who are you|what are you|what is finmind|what's your name|introduce yourself)[\s?!.]*$",
     ["I'm FinMind, a finance-operations assistant. I answer from your policy and contract documents, "
      "query the ledger, and run payments only after a human approves.\n\n" + CAPABILITIES]),
    (r"^(help|what can you do|how can you help( me)?|what do you do|how do i use this|options)[\s?!.]*$",
     [CAPABILITIES]),
]


def smalltalk(text: str) -> str | None:
    t = re.sub(r"\s+", " ", text.strip().lower())
    if not t or len(t.split()) > 6:
        return None
    for pattern, replies in _RULES:
        if re.match(pattern, t):
            return random.choice(replies)
    return None
