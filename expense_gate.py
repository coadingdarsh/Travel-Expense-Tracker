"""
expense_gate.py
===============
Policy evaluation and execution gate for Trip Advocate.

expense_gate  — decides if an expense is approved or flagged
execute_expense — records a pre-approved expense (no policy check)
"""

from datetime import datetime

# ---------------------------------------------------------------------------
# DEFAULT_POLICY
# ---------------------------------------------------------------------------
# Per-category spending limits. Imported by app.py and expense_pipeline.py.
DEFAULT_POLICY = {
    "meals":     75.0,   # per meal
    "lodging":   250.0,  # per night
    "transport": 150.0,  # per segment
    "other":     100.0,  # catch-all
}


# ---------------------------------------------------------------------------
# expense_gate
# ---------------------------------------------------------------------------
def expense_gate(amount: float, category: str, context: str, policy: dict) -> dict:
    """Evaluate a proposed expense against the policy limits.

    Args:
        amount:   The dollar amount of the expense.
        category: One of meals / lodging / transport / other.
        context:  The traveler's stated reason (used downstream for justification).
        policy:   Dict mapping category names to their spending limits.

    Returns:
        {
            "decision": "approved" | "flagged",
            "limit":    float — the applicable policy limit,
            "over_by":  float — how much over the limit (0 if approved),
        }
    """
    limit   = policy.get(category, policy.get("other", 100.0))
    over_by = max(0.0, amount - limit)

    decision = "approved" if over_by == 0 else "flagged"

    return {
        "decision": decision,
        "limit":    limit,
        "over_by":  round(over_by, 2),
    }


# ---------------------------------------------------------------------------
# execute_expense
# ---------------------------------------------------------------------------
def execute_expense(title: str, amount: float, category: str) -> dict:
    """Record an already-approved expense. Performs NO policy check.

    Args:
        title:    Short description of the expense.
        amount:   Dollar amount.
        category: Expense category.

    Returns:
        Confirmation dict with status and timestamp.
    """
    return {
        "status":    "recorded",
        "title":     title,
        "amount":    amount,
        "category":  category,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
