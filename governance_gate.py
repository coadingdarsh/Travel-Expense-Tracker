"""
governance_gate.py
==================
Part 1 — Core policy logic  (governance_gate function)
Part 2 — Execution tool     (execute_action function)

These two functions are the foundation of the whole pipeline.
governance_gate decides; execute_action acts — never the other way around.
"""

# datetime is used in execute_action to stamp every confirmed execution
from datetime import datetime


# ---------------------------------------------------------------------------
# DEFAULT_POLICY
# ---------------------------------------------------------------------------
# This dict is the single source of truth for every policy rule.
# It is imported by both agent_pipeline.py and app.py so the same
# constraints apply everywhere without duplication.
DEFAULT_POLICY = {
    # Any request above this dollar value must be escalated for human review
    "max_amount": 10000,
    # Only requests originating from these known systems are permitted
    "allowed_sources": ["finance_ops", "ap_system"],
    # Requests made by anyone in this list always require a manual approval step
    "roles_requiring_approval": ["analyst"],
}


# ---------------------------------------------------------------------------
# governance_gate
# ---------------------------------------------------------------------------
def governance_gate(amount: float, source: str, role: str, policy: dict) -> dict:
    """Evaluate a proposed action against a policy dict.

    Rules are checked independently so every triggered reason is captured.
    Decision priority (highest wins): blocked > escalated > approved.

    Args:
        amount: The monetary value of the proposed action.
        source: The system or channel the request originates from.
        role:   The organisational role of the person making the request.
        policy: Dict containing max_amount, allowed_sources,
                and roles_requiring_approval keys.

    Returns:
        {"decision": str, "reasons": list[str]}
        decision is one of "approved", "escalated", or "blocked".
    """
    # Start optimistic — innocent until proven otherwise
    decision = "approved"
    # Accumulate every triggered reason; multiple rules can fire at once
    reasons = []

    # --- Rule 1: amount cap ---
    # If the dollar value exceeds the policy ceiling, escalate for human sign-off
    if amount > policy["max_amount"]:
        decision = "escalated"                      # upgrade from approved to escalated
        reasons.append(
            # State both the actual amount and the limit so the reviewer has full context
            f"Amount ${amount:,.2f} exceeds the policy limit of ${policy['max_amount']:,.2f}."
        )

    # --- Rule 2: source allowlist ---
    # If the originating system is not on the approved list, the request is blocked outright
    if source not in policy["allowed_sources"]:
        decision = "blocked"                        # blocked overrides escalated
        reasons.append(
            # Name the invalid source and list the valid ones so the user knows what to use
            f"Source '{source}' is not in the list of allowed sources "
            f"({', '.join(policy['allowed_sources'])})."
        )

    # --- Rule 3: role-based escalation ---
    # Analysts always need a second pair of eyes; skip this check only when already blocked
    # (we don't downgrade "blocked" to "escalated" — blocked is the stronger outcome)
    if role in policy["roles_requiring_approval"] and decision != "blocked":
        decision = "escalated"                      # role alone is enough to escalate
        reasons.append(
            f"Role '{role}' requires manual approval before execution."
        )

    # --- No-flag path ---
    # If none of the rules triggered, the action is clean — say so explicitly
    if not reasons:
        reasons.append("Within policy — no flags.")

    # Return a consistent dict that callers can pattern-match on "decision"
    return {"decision": decision, "reasons": reasons}


# ---------------------------------------------------------------------------
# execute_action
# ---------------------------------------------------------------------------
def execute_action(action: str, amount: float, source: str) -> dict:
    """Simulate carrying out a pre-approved action (e.g. posting a journal entry).

    This function performs NO policy check whatsoever — it trusts that the
    caller (agent or Streamlit app) has already passed the governance gate.
    It only records what happened and when.

    Args:
        action: Short human-readable description of the action being executed.
        amount: The monetary amount involved in the action.
        source: The system that originated the request.

    Returns:
        A confirmation dict with status, action, amount, source, and timestamp.
    """
    return {
        "status": "executed",                               # fixed marker so callers can assert success
        "action": action,                                   # echo the action name for the audit trail
        "amount": amount,                                   # echo the amount for the audit trail
        "source": source,                                   # echo the source for the audit trail
        "timestamp": datetime.now().isoformat(timespec="seconds"),  # ISO-8601 wall-clock time of execution
    }
