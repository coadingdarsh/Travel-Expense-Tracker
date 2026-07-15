"""
Part 4 — Streamlit demo app
Run with: streamlit run app.py

Requires GOOGLE_API_KEY set in st.secrets (Streamlit Cloud) or as an environment variable.
"""

import asyncio
import os
import uuid
from datetime import datetime

import streamlit as st
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from governance_gate import DEFAULT_POLICY, execute_action, governance_gate

# Load API key from Streamlit secrets (Cloud) or environment variable (local)
api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Governance Gate + Execution Pipeline",
    page_icon="🔐",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state — audit log
# ---------------------------------------------------------------------------
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []

# ---------------------------------------------------------------------------
# ADK agent (shared, stateless — re-used across button presses)
# ---------------------------------------------------------------------------

# Tool wrappers used by the ADK agent
def tool_governance_gate(amount: float, source: str, role: str) -> dict:
    """Check whether a proposed action is allowed under the active policy.

    Args:
        amount: Monetary amount of the proposed action.
        source: Originating system identifier.
        role:   Requester's role.
    """
    return governance_gate(amount, source, role, DEFAULT_POLICY)


def tool_execute_action(action: str, amount: float, source: str) -> dict:
    """Execute an already-approved action. ONLY call after governance_gate returns 'approved'.

    Args:
        action: Short description of the action.
        amount: Monetary amount.
        source: Originating system.
    """
    return execute_action(action, amount, source)


SYSTEM_INSTRUCTION = """
You are the Governance Gate + Execution Pipeline agent.

Your job is to:
1. Parse the user's natural-language action request to extract:
   - action  : a short description of what is being requested
   - amount  : the monetary amount (numeric only, no $ sign)
   - source  : the originating system identifier (e.g. ap_system, finance_ops)
   - role    : the requester's role (e.g. manager, analyst, director)

2. Call tool_governance_gate(amount, source, role) FIRST, always.

3. Based on the decision:
   - "approved"  → call tool_execute_action(action, amount, source), then report:
       EXECUTED: [action details]
       Confirmation: [full confirmation dict]
   - "escalated" → DO NOT call execute_action. Report:
       ESCALATED: [explain each reason]
       Action is on hold and must be reviewed by a senior approver or finance controller.
   - "blocked"   → DO NOT call execute_action. Report:
       BLOCKED: [explain each reason]
       The action cannot proceed under the current policy.

Always lead with the outcome word (EXECUTED / ESCALATED / BLOCKED) on the first line,
then provide the reasoning in plain English. Never call execute_action for escalated
or blocked decisions.
""".strip()


@st.cache_resource
def get_agent():
    return LlmAgent(
        name="governance_agent",
        model="gemini-2.0-flash",
        instruction=SYSTEM_INSTRUCTION,
        tools=[tool_governance_gate, tool_execute_action],
    )


def run_agent_query(prompt: str) -> str:
    """Run a single prompt through the ADK agent, return final text reply."""
    session_service = InMemorySessionService()
    app_name = "governance_gate_app"

    session = asyncio.run(
        session_service.create_session(app_name=app_name, user_id="streamlit_user")
    )

    runner = Runner(
        agent=get_agent(),
        app_name=app_name,
        session_service=session_service,
    )

    new_message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=prompt)],
    )

    final_response = ""
    for event in runner.run(
        user_id="streamlit_user",
        session_id=session.id,
        new_message=new_message,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response = event.content.parts[0].text
    return final_response


# ---------------------------------------------------------------------------
# Direct pipeline (no LLM) used to populate the audit log reliably
# ---------------------------------------------------------------------------

def run_direct_pipeline(action: str, amount: float, source: str, role: str) -> dict:
    """Run gate + conditional execute directly (no LLM) and return an audit record."""
    gate_result = governance_gate(amount, source, role, DEFAULT_POLICY)
    decision = gate_result["decision"]
    reasons = gate_result["reasons"]

    executed = False
    confirmation = None

    if decision == "approved":
        confirmation = execute_action(action, amount, source)
        executed = True

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "amount": amount,
        "source": source,
        "role": role,
        "decision": decision,
        "reasons": " | ".join(reasons),
        "executed": "Yes" if executed else "No",
        "_confirmation": confirmation,
        "_gate_result": gate_result,
    }


# ---------------------------------------------------------------------------
# Sidebar — active policy
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔐 Active Policy")
    st.caption(
        "This gate decides whether an action is allowed. "
        "Nothing executes without passing through it first."
    )
    st.divider()
    st.metric("Max Amount", f"${DEFAULT_POLICY['max_amount']:,}")
    st.markdown("**Allowed Sources**")
    for s in DEFAULT_POLICY["allowed_sources"]:
        st.markdown(f"- `{s}`")
    st.markdown("**Roles Requiring Approval**")
    for r in DEFAULT_POLICY["roles_requiring_approval"]:
        st.markdown(f"- `{r}`")


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
st.title("Governance Gate + Execution Pipeline")
st.markdown(
    "Submit a business action request. The governance gate evaluates it first — "
    "the executor only runs if the gate approves."
)

with st.form("request_form"):
    col1, col2 = st.columns(2)
    with col1:
        action_name = st.text_input(
            "Action Name",
            value="Post journal entry",
            help="A short description of the action to perform.",
        )
        amount = st.number_input(
            "Amount ($)",
            min_value=0.0,
            step=500.0,
            value=4000.0,
            format="%.2f",
        )
    with col2:
        source = st.selectbox(
            "Source System",
            options=["finance_ops", "ap_system", "spreadsheet_upload", "manual_entry"],
        )
        role = st.selectbox(
            "Requester Role",
            options=["analyst", "manager", "director"],
            index=1,
        )

    use_agent = st.checkbox(
        "Also show ADK agent response (requires GOOGLE_API_KEY)",
        value=False,
    )

    submitted = st.form_submit_button("Submit Request", type="primary")

if submitted:
    # Always run the direct pipeline for reliable audit + UI outcome
    record = run_direct_pipeline(action_name, amount, source, role)
    decision = record["decision"]
    reasons_text = record["reasons"]
    confirmation = record["_confirmation"]

    # Display outcome box
    if decision == "approved":
        st.success(
            f"✅ **EXECUTED** — {action_name}\n\n"
            f"**Confirmation:**\n"
            f"- Status: {confirmation['status']}\n"
            f"- Action: {confirmation['action']}\n"
            f"- Amount: ${confirmation['amount']:,.2f}\n"
            f"- Source: {confirmation['source']}\n"
            f"- Timestamp: {confirmation['timestamp']}"
        )
    elif decision == "escalated":
        st.warning(
            f"⚠️ **ESCALATED** — {action_name}\n\n"
            f"**Reason(s):** {reasons_text}\n\n"
            "Action is on hold. A senior approver or finance controller must review."
        )
    else:  # blocked
        st.error(
            f"🚫 **BLOCKED** — {action_name}\n\n"
            f"**Reason(s):** {reasons_text}\n\n"
            "This action cannot proceed under the current policy."
        )

    # ADK agent response (optional)
    if use_agent:
        with st.expander("ADK Agent Response", expanded=True):
            with st.spinner("Running ADK agent…"):
                try:
                    prompt = (
                        f"{action_name} of ${amount:,.2f} from {source}, "
                        f"requested by a {role}."
                    )
                    agent_reply = run_agent_query(prompt)
                    st.markdown(agent_reply)
                except Exception as exc:
                    st.error(f"Agent error: {exc}")

    # Append to audit log (newest first at display time)
    st.session_state.audit_log.insert(0, record)

# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Audit Trail")

if not st.session_state.audit_log:
    st.caption("No requests submitted yet.")
else:
    import pandas as pd

    display_cols = ["timestamp", "action", "amount", "source", "role",
                    "decision", "reasons", "executed"]
    df = pd.DataFrame(st.session_state.audit_log)[display_cols]
    df["amount"] = df["amount"].apply(lambda x: f"${x:,.2f}")

    st.dataframe(df, use_container_width=True, hide_index=True)
