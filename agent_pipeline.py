"""
agent_pipeline.py
=================
Part 3 — ADK agent pipeline

Wires governance_gate and execute_action into Google ADK as native Python
tools, defines the LlmAgent with a strict system prompt, and exposes a
run_query() helper that any caller can use to get a plain-text response.

Running this file directly ( python agent_pipeline.py ) executes the three
spec test cases and prints the agent's response for each.

Requires GOOGLE_API_KEY set in the environment (or GOOGLE_GENAI_USE_VERTEXAI=true
for Vertex AI credentials).
"""

# asyncio is needed to call the async InMemorySessionService.create_session
# from synchronous code (runner.run itself handles the rest)
import asyncio

# Standard ADK building blocks
from google.adk.agents import LlmAgent           # the LLM-backed agent class
from google.adk.runners import Runner            # orchestrates agent + session I/O
from google.adk.sessions import InMemorySessionService  # ephemeral in-process session store

# genai_types gives us Content + Part — the message envelope ADK expects
from google.genai import types as genai_types

# Import the pure-Python core functions and the shared policy dict
from governance_gate import DEFAULT_POLICY, execute_action, governance_gate


# ---------------------------------------------------------------------------
# Tool wrappers
# ---------------------------------------------------------------------------
# ADK requires tools to be plain Python callables.  Docstrings + type
# annotations are parsed by ADK to auto-generate the tool schema sent to
# the model, so every arg and return must be documented clearly.

def tool_governance_gate(amount: float, source: str, role: str) -> dict:
    """Check whether a proposed action is allowed under the active policy.

    The tool delegates directly to governance_gate() with DEFAULT_POLICY
    so the agent never needs to know the policy structure itself.

    Args:
        amount: The monetary amount of the proposed action (numeric, e.g. 4000).
        source: The originating system (e.g. 'ap_system', 'finance_ops').
        role:   The requester's role (e.g. 'manager', 'analyst').

    Returns:
        A dict with 'decision' ('approved', 'escalated', or 'blocked')
        and 'reasons' (list of plain-English strings).
    """
    # Pass the hardcoded DEFAULT_POLICY — the agent doesn't pick the policy
    return governance_gate(amount, source, role, DEFAULT_POLICY)


def tool_execute_action(action: str, amount: float, source: str) -> dict:
    """Execute an already-approved action and return a confirmation record.

    The system prompt instructs the model to ONLY call this tool when
    tool_governance_gate returned 'approved'.  The function itself performs
    no policy check — it trusts the agent to gate-keep correctly.

    Args:
        action: A short description of the action being performed.
        amount: The monetary amount.
        source: The originating system.

    Returns:
        A confirmation dict with status, action, amount, source, and timestamp.
    """
    # Thin wrapper so the ADK tool schema is decoupled from the core function signature
    return execute_action(action, amount, source)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
# The instruction string defines the agent's entire decision logic in plain
# English.  It is deliberately detailed so the model never has to guess:
#   1. parse the request
#   2. call the gate tool first
#   3. branch on the decision — execute only on "approved"
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
""".strip()  # strip() removes the leading newline so the model sees a clean prompt


# ---------------------------------------------------------------------------
# Agent instance
# ---------------------------------------------------------------------------
# A single module-level LlmAgent is constructed once and reused for all
# queries.  It is stateless — all conversational state lives in the session.
governance_agent = LlmAgent(
    name="governance_agent",               # unique identifier used by ADK internally
    model="gemini-2.0-flash",             # fast, capable model; swap to gemini-1.5-pro for more complex parsing
    instruction=SYSTEM_INSTRUCTION,        # the system prompt defined above
    tools=[tool_governance_gate, tool_execute_action],  # the two callable tools the model may invoke
)


# ---------------------------------------------------------------------------
# run_query — synchronous convenience wrapper
# ---------------------------------------------------------------------------
def run_query(prompt: str) -> str:
    """Send a single natural-language prompt to the governance agent.

    Creates a fresh in-memory session for each call so queries are fully
    independent and there is no bleed-over between test cases or users.

    Args:
        prompt: The user's natural-language action request.

    Returns:
        The agent's final plain-text response (EXECUTED / ESCALATED / BLOCKED
        followed by reasoning).
    """
    # InMemorySessionService stores sessions in a dict — no database needed
    session_service = InMemorySessionService()

    # app_name groups sessions under one logical application namespace
    app_name = "governance_gate_app"

    # create_session is async; asyncio.run() bridges into sync context
    session = asyncio.run(
        session_service.create_session(app_name=app_name, user_id="demo_user")
    )

    # Runner ties together the agent, the session store, and the event loop
    runner = Runner(
        agent=governance_agent,      # the agent to invoke
        app_name=app_name,           # must match the app_name used when creating the session
        session_service=session_service,  # the same service instance that holds our session
    )

    # Wrap the raw string in a Content envelope so ADK knows this is a user turn
    new_message = genai_types.Content(
        role="user",                              # "user" role identifies the human side of the conversation
        parts=[genai_types.Part(text=prompt)],    # Part carries the actual text payload
    )

    # Accumulate the final agent response text across all events
    final_response = ""

    # runner.run() is a generator; it yields Event objects for every step
    # (tool calls, tool results, model tokens, final reply, etc.)
    for event in runner.run(
        user_id="demo_user",         # must match the user_id used when creating the session
        session_id=session.id,       # ties this run to the specific session we just created
        new_message=new_message,     # the user's message to process
    ):
        # is_final_response() is True only on the last text event the model emits
        if event.is_final_response():
            # Guard against events that carry no text content (e.g. tool-result-only events)
            if event.content and event.content.parts:
                # Take the first part's text as the reply (there is only one for plain responses)
                final_response = event.content.parts[0].text

    # Return the collected final text; empty string if the model produced no output
    return final_response


# ---------------------------------------------------------------------------
# CLI entry-point — run the three spec test cases
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Each tuple is (prompt_string, human_readable_label)
    test_cases = [
        (
            # TC-1: well within limits, valid source, non-escalating role → expect EXECUTED
            "Post a $4,000 journal entry from ap_system, requested by a manager.",
            "TC-1: $4k / ap_system / manager → expect EXECUTED",
        ),
        (
            # TC-2: over limit AND role requires approval → expect ESCALATED with two reasons
            "Process a $32,000 payment from finance_ops, requested by an analyst.",
            "TC-2: $32k / finance_ops / analyst → expect ESCALATED (2 reasons)",
        ),
        (
            # TC-3: invalid source → expect BLOCKED regardless of amount or role
            "Upload a $2,000 invoice from spreadsheet_upload, requested by a manager.",
            "TC-3: $2k / spreadsheet_upload / manager → expect BLOCKED",
        ),
    ]

    # Iterate over each case, print the label, and display the agent's reply
    for prompt, label in test_cases:
        print(f"\n{'='*70}")
        print(f"  {label}")                 # print the expected outcome for easy comparison
        print(f"  Prompt: {prompt}")        # print the exact input sent to the agent
        print(f"{'='*70}")
        reply = run_query(prompt)           # invoke the agent and collect its final response
        print(reply)                        # display the agent's EXECUTED / ESCALATED / BLOCKED output
