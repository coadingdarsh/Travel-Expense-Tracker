"""
Trip Advocate — an expense tracker built to defend the traveler, not audit them.
Built on IBM watsonx.ai (meta-llama/llama-3-3-70b-instruct).

Run with: streamlit run app.py
Requires: pip install streamlit requests pandas
"""

from datetime import date

import pandas as pd
import streamlit as st

# Load IBM watsonx secrets from Streamlit Cloud if present
import os
for _key in ("WATSONX_API_KEY", "WATSONX_PROJECT_ID", "WATSONX_URL", "WATSONX_MODEL"):
    _val = st.secrets.get(_key)
    if _val:
        os.environ.setdefault(_key, _val)

from expense_gate import DEFAULT_POLICY, execute_expense, expense_gate
from expense_pipeline import (
    generate_debrief,
    generate_justification,
    generate_pretrip_briefing,
    parse_expense_request,
)
from external_apis import check_fare_reasonableness, get_weather_context

# ---------------------------------------------------------------------------
# Reliable, no-LLM-required direct pipeline
# ---------------------------------------------------------------------------
def run_direct_expense(title: str, amount: float, category: str, context: str) -> dict:
    gate_result = expense_gate(amount, category, context, DEFAULT_POLICY)
    decision = gate_result["decision"]

    confirmation = None
    justification = None

    if decision == "approved":
        confirmation = execute_expense(title, amount, category)
    else:
        justification = generate_justification(
            amount, category, context, gate_result["limit"], gate_result["over_by"]
        )

    return {
        "title": title,
        "amount": amount,
        "category": category,
        "context": context,
        "decision": decision,
        "limit": gate_result["limit"],
        "over_by": gate_result["over_by"],
        "justification": justification,
        "confirmation": confirmation,
    }


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "trip" not in st.session_state:
    st.session_state.trip = None
if "expenses" not in st.session_state:
    st.session_state.expenses = []
if "briefing" not in st.session_state:
    st.session_state.briefing = None
if "debrief" not in st.session_state:
    st.session_state.debrief = None

# ---------------------------------------------------------------------------
# Page config + design system
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Trip Advocate", page_icon="🧳", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #FAF9F6; }
    h1, h2, h3 { color: #1B2430; font-weight: 700; }

    .advocate-card {
        background-color: #FFFFFF;
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 2px 12px rgba(27, 36, 48, 0.06);
        margin-bottom: 16px;
        border: 1px solid rgba(27, 36, 48, 0.05);
    }

    .quote-card {
        background-color: rgba(42, 157, 143, 0.08);
        border-left: 3px solid #2A9D8F;
        border-radius: 12px;
        padding: 14px 18px;
        margin-top: 10px;
        font-style: italic;
        color: #1B2430;
    }

    .quote-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #2A9D8F;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-style: normal;
        margin-bottom: 4px;
    }

    .status-approved {
        display: inline-block;
        background-color: rgba(107, 144, 128, 0.15);
        color: #4A6D5C;
        padding: 4px 12px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    .status-flagged {
        display: inline-block;
        background-color: rgba(232, 163, 61, 0.18);
        color: #92651A;
        padding: 4px 12px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    .amount-mono { font-family: 'JetBrains Mono', monospace; font-weight: 500; }

    .stButton>button {
        background-color: #2A9D8F;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
    }
    .stButton>button:hover { background-color: #23857A; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — active policy + IBM badge
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🧳 Trip Advocate")
    st.caption("Built on IBM watsonx.ai — meta-llama/llama-3-3-70b-instruct")
    st.divider()
    st.markdown("**Active Policy**")
    st.markdown(f"- Meals: `${DEFAULT_POLICY['meals']:.0f}/meal`")
    st.markdown(f"- Lodging: `${DEFAULT_POLICY['lodging']:.0f}/night`")
    st.markdown(f"- Transport: `${DEFAULT_POLICY['transport']:.0f}/segment`")
    st.caption("This gate decides whether an expense needs a closer look. It doesn't accuse — it explains.")

st.title("Trip Advocate")
st.caption("Most expense tools are built to catch you. This one is built to back you up.")

# ---------------------------------------------------------------------------
# Step 1 — Trip setup
# ---------------------------------------------------------------------------
if st.session_state.trip is None:
    st.markdown("### Start a trip")
    st.markdown('<div class="advocate-card">', unsafe_allow_html=True)
    name = st.text_input("Trip name", placeholder="Chicago Sales Conference")
    destination = st.text_input("Destination", placeholder="Chicago, IL")
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Start date", value=date.today())
    with col2:
        end = st.date_input("End date", value=date.today())
    past_trip_note = st.text_area(
        "Optional — note about a past trip (used for your pre-trip briefing)",
        placeholder="Last trip to Chicago, meals ran over budget most nights due to client dinners.",
    )
    if st.button("Create Trip"):
        if name and destination:
            st.session_state.trip = {"name": name, "destination": destination,
                                      "start": str(start), "end": str(end)}
            with st.spinner("Generating your pre-trip briefing..."):
                st.session_state.briefing = generate_pretrip_briefing(
                    destination, past_trip_note or "No prior trip history available."
                )
            st.rerun()
        else:
            st.warning("Trip name and destination are required.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main trip view
# ---------------------------------------------------------------------------
else:
    trip = st.session_state.trip
    st.markdown(f"## {trip['name']}")
    st.caption(f"{trip['destination']}  ·  {trip['start']} → {trip['end']}")

    if st.session_state.briefing:
        st.markdown('<div class="quote-card">', unsafe_allow_html=True)
        st.markdown('<div class="quote-label">Your pre-trip briefing</div>', unsafe_allow_html=True)
        st.write(st.session_state.briefing)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Add an expense")

    entry_mode = st.radio(
        "How do you want to log it?",
        ["Quick form", "Natural language (watsonx parses it)"],
        horizontal=True,
    )

    st.markdown('<div class="advocate-card">', unsafe_allow_html=True)

    if entry_mode == "Quick form":
        col1, col2 = st.columns(2)
        with col1:
            title = st.text_input("What was it", placeholder="Client dinner")
            amount = st.number_input("Amount ($)", min_value=0.0, step=5.0, value=45.0)
        with col2:
            category = st.selectbox("Category", ["meals", "lodging", "transport", "other"])
        context = st.text_area(
            "In your own words — why?",
            placeholder="Dinner ran long, client wanted to finalize contract terms.",
            help="Your own explanation becomes your justification if anything's flagged.",
        )
        if st.button("Submit Expense"):
            record = run_direct_expense(title or category.title(), amount, category, context)
            st.session_state.expenses.insert(0, record)
            st.rerun()

    else:
        nl_request = st.text_area(
            "Describe the expense in one sentence",
            placeholder="Log a $210 client dinner, it ran long because we were finalizing contract terms.",
        )
        if st.button("Parse & Submit"):
            with st.spinner("watsonx is parsing your request..."):
                fields = parse_expense_request(nl_request)
            if fields.get("_parse_error"):
                st.error("Couldn't parse that request cleanly — try the quick form instead.")
            else:
                record = run_direct_expense(
                    fields["title"], fields["amount"], fields["category"], fields["context"]
                )
                st.session_state.expenses.insert(0, record)
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Expense list ----------
    if st.session_state.expenses:
        st.markdown("### This trip's expenses")
        total = sum(e["amount"] for e in st.session_state.expenses)
        st.markdown(f"**Total spend:** <span class='amount-mono'>${total:,.2f}</span>", unsafe_allow_html=True)

        for e in st.session_state.expenses:
            st.markdown('<div class="advocate-card">', unsafe_allow_html=True)
            badge = (
                '<span class="status-approved">Approved</span>'
                if e["decision"] == "approved"
                else '<span class="status-flagged">Flagged</span>'
            )
            st.markdown(
                f"**{e['title']}** — <span class='amount-mono'>${e['amount']:,.2f}</span> · {e['category'].title()}  {badge}",
                unsafe_allow_html=True,
            )
            if e["decision"] == "flagged":
                st.caption(f"${e['over_by']:.2f} over the ${e['limit']:.2f} policy limit")
                weather = get_weather_context(
                    trip.get("destination", ""), str(date.today())
                )
                if weather["available"] and weather["summary"] != "Weather was unremarkable that day.":
                    st.caption(f"🌦 {weather['summary']}")
                if e["category"] == "transport" and e.get("context"):
                    fare = check_fare_reasonableness("origin", "destination", e["amount"])
                    if fare["available"]:
                        st.caption(f"🚗 Fare check: {fare['verdict']}")
                st.markdown('<div class="quote-card">', unsafe_allow_html=True)
                st.markdown('<div class="quote-label">In your own words</div>', unsafe_allow_html=True)
                st.write(e["justification"])
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ---------- Audit trail table ----------
        st.markdown("### Audit Trail")
        df = pd.DataFrame(st.session_state.expenses)[["title", "amount", "category", "decision", "over_by"]]
        df["amount"] = df["amount"].apply(lambda x: f"${x:,.2f}")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # ---------- End of trip debrief ----------
        st.markdown("### Wrap up the trip")
        if st.button("Generate Trip Debrief"):
            summaries = [
                f"- {e['title']} (${e['amount']:.2f}, {e['category']}): "
                + (f"flagged — {e['justification']}" if e["decision"] == "flagged" else "approved, no issues")
                for e in st.session_state.expenses
            ]
            with st.spinner("Putting together your recap..."):
                st.session_state.debrief = generate_debrief(trip["name"], total, summaries)

        if st.session_state.debrief:
            st.markdown('<div class="quote-card">', unsafe_allow_html=True)
            st.markdown('<div class="quote-label">Trip debrief</div>', unsafe_allow_html=True)
            st.write(st.session_state.debrief)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No expenses yet — add your first one above to get started.")

    st.divider()
    if st.button("Reset / Start New Trip"):
        st.session_state.trip = None
        st.session_state.expenses = []
        st.session_state.briefing = None
        st.session_state.debrief = None
        st.rerun()
