"""
expense_pipeline.py
===================
IBM watsonx.ai-backed LLM helpers for Trip Advocate.

generate_justification    — one-sentence advocate justification for a flagged expense
generate_pretrip_briefing — spoken-style pre-trip heads-up
generate_debrief          — warm end-of-trip recap
parse_expense_request     — parse a natural-language expense description into fields

All functions fail gracefully: if watsonx credentials are missing or the
call fails, they return a safe fallback string instead of crashing.
"""

from __future__ import annotations

import os
import re

import requests

# ---------------------------------------------------------------------------
# watsonx.ai helpers
# ---------------------------------------------------------------------------
_IAM_URL    = "https://iam.cloud.ibm.com/identity/token"
_token_cache: dict = {}


def _get_iam_token(api_key: str) -> str:
    if _token_cache.get("token"):
        return _token_cache["token"]
    resp = requests.post(
        _IAM_URL,
        data={"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": api_key},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    resp.raise_for_status()
    token = resp.json()["access_token"]
    _token_cache["token"] = token
    return token


def _generate(prompt: str, max_new_tokens: int = 300) -> str:
    """Call watsonx.ai and return generated text. Returns '' on any failure."""
    api_key    = os.environ.get("WATSONX_API_KEY", "")
    project_id = os.environ.get("WATSONX_PROJECT_ID", "")
    base_url   = os.environ.get("WATSONX_URL", "").rstrip("/")
    model      = os.environ.get("WATSONX_MODEL", "meta-llama/llama-3-3-70b-instruct")

    if not (api_key and project_id and base_url):
        return ""   # credentials not configured — caller will use fallback

    url = f"{base_url}/ml/v1/text/generation?version=2023-05-29"

    for attempt in range(2):
        try:
            token = _get_iam_token(api_key)
            payload = {
                "model_id": model,
                "project_id": project_id,
                "input": prompt,
                "parameters": {
                    "decoding_method": "greedy",
                    "max_new_tokens": max_new_tokens,
                    "repetition_penalty": 1.1,
                },
            }
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            if resp.status_code == 401 and attempt == 0:
                _token_cache.clear()
                continue
            resp.raise_for_status()
            return resp.json()["results"][0]["generated_text"].strip()
        except Exception:
            return ""

    return ""


def _first_sentence(text: str) -> str:
    """Return the first clean sentence from a model response."""
    # Strip preamble labels
    for label in ("Justification:", "Answer:", "Response:", "Briefing:", "Debrief:"):
        if text.lower().startswith(label.lower()):
            text = text[len(label):].strip()
    sentence = text.split("\n")[0].strip()
    sentence = re.sub(r"\s*\([^)]{0,120}\)\.?$", "", sentence).strip().rstrip(".")
    return sentence + "." if sentence else ""


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def generate_justification(
    amount: float, category: str, context: str, limit: float, over_by: float
) -> str:
    """Generate a one-sentence advocate justification for a flagged expense."""
    prompt = (
        f"An expense of ${amount:.2f} for {category} exceeded the policy limit "
        f"of ${limit:.2f} by ${over_by:.2f}. "
        f"The traveler's stated reason was: '{context}'. "
        "Write a single professional sentence justifying this expense that a manager "
        "would find reasonable. Base it only on the traveler's context — do not invent facts. "
        "Begin directly, no preamble.\n\nJustification:"
    )
    raw = _generate(prompt, max_new_tokens=120)
    if raw:
        return _first_sentence(raw)
    # Fallback — no LLM
    return (
        f"This ${amount:.2f} {category} expense exceeded the ${limit:.2f} policy limit "
        f"by ${over_by:.2f}; traveler context: \"{context}\"."
    )


def generate_pretrip_briefing(destination: str, past_trip_note: str) -> str:
    """Generate a spoken-style pre-trip briefing paragraph."""
    prompt = (
        f"You are a friendly travel advisor helping a business traveler prepare for "
        f"a trip to {destination}.\n\n"
        f"Here is a note from a past trip: {past_trip_note}\n\n"
        "Write a short spoken-style briefing (3–5 sentences, conversational tone, "
        "no bullet points) that tells the traveler what typically goes over budget, "
        "what to watch for, and one practical tip. Speak directly as 'you'. "
        "Write flowing prose only — no lists.\n\nBriefing:"
    )
    raw = _generate(prompt, max_new_tokens=250)
    if raw:
        # Strip model artefacts
        for marker in ("```", "Note:", "Disclaimer:"):
            idx = raw.find(marker)
            if idx > 60:
                raw = raw[:idx].rstrip()
        return raw
    return (
        f"Heading to {destination}? Based on past trips, meals tend to be the category "
        "most likely to run over budget — especially when client dinners are involved. "
        "Keep receipts and add a quick note about why each meal happened. "
        "That context is all you need if anything gets flagged."
    )


def generate_debrief(trip_name: str, total: float, summaries: list[str]) -> str:
    """Generate a warm end-of-trip debrief."""
    summary_block = "\n".join(summaries)
    prompt = (
        f"You are writing a warm, friendly end-of-trip summary for a traveler "
        f"returning from '{trip_name}'. Total spend: ${total:.2f}.\n\n"
        f"Expense summaries:\n{summary_block}\n\n"
        "Write a short debrief (4–6 sentences) in a warm, conversational tone — "
        "like a friend on the traveler's side giving a quick recap. "
        "Mention the total spend, briefly acknowledge any flagged items by their "
        "justification (not as accusations), and end encouragingly. "
        "No bullet lists. Give it a punchy episode-style title on the first line "
        "(e.g. 'Chicago: The $11 Coffee Incident'), then a blank line, then the prose.\n\nDebrief:"
    )
    raw = _generate(prompt, max_new_tokens=350)
    if raw:
        for marker in ("```", "Best,", "Sincerely,"):
            idx = raw.find(marker)
            if idx > 60:
                raw = raw[:idx].rstrip()
        return raw
    flagged = [s for s in summaries if "flagged" in s]
    flag_note = f" {len(flagged)} item(s) were flagged but justified." if flagged else " Everything came in clean."
    return (
        f"{trip_name}: A Clean Run\n\n"
        f"You wrapped up '{trip_name}' with a total spend of ${total:.2f}.{flag_note} "
        "Great work keeping things documented — your notes make all the difference "
        "if anyone ever asks questions. Safe travels next time!"
    )


def parse_expense_request(text: str) -> dict:
    """Parse a natural-language expense description into structured fields.

    Returns a dict with keys: title, amount, category, context.
    On failure returns {"_parse_error": True}.
    """
    prompt = (
        "Extract the following fields from this expense description and return them "
        "as a plain Python dict with no extra text:\n"
        "  title    : short description (str)\n"
        "  amount   : numeric dollar amount (float, no $ sign)\n"
        "  category : one of meals / lodging / transport / other (str)\n"
        "  context  : the traveler's stated reason (str)\n\n"
        f"Description: {text}\n\n"
        "Return only the dict, nothing else. Example:\n"
        '{\"title\": \"Client dinner\", \"amount\": 85.0, \"category\": \"meals\", \"context\": \"contract discussion ran late\"}'
    )
    raw = _generate(prompt, max_new_tokens=150)
    if not raw:
        return {"_parse_error": True}
    try:
        # Extract the first {...} block
        match = re.search(r"\{[^}]+\}", raw, re.DOTALL)
        if not match:
            return {"_parse_error": True}
        import ast
        data = ast.literal_eval(match.group())
        # Validate required keys
        for key in ("title", "amount", "category", "context"):
            if key not in data:
                return {"_parse_error": True}
        data["amount"]   = float(data["amount"])
        data["category"] = str(data["category"]).lower()
        if data["category"] not in ("meals", "lodging", "transport", "other"):
            data["category"] = "other"
        return data
    except Exception:
        return {"_parse_error": True}
