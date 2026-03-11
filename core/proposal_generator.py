"""Thor — DarkCode AI Proposal Generator.

Takes lead data from Viper → generates a 1-page proposal using Claude API.
Proposals are saved to ~/thor/data/proposals/ as markdown files.

Proposal structure (from DARKCODE_AGENCY_SPEC.md):
- THE PROBLEM: 1-2 sentences, their specific pain point
- THE SOLUTION: what we'll build, specific not vague
- DELIVERABLES: 3-5 items + documentation + 30-day support
- TIMELINE: X days from kickoff
- INVESTMENT: fixed price, 50/50 split
- WHY DARKCODE: 1-2 sentences
- NEXT STEP: reply to confirm + Stripe/PayPal payment link
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger("thor.proposal")

ET = timezone(timedelta(hours=-5))

PROPOSALS_DIR = Path(__file__).resolve().parent.parent / "data" / "proposals"
PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)

# Service tier details for proposal context
SERVICE_TIERS = {
    "starter": {
        "name": "Starter",
        "price_range": "$500-$1,500",
        "timeline": "3-7 days",
        "includes": "30 days post-launch support, documentation, handoff",
    },
    "growth": {
        "name": "Growth",
        "price_range": "$2,000-$5,000 + $500/mo retainer",
        "timeline": "2-4 weeks",
        "includes": "Analytics dashboard, multi-channel support, 30 days support, documentation",
    },
    "scale": {
        "name": "Scale",
        "price_range": "$5,000-$15,000 + $1,500/mo retainer",
        "timeline": "4-8 weeks",
        "includes": "Full audit, multiple integrations, training, documentation, ongoing support",
    },
}

PROPOSAL_SYSTEM = """You are a senior copywriter for DarkCode AI, a lean AI development studio.
You write proposals that convert using proven frameworks: AIDA, price anchoring,
social proof, loss aversion, and urgency. Every sentence earns the next sentence.
No filler. No jargon. Specific, quantified, and action-oriented.

COPYWRITING RULES:
- Open with the prospect's specific pain point — make them feel understood
- Quantify the cost of inaction (missed leads, wasted staff hours, lost revenue)
- Paint the after-state with specifics: hours saved, leads captured, response time
- Present pricing expensive-first (anchor high, make the recommended tier feel smart)
- Use social proof: "Built AI systems for dental offices, real estate agencies, and service businesses — 92% auto-resolution rate"
- Add scarcity: scope and pricing valid for 7 days
- Add loss aversion: "Without this, roughly X% of after-hours inquiries go unanswered"
- Single CTA with payment link — no friction
- NO phone calls or Calendly — closing happens over email"""

PROPOSAL_PROMPT = """Generate a 1-page proposal for DarkCode AI using the AIDA framework.

LEAD INFO:
- Client/Project: {client_name}
- Their need: {description}
- Budget hint: {budget}
- Source: {source}
- Service tier match: {service_tier} ({tier_details})

PROPOSAL FORMAT (follow EXACTLY):

DARKCODE AI — PROPOSAL FOR [CLIENT NAME]

[ATTENTION — 1-2 sentences]
Open with their specific pain point. Reference THEIR situation. Make them feel seen.
Quantify the cost: "Your team spends roughly X hours/week answering the same questions."

[INTEREST — 2-3 sentences]
An insight they haven't considered. What's the real cost of not solving this?
"Every after-hours visitor who can't get an answer is a lost lead. For most practices,
that's 30-40% of website traffic."

THE SOLUTION:
[What we'll build — specific technologies, specific features, NOT vague]

DELIVERABLES:
- [Deliverable 1 — specific]
- [Deliverable 2 — specific]
- [Deliverable 3 — specific]
- Complete documentation + handoff
- 30 days post-launch support

TIMELINE: [X] days from kickoff

INVESTMENT (present expensive tier FIRST — anchor high):

Option C — Full AI System: $[highest price]
[List what's included — this is the anchor]

Option B — Core Solution: $[mid price] (RECOMMENDED)
[The "right" choice — most value for the price]

Option A — Starter: $[lowest price]
[Basic version — feels like a steal after seeing Option C]

→ We recommend Option B for {client_name}.
→ 50% upfront to start, 50% on delivery.
→ This scope and pricing is valid for 7 days from today.

WHY DARKCODE:
Built AI systems for dental offices, real estate agencies, and service businesses —
92% auto-resolution rate. We build working systems, not strategy decks. AI-native
workflow = faster delivery at lower cost than agencies charging $15K+.

Without this system, roughly 30-40% of after-hours inquiries go unanswered —
each one a potential customer who moves on to a competitor.

NEXT STEP:
Reply to this email to lock in the scope. Once confirmed:
→ 50% deposit via PayPal: [PAYMENT_LINK]
→ We start within 24 hours.

RULES:
- Never exceed 1 page
- Always reference their specific situation — use the lead info above
- Always include fixed price (never hourly)
- Price tiers must be realistic for the service tier
- Present Option C first, then B (recommended), then A
- Be specific about what you'll build — mention actual technologies (Python, Claude API, n8n, Telegram, WhatsApp, etc.)
- Include the 7-day validity line
- Include the loss aversion line about unanswered inquiries
- End with payment link, not a booking link

Return ONLY the proposal text. No meta-commentary."""

PAYPAL_LINK = "https://paypal.me/darkcodeai"


def generate_proposal(lead: dict, payment_link: str = PAYPAL_LINK) -> dict:
    """Generate a 1-page proposal from lead data using Claude API.

    Args:
        lead: Lead dict from Viper (must have title, description, scores)
        payment_link: PayPal payment link for deposits

    Returns:
        dict with proposal_text, file_path, lead_id, generated_at
    """
    # Get Anthropic client
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        log.error("No ANTHROPIC_API_KEY — cannot generate proposal")
        return {"ok": False, "error": "No API key"}

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    # Extract lead info
    scores = lead.get("scores", {})
    service_tier = scores.get("service_tier", "starter")
    tier_info = SERVICE_TIERS.get(service_tier, SERVICE_TIERS["starter"])

    client_name = lead.get("title", "Prospective Client")
    description = lead.get("description", lead.get("title", ""))
    budget = lead.get("budget", scores.get("recommended_bid", "TBD"))

    prompt = PROPOSAL_PROMPT.format(
        client_name=client_name,
        description=description[:800],
        budget=budget,
        source=lead.get("source", "unknown"),
        service_tier=service_tier,
        tier_details=f"{tier_info['price_range']}, {tier_info['timeline']}",
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=PROPOSAL_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        proposal_text = response.content[0].text.strip()

        # Replace payment link placeholder
        proposal_text = proposal_text.replace("[PAYMENT_LINK]", payment_link)

        # Save to file
        lead_id = lead.get("id", "unknown")
        ts = datetime.now(ET).strftime("%Y%m%d_%H%M%S")
        filename = f"proposal_{lead_id}_{ts}.md"
        filepath = PROPOSALS_DIR / filename
        filepath.write_text(proposal_text)

        log.info("Generated proposal for '%s' → %s", client_name[:40], filename)

        return {
            "ok": True,
            "proposal_text": proposal_text,
            "file_path": str(filepath),
            "lead_id": lead_id,
            "service_tier": service_tier,
            "generated_at": datetime.now(ET).isoformat(),
        }

    except Exception as e:
        log.error("Proposal generation failed: %s", e)
        return {"ok": False, "error": str(e)}


def generate_from_lead_id(lead_id: str) -> dict:
    """Find a lead by ID from viper_leads.json and generate proposal."""
    leads_file = Path.home() / "polymarket-bot" / "data" / "viper_leads.json"
    if not leads_file.exists():
        return {"ok": False, "error": "No viper_leads.json found"}

    leads = json.loads(leads_file.read_text())
    lead = next((l for l in leads if l.get("id") == lead_id), None)
    if not lead:
        return {"ok": False, "error": f"Lead {lead_id} not found"}

    return generate_proposal(lead)


def list_proposals() -> list[dict]:
    """List all generated proposals."""
    proposals = []
    for f in sorted(PROPOSALS_DIR.glob("proposal_*.md"), reverse=True):
        proposals.append({
            "filename": f.name,
            "path": str(f),
            "size": f.stat().st_size,
            "created": datetime.fromtimestamp(f.stat().st_ctime, tz=ET).isoformat(),
        })
    return proposals
