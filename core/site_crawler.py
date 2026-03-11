"""Thor Site Crawler — Cloudflare /crawl for client website ingestion.

When Thor starts building a chatbot for a client, this module crawls their
entire website and extracts structured data for the knowledge base:
  - Services offered
  - FAQ content (questions + answers)
  - Staff / doctor / team names
  - Hours of operation
  - Locations / addresses
  - Insurance accepted
  - Pricing / fees
  - Booking / appointment info
  - Contact info

One crawl = full knowledge base built. No manual copy-paste.
Results saved to data/client_sites/{client_name}.json.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

log = logging.getLogger("thor.site_crawler")

ET = timezone(timedelta(hours=-5))

CLIENT_SITES_DIR = Path(__file__).resolve().parent.parent / "data" / "client_sites"
CLIENT_SITES_DIR.mkdir(parents=True, exist_ok=True)

_CF_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
_CF_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "")

if not _CF_ACCOUNT_ID:
    _dotenv = Path.home() / "polymarket-bot" / ".env"
    if _dotenv.exists():
        for line in _dotenv.read_text().splitlines():
            if line.startswith("CLOUDFLARE_ACCOUNT_ID="):
                _CF_ACCOUNT_ID = line.split("=", 1)[1].strip()
            elif line.startswith("CLOUDFLARE_API_TOKEN="):
                _CF_API_TOKEN = line.split("=", 1)[1].strip()


@dataclass
class ClientSiteData:
    """Structured data extracted from a client's website."""
    client_name: str = ""
    website_url: str = ""
    pages_crawled: int = 0
    services: list[str] = field(default_factory=list)
    faq: list[dict] = field(default_factory=list)  # [{q: str, a: str}]
    staff: list[str] = field(default_factory=list)
    hours: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    insurance: list[str] = field(default_factory=list)
    pricing: list[str] = field(default_factory=list)
    booking_info: str = ""
    contact_info: dict = field(default_factory=dict)  # {phone, email, address}
    raw_pages: list[dict] = field(default_factory=list)  # [{url, content}]
    crawled_at: str = ""
    error: str = ""


def crawl_client_site(
    website_url: str,
    client_name: str,
    page_limit: int = 100,
) -> ClientSiteData:
    """Crawl a client's entire website and extract structured data.

    Args:
        website_url: The client's website URL.
        client_name: Business name (used for file naming).
        page_limit: Max pages to crawl (default 100).

    Returns:
        ClientSiteData with all extracted information.
    """
    result = ClientSiteData(
        client_name=client_name,
        website_url=website_url,
        crawled_at=datetime.now(ET).isoformat(),
    )

    if not _CF_ACCOUNT_ID or not _CF_API_TOKEN:
        result.error = "Cloudflare credentials not configured"
        log.error("[SITE_CRAWL] %s", result.error)
        return result

    crawl_url = (
        f"https://api.cloudflare.com/client/v4/accounts/{_CF_ACCOUNT_ID}"
        f"/browser-rendering/crawl"
    )

    headers = {
        "Authorization": f"Bearer {_CF_API_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "url": website_url,
        "render": False,
        "limit": page_limit,
    }

    log.info("[SITE_CRAWL] Crawling %s (limit=%d)...", website_url, page_limit)

    try:
        resp = requests.post(crawl_url, json=payload, headers=headers, timeout=120)
        if resp.status_code != 200:
            result.error = f"CF API {resp.status_code}: {resp.text[:300]}"
            log.error("[SITE_CRAWL] %s", result.error)
            return result

        data = resp.json()
        if not data.get("success"):
            result.error = f"CF API error: {data.get('errors', [])}"
            log.error("[SITE_CRAWL] %s", result.error)
            return result

        pages = data.get("result", {}).get("pages", [])
        result.pages_crawled = len(pages)

        # Store raw pages for knowledge base
        for page in pages:
            page_url = page.get("url", "")
            content = page.get("content", "")
            result.raw_pages.append({"url": page_url, "content": content})

        # Extract structured data from all pages
        all_content = "\n\n".join(p.get("content", "") for p in pages)
        _extract_services(result, pages)
        _extract_faq(result, pages)
        _extract_staff(result, all_content)
        _extract_hours(result, all_content)
        _extract_locations(result, all_content)
        _extract_insurance(result, all_content)
        _extract_pricing(result, all_content)
        _extract_booking(result, all_content)
        _extract_contact(result, all_content)

        log.info(
            "[SITE_CRAWL] Done: %s — %d pages, %d services, %d FAQ, %d staff",
            client_name, result.pages_crawled, len(result.services),
            len(result.faq), len(result.staff),
        )

    except requests.Timeout:
        result.error = "Crawl timed out (120s)"
        log.error("[SITE_CRAWL] Timeout: %s", website_url)
    except Exception as e:
        result.error = str(e)[:300]
        log.error("[SITE_CRAWL] Error: %s", result.error)

    # Save to disk
    _save_crawl(client_name, result)

    return result


def _save_crawl(client_name: str, data: ClientSiteData) -> Path:
    """Save crawl result to data/client_sites/{slug}.json."""
    slug = re.sub(r"[^a-z0-9]+", "-", client_name.lower()).strip("-")[:60]
    filepath = CLIENT_SITES_DIR / f"{slug}.json"

    output = {
        "client_name": data.client_name,
        "website_url": data.website_url,
        "pages_crawled": data.pages_crawled,
        "services": data.services,
        "faq": data.faq,
        "staff": data.staff,
        "hours": data.hours,
        "locations": data.locations,
        "insurance": data.insurance,
        "pricing": data.pricing,
        "booking_info": data.booking_info,
        "contact_info": data.contact_info,
        "crawled_at": data.crawled_at,
        "error": data.error,
        # Raw pages stored separately for knowledge base generation
        "page_urls": [p["url"] for p in data.raw_pages],
    }

    filepath.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    log.info("[SITE_CRAWL] Saved to %s", filepath)

    # Also save raw content in a separate file for knowledge base
    raw_path = CLIENT_SITES_DIR / f"{slug}_raw.json"
    raw_output = {
        "client_name": data.client_name,
        "pages": data.raw_pages,
        "crawled_at": data.crawled_at,
    }
    raw_path.write_text(json.dumps(raw_output, indent=2, ensure_ascii=False))

    return filepath


def build_knowledge_base(data: ClientSiteData) -> str:
    """Build a chatbot knowledge base document from crawl data.

    Returns a formatted text document ready to feed into a chatbot's
    training data / system prompt.
    """
    sections = []

    sections.append(f"# {data.client_name} — Knowledge Base")
    sections.append(f"Website: {data.website_url}")
    sections.append(f"Pages crawled: {data.pages_crawled}")
    sections.append("")

    if data.services:
        sections.append("## Services")
        for s in data.services:
            sections.append(f"- {s}")
        sections.append("")

    if data.faq:
        sections.append("## Frequently Asked Questions")
        for qa in data.faq:
            sections.append(f"**Q: {qa.get('q', '')}**")
            sections.append(f"A: {qa.get('a', '')}")
            sections.append("")

    if data.staff:
        sections.append("## Staff / Team")
        for name in data.staff:
            sections.append(f"- {name}")
        sections.append("")

    if data.hours:
        sections.append("## Hours of Operation")
        for h in data.hours:
            sections.append(f"- {h}")
        sections.append("")

    if data.locations:
        sections.append("## Locations")
        for loc in data.locations:
            sections.append(f"- {loc}")
        sections.append("")

    if data.insurance:
        sections.append("## Insurance Accepted")
        for ins in data.insurance:
            sections.append(f"- {ins}")
        sections.append("")

    if data.pricing:
        sections.append("## Pricing")
        for p in data.pricing:
            sections.append(f"- {p}")
        sections.append("")

    if data.booking_info:
        sections.append("## Booking / Appointments")
        sections.append(data.booking_info)
        sections.append("")

    if data.contact_info:
        sections.append("## Contact Information")
        for key, val in data.contact_info.items():
            if val:
                sections.append(f"- {key.title()}: {val}")
        sections.append("")

    return "\n".join(sections)


# --- Extraction helpers ---

def _extract_services(result: ClientSiteData, pages: list[dict]) -> None:
    """Extract services from pages with /services in URL or service-like content."""
    for page in pages:
        url = page.get("url", "").lower()
        content = page.get("content", "")

        if "/services" in url or "/what-we-do" in url or "/treatments" in url:
            # Extract list items that look like services
            lines = content.split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith(("- ", "* ", "• ")):
                    service = line.lstrip("-*• ").strip()
                    if 3 < len(service) < 100 and service not in result.services:
                        result.services.append(service)
                elif re.match(r"^#{1,3}\s+\w", line):
                    # Heading = likely service category
                    service = line.lstrip("#").strip()
                    if 3 < len(service) < 80 and service not in result.services:
                        result.services.append(service)


def _extract_faq(result: ClientSiteData, pages: list[dict]) -> None:
    """Extract FAQ Q&A pairs from pages with FAQ-like URLs."""
    faq_re = re.compile(r"/faq|/frequently|/questions|/help", re.IGNORECASE)

    for page in pages:
        url = page.get("url", "")
        content = page.get("content", "")

        if not faq_re.search(url):
            continue

        # Look for Q&A patterns
        lines = content.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # Question patterns: starts with Q:, ends with ?, or is a heading
            if line.endswith("?") or line.startswith("Q:") or line.startswith("Q."):
                question = line.lstrip("Q:.#").strip()
                # Next non-empty line is likely the answer
                answer = ""
                for j in range(i + 1, min(i + 5, len(lines))):
                    if lines[j].strip():
                        answer = lines[j].strip().lstrip("A:.").strip()
                        break
                if question and len(question) > 10:
                    result.faq.append({"q": question[:200], "a": answer[:500]})
            i += 1


def _extract_staff(result: ClientSiteData, content: str) -> None:
    """Extract staff/doctor/team names."""
    # Look for "Dr. Name" or "Name, DDS/MD/etc"
    dr_pattern = re.compile(
        r"(?:Dr\.?\s+|Doctor\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
    )
    for match in dr_pattern.finditer(content):
        name = f"Dr. {match.group(1)}"
        if name not in result.staff:
            result.staff.append(name)

    # Title patterns: "Name, DDS" or "Name, MD"
    title_pattern = re.compile(
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}),?\s*(?:DDS|DMD|MD|DO|RN|PA|NP|DPT|DC)",
    )
    for match in title_pattern.finditer(content):
        name = match.group(0)[:60]
        if name not in result.staff:
            result.staff.append(name)


def _extract_hours(result: ClientSiteData, content: str) -> None:
    """Extract hours of operation."""
    hours_pattern = re.compile(
        r"((?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*(?:\s*[-–]\s*"
        r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*)?)\s*[:]\s*"
        r"(\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm)\s*[-–]\s*"
        r"\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm))",
        re.IGNORECASE,
    )
    for match in hours_pattern.finditer(content):
        hours_str = match.group(0).strip()
        if hours_str not in result.hours:
            result.hours.append(hours_str)


def _extract_locations(result: ClientSiteData, content: str) -> None:
    """Extract physical addresses."""
    # US address pattern
    addr_pattern = re.compile(
        r"\d{1,5}\s+[A-Z][a-zA-Z\s]+(?:St|Ave|Blvd|Dr|Rd|Ln|Way|Ct|Pl|Cir)"
        r"[.,]?\s*(?:Suite|Ste|#|Apt|Unit)?\s*\d*[.,]?\s*"
        r"[A-Z][a-z]+[.,]?\s+[A-Z]{2}\s+\d{5}",
    )
    for match in addr_pattern.finditer(content):
        addr = match.group(0).strip()
        if addr not in result.locations:
            result.locations.append(addr)


def _extract_insurance(result: ClientSiteData, content: str) -> None:
    """Extract accepted insurance providers."""
    insurance_names = [
        "Aetna", "BlueCross", "Blue Cross", "Cigna", "Delta Dental",
        "Humana", "MetLife", "United Healthcare", "UnitedHealthcare",
        "Guardian", "Principal", "Anthem", "Kaiser", "Medicare",
        "Medicaid", "Tricare", "GEHA", "Aflac",
    ]
    for name in insurance_names:
        if name.lower() in content.lower() and name not in result.insurance:
            result.insurance.append(name)


def _extract_pricing(result: ClientSiteData, content: str) -> None:
    """Extract pricing information."""
    price_pattern = re.compile(
        r"(?:starting\s+(?:at|from)|from|price|cost|fee)[:\s]*\$\d[\d,]*(?:\.\d{2})?",
        re.IGNORECASE,
    )
    for match in price_pattern.finditer(content):
        price_str = match.group(0).strip()[:100]
        if price_str not in result.pricing:
            result.pricing.append(price_str)


def _extract_booking(result: ClientSiteData, content: str) -> None:
    """Extract booking/appointment info."""
    booking_patterns = [
        r"book\s*(?:an?\s*)?(?:appointment|consultation|visit)",
        r"schedule\s*(?:an?\s*)?(?:appointment|consultation|visit)",
        r"online\s*(?:booking|scheduling)",
        r"request\s*(?:an?\s*)?appointment",
    ]
    booking_re = re.compile("|".join(booking_patterns), re.IGNORECASE)

    matches = booking_re.findall(content)
    if matches:
        result.booking_info = "Online booking available"

    # Check for known booking platforms
    platforms = ["calendly", "acuity", "appointy", "setmore", "zocdoc", "healthgrades"]
    for platform in platforms:
        if platform in content.lower():
            result.booking_info = f"Online booking via {platform.title()}"
            break


def _extract_contact(result: ClientSiteData, content: str) -> None:
    """Extract contact information."""
    # Phone
    phone_re = re.compile(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
    phone_match = phone_re.search(content)
    if phone_match:
        result.contact_info["phone"] = phone_match.group(0)

    # Email
    email_re = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
    email_match = email_re.search(content)
    if email_match:
        email = email_match.group(0)
        if not any(x in email for x in ["example", "test", "noreply", "wordpress"]):
            result.contact_info["email"] = email
