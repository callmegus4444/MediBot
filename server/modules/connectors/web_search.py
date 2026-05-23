"""Tavily web search connector restricted to trusted medical domains.

Used as a second external source alongside PubMed inside the strict
orchestrator. Falls back gracefully (returns []) if the API key is missing
or the request fails.
"""
from __future__ import annotations

import os
import re
from typing import List, Optional
from urllib.parse import urlparse

import requests

from logger import logger
from schemas.strict import Reference, now_iso

TAVILY_ENDPOINT = "https://api.tavily.com/search"
DEFAULT_TIMEOUT = 15.0

# Curated whitelist of authoritative medical / clinical-guideline domains.
TRUSTED_MEDICAL_DOMAINS: List[str] = [
    "who.int",
    "nih.gov",
    "ncbi.nlm.nih.gov",
    "medlineplus.gov",
    "cdc.gov",
    "fda.gov",
    "ema.europa.eu",
    "nice.org.uk",
    "cochrane.org",
    "cochranelibrary.com",
    "ahrq.gov",
    "mayoclinic.org",
    "clevelandclinic.org",
    "hopkinsmedicine.org",
    "uptodate.com",
    "merckmanuals.com",
    "aafp.org",
    "ama-assn.org",
    "acc.org",
    "heart.org",
    "diabetes.org",
    "cancer.gov",
    "cancer.org",
    "rxlist.com",
    "drugs.com",
    "medscape.com",
    "bmj.com",
    "nejm.org",
    "thelancet.com",
    "jamanetwork.com",
    "nature.com",
    "sciencedirect.com",
    "springer.com",
    "wiley.com",
]


def _api_key() -> Optional[str]:
    return os.getenv("TAVILY_API_KEY") or None


def _domain_of(url: str) -> str:
    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        return ""
    return host.lower().lstrip("www.") if host else ""


def _credibility_for(url: str) -> str:
    host = _domain_of(url)
    # Government and major society sources are treated as Tier A (peer / authoritative).
    high_tier = (".gov", "who.int", "cochrane", "nejm.org", "bmj.com", "thelancet.com", "jamanetwork.com", "nature.com")
    if any(token in host for token in high_tier):
        return "A"
    return "B"


def search(query: str, *, retmax: int = 5) -> List[Reference]:
    if not query or not query.strip():
        return []
    key = _api_key()
    if not key:
        logger.info("Tavily web_search skipped: TAVILY_API_KEY not set")
        return []

    payload = {
        "api_key": key,
        "query": query.strip(),
        "search_depth": "advanced",
        "max_results": max(1, min(int(retmax), 10)),
        "include_answer": False,
        "include_raw_content": False,
        "include_domains": TRUSTED_MEDICAL_DOMAINS,
    }

    try:
        r = requests.post(TAVILY_ENDPOINT, json=payload, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        data = r.json() or {}
    except requests.RequestException as exc:
        logger.warning(f"Tavily web search failed: {exc}")
        return []
    except ValueError as exc:
        logger.warning(f"Tavily returned invalid JSON: {exc}")
        return []

    results = data.get("results") or []
    references: List[Reference] = []
    retrieved_at = now_iso()
    for i, item in enumerate(results):
        if not isinstance(item, dict):
            continue
        url = (item.get("url") or "").strip()
        title = (item.get("title") or "").strip()
        content = (item.get("content") or "").strip()
        if not url or not title:
            continue
        host = _domain_of(url)
        snippet = re.sub(r"\s+", " ", content)[:1500] if content else ""
        published_at = (item.get("published_date") or None)
        if isinstance(published_at, str):
            published_at = published_at[:10]
        score_raw = item.get("score")
        try:
            score_pct = int(round(float(score_raw) * 100)) if score_raw is not None else 0
        except (TypeError, ValueError):
            score_pct = 0
        references.append(
            Reference(
                id=f"web_{i}_{host.replace('.', '_')}",
                title=title,
                source=host or "Web",
                sourceType="web",
                url=url,
                confidenceScore=max(0, min(100, score_pct)),
                publishedAt=published_at,
                retrievedAt=retrieved_at,
                keyFindings=[snippet] if snippet else [],
                usedInAnswer=False,
                credibilityTier=_credibility_for(url),
            )
        )
    logger.info(f"Tavily web search '{query[:60]}' returned {len(references)} refs")
    return references
