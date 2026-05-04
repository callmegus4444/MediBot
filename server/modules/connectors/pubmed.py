"""PubMed E-utilities connector.

Public reference: https://www.ncbi.nlm.nih.gov/home/develop/api/

Two-step retrieval:
  1. esearch -> PMIDs for a query.
  2. esummary -> title, authors, journal, publication date for those PMIDs.

We deliberately do not pull full text. Abstracts via efetch can be added later
behind the same interface.
"""
from __future__ import annotations

import os
import time
from typing import List, Optional

import requests

from logger import logger
from schemas.strict import Reference, now_iso

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_VIEW_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
DEFAULT_TIMEOUT = 8.0


def _api_key() -> Optional[str]:
    return os.getenv("NCBI_API_KEY") or None


def _params(extra: dict) -> dict:
    base = {"db": "pubmed", "retmode": "json", "tool": "medibot-strict", "email": os.getenv("NCBI_CONTACT_EMAIL", "noreply@example.com")}
    key = _api_key()
    if key:
        base["api_key"] = key
    base.update(extra)
    return base


def _esearch(query: str, retmax: int) -> List[str]:
    params = _params({"term": query, "retmax": str(retmax), "sort": "relevance"})
    r = requests.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    data = r.json() or {}
    ids = data.get("esearchresult", {}).get("idlist", []) or []
    return [str(pmid) for pmid in ids]


def _esummary(pmids: List[str]) -> dict:
    if not pmids:
        return {}
    params = _params({"id": ",".join(pmids)})
    r = requests.get(f"{EUTILS_BASE}/esummary.fcgi", params=params, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    data = r.json() or {}
    return data.get("result", {}) or {}


def _format_pubdate(record: dict) -> Optional[str]:
    pubdate = record.get("pubdate") or record.get("epubdate") or record.get("sortpubdate")
    if not pubdate:
        return None
    return str(pubdate)[:10]


def search(query: str, *, retmax: int = 5) -> List[Reference]:
    if not query or not query.strip():
        return []
    try:
        pmids = _esearch(query.strip(), retmax)
        if not pmids:
            return []
        # Light throttle when no API key.
        if not _api_key():
            time.sleep(0.34)
        result = _esummary(pmids)
    except requests.RequestException as exc:
        logger.warning(f"PubMed connector failed: {exc}")
        return []
    except ValueError as exc:
        logger.warning(f"PubMed connector returned invalid JSON: {exc}")
        return []

    references: List[Reference] = []
    retrieved_at = now_iso()
    for pmid in pmids:
        record = result.get(pmid)
        if not isinstance(record, dict):
            continue
        title = (record.get("title") or "").strip()
        if not title:
            continue
        journal = record.get("fulljournalname") or record.get("source") or ""
        published_at = _format_pubdate(record)
        references.append(
            Reference(
                id=f"pubmed_{pmid}",
                title=title,
                source="PubMed",
                sourceType="peer_reviewed_journal",
                url=PUBMED_VIEW_URL.format(pmid=pmid),
                confidenceScore=0,
                publishedAt=published_at,
                retrievedAt=retrieved_at,
                keyFindings=[journal] if journal else [],
                usedInAnswer=False,
                credibilityTier="A",
            )
        )
    return references
