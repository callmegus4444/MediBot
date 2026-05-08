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
import re
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

import requests

from logger import logger
from schemas.strict import Reference, now_iso

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_VIEW_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
DEFAULT_TIMEOUT = 12.0


def _api_key() -> Optional[str]:
    return os.getenv("NCBI_API_KEY") or None


def _params(extra: dict) -> dict:
    base = {"db": "pubmed", "retmode": "json", "tool": "medibot-strict", "email": os.getenv("NCBI_CONTACT_EMAIL", "noreply@example.com")}
    key = _api_key()
    if key:
        base["api_key"] = key
    base.update(extra)
    return base


_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "of", "in", "on", "at", "to", "for", "by", "with", "and", "or", "but",
    "do", "does", "did", "can", "could", "should", "would", "may", "might",
    "this", "that", "these", "those", "i", "you", "he", "she", "it", "we",
    "they", "my", "your", "his", "her", "its", "our", "their",
}


def _simplify(query: str) -> str:
    cleaned = re.sub(r"[^\w\s\-]", " ", query)
    tokens = [t for t in cleaned.split() if t.lower() not in _STOPWORDS]
    return " ".join(tokens).strip()


def _esearch_call(term: str, retmax: int) -> List[str]:
    params = _params({"term": term, "retmax": str(retmax), "sort": "relevance"})
    r = requests.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    data = r.json() or {}
    ids = data.get("esearchresult", {}).get("idlist", []) or []
    return [str(pmid) for pmid in ids]


def _llm_rewrite(query: str) -> Optional[str]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        from langchain_groq import ChatGroq
    except ImportError:
        return None
    try:
        model = ChatGroq(
            groq_api_key=api_key,
            model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0,
        )
        prompt = (
            "You convert a doctor's natural-language clinical question into a concise "
            "PubMed search query. Fix obvious typos. Output ONLY the search keywords, "
            "no quotes, no explanation, max 8 words.\n\nQuestion: " + query
        )
        result = model.invoke([{"role": "user", "content": prompt}])
        rewritten = (getattr(result, "content", "") or "").strip().splitlines()[0]
        rewritten = re.sub(r"[^\w\s\-]", " ", rewritten).strip()
        return rewritten or None
    except Exception as exc:
        logger.warning(f"PubMed LLM rewrite failed: {exc}")
        return None


def _esearch(query: str, retmax: int) -> List[str]:
    # Always try LLM rewrite first — it produces MeSH-friendly queries that
    # match PubMed indexing far better than the doctor's casual phrasing.
    rewritten = _llm_rewrite(query)
    primary = rewritten or query
    if rewritten:
        logger.info(f"PubMed esearch using LLM-rewritten query: {rewritten!r}")
    ids = _esearch_call(primary, retmax)
    if ids:
        return ids
    # Fallback 1: original user query (in case the rewriter over-narrowed).
    if rewritten and rewritten.lower() != query.lower():
        if not _api_key():
            time.sleep(0.34)
        logger.info(f"PubMed esearch fallback to raw user query: {query!r}")
        ids = _esearch_call(query, retmax)
        if ids:
            return ids
    # Fallback 2: stopword-stripped simplified query.
    simplified = _simplify(query)
    if simplified and simplified.lower() not in (query.lower(), (rewritten or "").lower()):
        if not _api_key():
            time.sleep(0.34)
        logger.info(f"PubMed esearch fallback simplified: {simplified!r}")
        return _esearch_call(simplified, retmax)
    return ids


def _esummary(pmids: List[str]) -> dict:
    if not pmids:
        return {}
    params = _params({"id": ",".join(pmids)})
    r = requests.get(f"{EUTILS_BASE}/esummary.fcgi", params=params, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    data = r.json() or {}
    return data.get("result", {}) or {}


def _efetch_abstracts(pmids: List[str]) -> Dict[str, str]:
    if not pmids:
        return {}
    params = _params({"id": ",".join(pmids), "rettype": "abstract", "retmode": "xml"})
    try:
        r = requests.get(f"{EUTILS_BASE}/efetch.fcgi", params=params, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        root = ET.fromstring(r.text)
    except (requests.RequestException, ET.ParseError) as exc:
        logger.warning(f"PubMed efetch failed: {exc}")
        return {}

    abstracts: Dict[str, str] = {}
    for article in root.iter("PubmedArticle"):
        pmid_el = article.find(".//PMID")
        if pmid_el is None or not pmid_el.text:
            continue
        pmid = pmid_el.text.strip()
        parts: List[str] = []
        for ab in article.iter("AbstractText"):
            label = ab.attrib.get("Label")
            text = "".join(ab.itertext()).strip()
            if not text:
                continue
            parts.append(f"{label}: {text}" if label else text)
        if parts:
            joined = re.sub(r"\s+", " ", " ".join(parts)).strip()
            abstracts[pmid] = joined[:1500]
    return abstracts


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
        if not _api_key():
            time.sleep(0.34)
        abstracts = _efetch_abstracts(pmids)
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
        abstract = abstracts.get(pmid, "").strip()
        findings: List[str] = []
        if abstract:
            findings.append(abstract)
        if journal:
            findings.append(f"Journal: {journal}")
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
                keyFindings=findings,
                usedInAnswer=False,
                credibilityTier="A",
            )
        )
    return references
