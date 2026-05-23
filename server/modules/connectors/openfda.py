"""OpenFDA drug label connector.

Free, no API key required (rate-limited). Pulls structured drug label data —
indications, dosage, warnings, contraindications, mechanism of action — for
questions that name a drug.
"""
from __future__ import annotations

import os
import re
from typing import List, Optional

import requests

from logger import logger
from schemas.strict import Reference, now_iso

OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
DEFAULT_TIMEOUT = 12.0


def _api_key() -> Optional[str]:
    return os.getenv("OPENFDA_API_KEY") or None


def _shorten(text: str, limit: int = 600) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _extract_drug_terms(query: str) -> List[str]:
    # Naive: take alphabetic tokens of length >= 4 that are not stopwords.
    stop = {
        "what", "which", "when", "where", "tell", "about", "this", "that",
        "side", "effects", "drug", "medication", "dose", "dosage", "uses",
        "treatment", "indication", "indications", "warnings", "contraindications",
        "interactions", "mechanism", "action", "patient", "patients", "adult",
        "adults", "child", "children", "elderly", "pregnancy",
    }
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", query.lower())
    return [t for t in tokens if t not in stop]


def search(query: str, *, retmax: int = 3) -> List[Reference]:
    if not query or not query.strip():
        return []

    terms = _extract_drug_terms(query)
    if not terms:
        return []

    # Try each candidate drug term; first hit wins (most queries name one drug).
    references: List[Reference] = []
    retrieved_at = now_iso()
    seen_ids: set[str] = set()
    for term in terms[:3]:
        params = {
            "search": f"openfda.generic_name:{term}+openfda.brand_name:{term}",
            "limit": max(1, min(int(retmax), 5)),
        }
        key = _api_key()
        if key:
            params["api_key"] = key
        try:
            r = requests.get(OPENFDA_LABEL_URL, params=params, timeout=DEFAULT_TIMEOUT)
            if r.status_code == 404:
                continue
            r.raise_for_status()
            data = r.json() or {}
        except requests.RequestException as exc:
            logger.warning(f"OpenFDA call failed for {term!r}: {exc}")
            continue
        except ValueError:
            continue

        for i, result in enumerate(data.get("results") or []):
            if not isinstance(result, dict):
                continue
            openfda = result.get("openfda") or {}
            brand = (openfda.get("brand_name") or [None])[0]
            generic = (openfda.get("generic_name") or [None])[0]
            name = brand or generic or term
            ref_id = f"openfda_{term}_{i}"
            if ref_id in seen_ids:
                continue
            seen_ids.add(ref_id)

            findings: List[str] = []
            for field, label in [
                ("indications_and_usage", "Indications"),
                ("dosage_and_administration", "Dosage"),
                ("contraindications", "Contraindications"),
                ("warnings", "Warnings"),
                ("warnings_and_cautions", "Warnings"),
                ("adverse_reactions", "Adverse reactions"),
                ("drug_interactions", "Interactions"),
                ("mechanism_of_action", "Mechanism"),
            ]:
                val = result.get(field)
                if isinstance(val, list) and val:
                    findings.append(f"{label}: {_shorten(val[0])}")
                if len(findings) >= 4:
                    break

            spl_set_id = (openfda.get("spl_set_id") or [None])[0]
            url = (
                f"https://labels.fda.gov/spl-doc?hl={spl_set_id}"
                if spl_set_id
                else f"https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=overview.process&ApplNo={(openfda.get('application_number') or [''])[0]}"
            )
            references.append(
                Reference(
                    id=ref_id,
                    title=f"FDA Drug Label: {name}",
                    source="OpenFDA",
                    sourceType="drug_label",
                    url=url,
                    confidenceScore=85,
                    publishedAt=(result.get("effective_time") or "")[:10] or None,
                    retrievedAt=retrieved_at,
                    keyFindings=findings,
                    usedInAnswer=False,
                    credibilityTier="A",
                )
            )
        if references:
            break

    logger.info(f"OpenFDA returned {len(references)} drug-label refs for terms {terms[:3]}")
    return references
