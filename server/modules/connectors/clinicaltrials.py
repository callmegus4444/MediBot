"""ClinicalTrials.gov v2 API connector.

Free, no key required. Pulls active/recent trials matching the question so
doctors can see ongoing studies alongside peer-reviewed evidence.
"""
from __future__ import annotations

import re
from typing import List

import requests

from logger import logger
from schemas.strict import Reference, now_iso

CT_BASE = "https://clinicaltrials.gov/api/v2/studies"
CT_VIEW = "https://clinicaltrials.gov/study/{nct}"
DEFAULT_TIMEOUT = 12.0


def _shorten(text: str, limit: int = 600) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:limit]


def search(query: str, *, retmax: int = 3) -> List[Reference]:
    if not query or not query.strip():
        return []
    params = {
        "query.term": query.strip(),
        "pageSize": max(1, min(int(retmax), 10)),
        "format": "json",
        "fields": (
            "NCTId,BriefTitle,OverallStatus,Phase,Condition,StudyType,"
            "BriefSummary,StartDate,PrimaryCompletionDate,LeadSponsorName"
        ),
    }
    try:
        r = requests.get(CT_BASE, params=params, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        data = r.json() or {}
    except requests.RequestException as exc:
        logger.warning(f"ClinicalTrials.gov call failed: {exc}")
        return []
    except ValueError:
        return []

    references: List[Reference] = []
    retrieved_at = now_iso()
    for i, study in enumerate(data.get("studies") or []):
        ps = (study or {}).get("protocolSection") or {}
        ident = ps.get("identificationModule") or {}
        status_mod = ps.get("statusModule") or {}
        design = ps.get("designModule") or {}
        desc = ps.get("descriptionModule") or {}
        sponsor = ps.get("sponsorCollaboratorsModule") or {}
        cond_mod = ps.get("conditionsModule") or {}

        nct = ident.get("nctId") or f"trial_{i}"
        title = ident.get("briefTitle") or "(untitled trial)"
        status = status_mod.get("overallStatus")
        phases = ", ".join(design.get("phases") or [])
        conditions = ", ".join(cond_mod.get("conditions") or [])
        lead = ((sponsor.get("leadSponsor") or {}).get("name")) or ""
        summary = _shorten(desc.get("briefSummary") or "")
        start = (status_mod.get("startDateStruct") or {}).get("date")

        findings: List[str] = []
        if summary:
            findings.append(summary)
        meta_bits = [b for b in [status, phases, conditions, lead] if b]
        if meta_bits:
            findings.append(" · ".join(meta_bits))

        references.append(
            Reference(
                id=f"trial_{nct}",
                title=title,
                source="ClinicalTrials.gov",
                sourceType="clinical_trial",
                url=CT_VIEW.format(nct=nct),
                confidenceScore=70,
                publishedAt=start[:10] if isinstance(start, str) else None,
                retrievedAt=retrieved_at,
                keyFindings=findings,
                usedInAnswer=False,
                credibilityTier="A",
            )
        )
    logger.info(f"ClinicalTrials.gov returned {len(references)} trials")
    return references
