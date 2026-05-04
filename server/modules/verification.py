"""Verification + response generation for strict mode.

Uses the existing Groq Llama model with temperature=0 and JSON-only outputs.
Two distinct prompts: one for verification, one for response generation.
Both fail closed on any parsing error or schema mismatch.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

from langchain_groq import ChatGroq

from logger import logger
from schemas.strict import ABSTENTION_SENTENCE, Reference

VERIFICATION_SYSTEM = """You are the Verification Agent for a strict clinical answering system.

Your job is to decide whether the evidence is sufficient to support a factual answer.

Rules:
- Do not generate the final answer.
- Verify each proposed claim against evidence.
- A claim is supported only if at least one source directly supports it.
- Prefer two independent sources for treatment, diagnosis, safety, or guideline claims.
- If sources conflict, mark conflict and lower confidence.
- If evidence is missing, stale, indirect, or not clinically applicable, mark insufficient.
- Do not fill gaps with medical knowledge.
- Return valid JSON only, with no prose, no markdown."""


VERIFICATION_USER_TMPL = """Doctor question:
{question}

Internal evidence (chunks from clinician-curated PDFs):
{internal_json}

External evidence (PubMed records):
{external_json}

Return this exact JSON shape:
{{
  "evidenceStatus": "sufficient|partial|insufficient|conflicting",
  "verifiedClaims": [
    {{
      "claim": "string",
      "supportingSourceIds": ["string"],
      "supportStrength": "strong|moderate|weak"
    }}
  ],
  "rejectedClaims": [
    {{ "claim": "string", "reason": "unsupported|conflicting|not_applicable|too_old" }}
  ],
  "conflictsDetected": false,
  "mustAbstain": true,
  "abstentionReason": "string|null"
}}"""


RESPONSE_SYSTEM = """You are the Response Generator Agent for MediBot strict mode. Your audience is doctors.

Rules:
- Use only verified claims supplied by the Verification Agent.
- Do not add medical facts from memory.
- Do not guess.
- Do not provide unsupported diagnosis, dosing, treatment, or safety claims.
- Use concise clinical language.
- If mustAbstain is true or evidenceStatus is insufficient/conflicting, the answer field must be exactly:
  "I do not have sufficient verified evidence to answer this question."
- Return valid JSON only, no prose, no markdown."""


RESPONSE_USER_TMPL = """Doctor question:
{question}

Verification result:
{verification_json}

Available references (subset of which may be cited):
{references_json}

Return this exact JSON shape:
{{
  "answer": "string",
  "status": "answered|partial|insufficient_evidence|conflicting_evidence",
  "citedSourceIds": ["string"],
  "clinicalCaveats": ["string"]
}}"""


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _llm() -> ChatGroq:
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name=os.getenv("GROQ_MODEL", "llama3-70b-8192"),
        temperature=0,
    )


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _internal_payload(internal_chunks: List[dict]) -> List[dict]:
    out: List[dict] = []
    for i, chunk in enumerate(internal_chunks or []):
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        out.append(
            {
                "id": chunk.get("id") or f"internal_{i}",
                "documentTitle": chunk.get("documentTitle"),
                "page": chunk.get("page"),
                "text": text[:1200],
                "score": chunk.get("score"),
            }
        )
    return out


def _external_payload(refs: List[Reference]) -> List[dict]:
    out: List[dict] = []
    for ref in refs or []:
        out.append(
            {
                "id": ref.id,
                "title": ref.title,
                "source": ref.source,
                "url": ref.url,
                "publishedAt": ref.publishedAt,
                "keyFindings": ref.keyFindings,
            }
        )
    return out


def verify(
    question: str,
    *,
    internal_chunks: List[dict],
    external_refs: List[Reference],
) -> Optional[dict]:
    user_msg = VERIFICATION_USER_TMPL.format(
        question=question,
        internal_json=json.dumps(_internal_payload(internal_chunks), ensure_ascii=False),
        external_json=json.dumps(_external_payload(external_refs), ensure_ascii=False),
    )
    try:
        result = _llm().invoke(
            [
                {"role": "system", "content": VERIFICATION_SYSTEM},
                {"role": "user", "content": user_msg},
            ]
        )
    except Exception as exc:
        logger.warning(f"Verification LLM call failed: {exc}")
        return None

    parsed = _extract_json(getattr(result, "content", "") or str(result))
    if not isinstance(parsed, dict):
        logger.warning("Verification agent returned non-JSON response")
        return None
    return parsed


def generate(
    question: str,
    *,
    verification: dict,
    references: List[Reference],
) -> Tuple[str, str, List[str], List[str]]:
    """Returns (answer, status, citedSourceIds, clinicalCaveats)."""
    user_msg = RESPONSE_USER_TMPL.format(
        question=question,
        verification_json=json.dumps(verification, ensure_ascii=False),
        references_json=json.dumps(_external_payload(references), ensure_ascii=False),
    )
    try:
        result = _llm().invoke(
            [
                {"role": "system", "content": RESPONSE_SYSTEM},
                {"role": "user", "content": user_msg},
            ]
        )
    except Exception as exc:
        logger.warning(f"Response generator LLM call failed: {exc}")
        return ABSTENTION_SENTENCE, "insufficient_evidence", [], []

    parsed = _extract_json(getattr(result, "content", "") or str(result))
    if not isinstance(parsed, dict):
        return ABSTENTION_SENTENCE, "insufficient_evidence", [], []

    answer = str(parsed.get("answer") or ABSTENTION_SENTENCE).strip()
    status = parsed.get("status") or "insufficient_evidence"
    cited = parsed.get("citedSourceIds") or []
    caveats = parsed.get("clinicalCaveats") or []

    if status not in ("answered", "partial", "insufficient_evidence", "conflicting_evidence"):
        status = "insufficient_evidence"

    if status in ("insufficient_evidence", "conflicting_evidence"):
        answer = ABSTENTION_SENTENCE
        cited = []

    if not isinstance(cited, list):
        cited = []
    if not isinstance(caveats, list):
        caveats = []

    return answer, status, [str(c) for c in cited], [str(c) for c in caveats]
