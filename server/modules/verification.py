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

VERIFICATION_SYSTEM = """You are the Verification Agent for a clinical answering system aimed at doctors.

Your job is to decide whether the supplied evidence (PubMed abstracts and any internal document chunks) lets us give a useful, evidence-grounded answer to the doctor's question.

Rules:
- Do not generate the final answer text.
- A claim is supported if at least one source's text (title, abstract, or chunk) directly mentions or implies it.
- Be permissive on minor typos in the user's question (e.g., "twll" -> "tell"), abbreviations, and casual phrasing — interpret intent.
- Treat peer-reviewed PubMed abstracts as Tier-A evidence on their own; even one clearly relevant abstract is SUFFICIENT to set evidenceStatus to "partial" or "sufficient".
- For widely established medical facts (e.g., symptoms of common diseases, mechanism of drugs, basic anatomy), you may treat the question as answerable even if the supplied evidence only partially covers it — set evidenceStatus to "partial" and mustAbstain to false.
- Only mark "insufficient" and mustAbstain true when: (a) NONE of the sources discuss the question's topic at all, AND (b) the question is NOT about a well-established mainstream medical concept.
- If sources contradict each other, mark conflictsDetected true and choose "conflicting".
- Default mustAbstain to FALSE unless you have a clear reason to abstain.
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
- Primarily use the verified claims supplied by the Verification Agent.
- When evidenceStatus is "partial" and the question is about a well-established mainstream medical fact (e.g., common disease symptoms, standard drug classes, basic physiology), you MAY supplement verified claims with your general clinical knowledge — but clearly note which parts are from the supplied sources and which are general knowledge.
- Do not guess on novel, complex, or treatment-specific claims without source support.
- Do not provide unsupported specific dosing or diagnostic criteria without a cited source.
- Use concise clinical language.
- Only set the answer field to "I do not have sufficient verified evidence to answer this question." when mustAbstain is explicitly true OR evidenceStatus is "insufficient" or "conflicting".
- When evidenceStatus is "partial" or "sufficient", always produce a real clinical answer — never refuse.
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
        model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
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


SYNTH_SYSTEM = """You are MediBot, a clinical assistant for doctors.

You will be given a doctor's question plus a list of source materials (PubMed abstracts and/or chunks from uploaded internal documents). Your job:

1. Write a concise, professional clinical answer.
2. Prefer specifics from the supplied sources whenever they are relevant — cite them by their `id` field using inline tags like [pubmed_12345] or [internal_3] right after the claim.
3. For well-established medical facts (common disease symptoms, standard drug classes, basic physiology, diagnostic criteria taught in medical school), you MAY use your general clinical knowledge even if the supplied abstracts only tangentially cover the topic. Mark such sentences with the suffix [general clinical knowledge].
4. Combine source-cited specifics with general knowledge fluently — do not write two separate sections.
5. If multiple sources discuss the same point, cite the strongest 1-2.
6. Use plain clinical language. No marketing tone, no hype.
7. Do NOT invent specific dosages, statistics, or trial outcomes that are not in the sources.
8. Set status to "insufficient_evidence" ONLY when the question is genuinely outside mainstream medicine AND no source covers it. For mainstream clinical questions, always answer.

Output ONLY valid JSON, no prose, no markdown fences."""


SYNTH_USER_TMPL = """Doctor question:
{question}

Internal document chunks (uploaded PDFs):
{internal_json}

External peer-reviewed sources (PubMed):
{external_json}

Return this exact JSON shape:
{{
  "answer": "string (the clinical answer with inline [source_id] citations)",
  "status": "answered|partial|insufficient_evidence",
  "citedSourceIds": ["source_id_1", "source_id_2"]
}}"""


def synthesize_from_sources(
    question: str,
    *,
    internal_chunks: List[dict],
    references: List[Reference],
) -> Tuple[str, str, List[str]]:
    """One-shot answer synthesis from sources. Returns (answer, status, citedSourceIds)."""
    user_msg = SYNTH_USER_TMPL.format(
        question=question,
        internal_json=json.dumps(_internal_payload(internal_chunks), ensure_ascii=False),
        external_json=json.dumps(_external_payload(references), ensure_ascii=False),
    )
    try:
        result = _llm().invoke(
            [
                {"role": "system", "content": SYNTH_SYSTEM},
                {"role": "user", "content": user_msg},
            ]
        )
    except Exception as exc:
        logger.warning(f"Synthesizer LLM call failed: {exc}")
        return ABSTENTION_SENTENCE, "insufficient_evidence", []

    parsed = _extract_json(getattr(result, "content", "") or str(result))
    if not isinstance(parsed, dict):
        logger.warning("Synthesizer returned non-JSON response")
        return ABSTENTION_SENTENCE, "insufficient_evidence", []

    answer = str(parsed.get("answer") or "").strip()
    status = parsed.get("status") or "answered"
    cited = parsed.get("citedSourceIds") or []
    if not isinstance(cited, list):
        cited = []

    if status not in ("answered", "partial", "insufficient_evidence"):
        status = "answered"

    if status == "insufficient_evidence":
        return ABSTENTION_SENTENCE, "insufficient_evidence", []
    if not answer:
        return ABSTENTION_SENTENCE, "insufficient_evidence", []

    return answer, status, [str(c) for c in cited]
