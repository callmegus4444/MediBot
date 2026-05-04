"""Strict-mode orchestrator.

Pipeline:
  1. Embed question -> Pinecone top_k retrieval.
  2. Score internal RAG confidence.
  3. If confidence too low, fan out to external connectors (PubMed first).
  4. Run verification agent on combined evidence.
  5. Run response generator agent. Fail closed on any error.
"""
from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Tuple

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone

from logger import logger
from modules.confidence import (
    INTERNAL_FALLBACK_THRESHOLD,
    combined_confidence,
    score_pinecone_matches,
    threshold_band,
    to_percent,
)
from modules.connectors import pubmed
from modules.verification import generate, verify
from schemas.strict import (
    ABSTENTION_SENTENCE,
    Reference,
    StrictAnswerResponse,
    UiHints,
    VerificationSummary,
    abstention_response,
    now_iso,
)

PINECONE_TOP_K = int(os.getenv("STRICT_TOP_K", "5"))
EXTERNAL_RETMAX = int(os.getenv("STRICT_EXTERNAL_RETMAX", "5"))


def _retrieve_internal(question: str) -> Tuple[List[dict], Any]:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
    embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embedded = embed_model.embed_query(question)
    res = index.query(vector=embedded, top_k=PINECONE_TOP_K, include_metadata=True)
    matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])
    return matches or [], res


def _internal_chunks_payload(matches: List[dict]) -> List[dict]:
    payload: List[dict] = []
    for i, match in enumerate(matches):
        if not isinstance(match, dict):
            continue
        meta = match.get("metadata") or {}
        text = meta.get("text") or ""
        payload.append(
            {
                "id": match.get("id") or f"internal_{i}",
                "documentTitle": meta.get("source") or meta.get("documentTitle") or meta.get("file_name"),
                "page": meta.get("page"),
                "text": text,
                "score": match.get("score"),
            }
        )
    return payload


def _internal_chunks_to_references(internal_chunks: List[dict]) -> List[Reference]:
    refs: List[Reference] = []
    retrieved_at = now_iso()
    for i, chunk in enumerate(internal_chunks):
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        title = chunk.get("documentTitle") or "Internal document"
        page = chunk.get("page")
        page_str = f"page {page}" if page is not None else None
        refs.append(
            Reference(
                id=str(chunk.get("id") or f"internal_{i}"),
                title=str(title),
                source="Internal Library",
                sourceType="internal_pdf",
                url=f"local://{title}{('#' + str(page)) if page is not None else ''}",
                confidenceScore=to_percent(float(chunk.get("score") or 0.0)),
                publishedAt=None,
                retrievedAt=retrieved_at,
                keyFindings=[page_str] if page_str else [],
                usedInAnswer=False,
                credibilityTier="B",
            )
        )
    return refs


def _filter_used(references: List[Reference], cited_ids: List[str]) -> List[Reference]:
    if not cited_ids:
        return []
    cited = set(cited_ids)
    out: List[Reference] = []
    for ref in references:
        if ref.id and ref.id in cited:
            ref.usedInAnswer = True
            out.append(ref)
    return out


def answer_strict(question: str) -> StrictAnswerResponse:
    request_id = str(uuid.uuid4())
    if not question or not question.strip():
        return abstention_response(request_id)

    try:
        matches, _ = _retrieve_internal(question)
    except Exception as exc:
        logger.exception(f"Internal retrieval failed: {exc}")
        return abstention_response(request_id)

    internal_conf = score_pinecone_matches(matches)
    internal_chunks = _internal_chunks_payload(matches)
    internal_refs = _internal_chunks_to_references(internal_chunks)

    external_refs: List[Reference] = []
    external_score = 0.0
    if internal_conf.needs_external or internal_conf.score < INTERNAL_FALLBACK_THRESHOLD:
        try:
            external_refs = pubmed.search(question, retmax=EXTERNAL_RETMAX)
            if external_refs:
                # Naive external score: presence of any peer-reviewed result yields a moderate floor.
                external_score = min(0.6 + 0.05 * max(0, len(external_refs) - 1), 0.85)
        except Exception as exc:
            logger.warning(f"External search failed: {exc}")
            external_refs = []
            external_score = 0.0

    all_refs: List[Reference] = internal_refs + external_refs

    if not all_refs:
        return abstention_response(
            request_id,
            internal_confidence=internal_conf.score,
            external_confidence=external_score,
        )

    verification = verify(question, internal_chunks=internal_chunks, external_refs=external_refs)
    if not verification:
        return abstention_response(
            request_id,
            internal_confidence=internal_conf.score,
            external_confidence=external_score,
        )

    must_abstain = bool(verification.get("mustAbstain", True))
    evidence_status = verification.get("evidenceStatus", "insufficient")
    conflicts = bool(verification.get("conflictsDetected", False))
    rejected = verification.get("rejectedClaims") or []
    unsupported_count = len([c for c in rejected if isinstance(c, dict)])

    if must_abstain or evidence_status in ("insufficient", "conflicting"):
        return abstention_response(
            request_id,
            internal_confidence=internal_conf.score,
            external_confidence=external_score,
            conflicts=conflicts,
            evidence_status="conflicting" if evidence_status == "conflicting" else "insufficient",
        )

    answer, status, cited_ids, caveats = generate(
        question, verification=verification, references=all_refs
    )

    if status in ("insufficient_evidence", "conflicting_evidence"):
        return abstention_response(
            request_id,
            internal_confidence=internal_conf.score,
            external_confidence=external_score,
            conflicts=conflicts,
            evidence_status="conflicting" if status == "conflicting_evidence" else "insufficient",
        )

    used_refs = _filter_used(all_refs, cited_ids)
    if not used_refs:
        # Strict policy: no references means no answer.
        return abstention_response(
            request_id,
            internal_confidence=internal_conf.score,
            external_confidence=external_score,
            conflicts=conflicts,
        )

    confidence_pct = combined_confidence(
        internal_conf.score, external_score, conflicts=conflicts, unsupported=unsupported_count
    )
    band = threshold_band(confidence_pct)
    if band == "abstain":
        return abstention_response(
            request_id,
            internal_confidence=internal_conf.score,
            external_confidence=external_score,
            conflicts=conflicts,
        )

    if caveats:
        answer = answer.strip()
        if not answer.endswith("."):
            answer += "."
        answer += "\n\nClinical caveats: " + "; ".join(caveats)

    return StrictAnswerResponse(
        requestId=request_id,
        answer=answer,
        confidenceScore=confidence_pct,
        status=status,
        references=used_refs,
        verification=VerificationSummary(
            evidenceStatus=evidence_status if evidence_status in ("sufficient", "partial") else "partial",
            unsupportedClaimsRemoved=unsupported_count,
            conflictsDetected=conflicts,
            internalRagConfidence=internal_conf.score,
            externalEvidenceConfidence=round(external_score, 4),
        ),
        ui=UiHints(),
    )
