"""Strict-mode orchestrator.

Pipeline:
  1. Embed question -> Pinecone top_k retrieval (scoped to a library namespace).
  2. Score internal RAG confidence.
  3. Fan out to PubMed + web + OpenFDA + ClinicalTrials.gov in parallel.
  4. Run synthesizer agent on combined evidence (supports conversation history).
  5. Build StrictAnswerResponse. Fail closed on any error.

Also exposes a streaming variant that yields evidence first, then token deltas.
"""
from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone

from logger import logger
from modules.confidence import (
    INTERNAL_FALLBACK_THRESHOLD,
    combined_confidence,
    score_pinecone_matches,
    to_percent,
)
from modules.connectors import clinicaltrials, openfda, pubmed, web_search
from modules.load_vectorstore import sanitize_library
from modules.verification import (
    stream_synthesize_from_sources,
    synthesize_from_sources,
)
from schemas.strict import (
    Reference,
    StrictAnswerResponse,
    UiHints,
    VerificationSummary,
    abstention_response,
    now_iso,
)

PINECONE_TOP_K = int(os.getenv("STRICT_TOP_K", "5"))
EXTERNAL_RETMAX = int(os.getenv("STRICT_EXTERNAL_RETMAX", "5"))
WEB_RETMAX = int(os.getenv("STRICT_WEB_RETMAX", "5"))
FDA_RETMAX = int(os.getenv("STRICT_FDA_RETMAX", "3"))
TRIALS_RETMAX = int(os.getenv("STRICT_TRIALS_RETMAX", "3"))


def _retrieve_internal(question: str, library: str) -> Tuple[List[dict], Any]:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
    embed_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", output_dimensionality=768
    )
    embedded = embed_model.embed_query(question)
    res = index.query(
        vector=embedded,
        top_k=PINECONE_TOP_K,
        include_metadata=True,
        namespace=library,
    )
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


def _gather_evidence(
    question: str,
    *,
    library: Optional[str],
    use_web: bool,
) -> Dict[str, Any]:
    """Run all retrieval + connectors. Pure data fetch; no LLM yet."""
    namespace = sanitize_library(library)
    try:
        matches, _ = _retrieve_internal(question, namespace)
    except Exception as exc:
        logger.exception(f"Internal retrieval failed: {exc}")
        matches = []

    internal_conf = score_pinecone_matches(matches)
    internal_chunks = _internal_chunks_payload(matches)
    internal_refs = _internal_chunks_to_references(internal_chunks)

    pubmed_refs: List[Reference] = []
    web_refs: List[Reference] = []
    fda_refs: List[Reference] = []
    trial_refs: List[Reference] = []

    with ThreadPoolExecutor(max_workers=4) as pool:
        f_pubmed = pool.submit(pubmed.search, question, retmax=EXTERNAL_RETMAX)
        f_web = pool.submit(web_search.search, question, retmax=WEB_RETMAX) if use_web else None
        f_fda = pool.submit(openfda.search, question, retmax=FDA_RETMAX)
        f_trials = pool.submit(clinicaltrials.search, question, retmax=TRIALS_RETMAX)

        for label, fut in [
            ("pubmed", f_pubmed),
            ("web", f_web),
            ("openfda", f_fda),
            ("trials", f_trials),
        ]:
            if fut is None:
                continue
            try:
                result = fut.result() or []
            except Exception as exc:
                logger.warning(f"{label} connector failed: {exc}")
                result = []
            if label == "pubmed":
                pubmed_refs = result
            elif label == "web":
                web_refs = result
            elif label == "openfda":
                fda_refs = result
            elif label == "trials":
                trial_refs = result

    external_refs: List[Reference] = pubmed_refs + web_refs + fda_refs + trial_refs
    external_score = (
        min(0.6 + 0.05 * max(0, len(external_refs) - 1), 0.9) if external_refs else 0.0
    )

    logger.info(
        f"strict: lib={namespace} internal={len(internal_refs)} pubmed={len(pubmed_refs)} "
        f"web={len(web_refs)} fda={len(fda_refs)} trials={len(trial_refs)}"
    )

    return {
        "library": namespace,
        "internal_chunks": internal_chunks,
        "internal_refs": internal_refs,
        "external_refs": external_refs,
        "all_refs": internal_refs + external_refs,
        "internal_conf": internal_conf,
        "external_score": external_score,
    }


def answer_strict(
    question: str,
    *,
    library: Optional[str] = None,
    use_web: bool = True,
    history: Optional[List[Dict[str, str]]] = None,
) -> StrictAnswerResponse:
    request_id = str(uuid.uuid4())
    if not question or not question.strip():
        return abstention_response(request_id)

    ev = _gather_evidence(question, library=library, use_web=use_web)
    all_refs: List[Reference] = ev["all_refs"]
    internal_conf = ev["internal_conf"]
    external_score = ev["external_score"]

    if not all_refs:
        return abstention_response(
            request_id,
            internal_confidence=internal_conf.score,
            external_confidence=external_score,
            suggestions=[],
        )

    answer, status, cited_ids = synthesize_from_sources(
        question,
        internal_chunks=ev["internal_chunks"],
        references=all_refs,
        history=history or [],
    )

    if status == "insufficient_evidence" or not answer.strip():
        return abstention_response(
            request_id,
            internal_confidence=internal_conf.score,
            external_confidence=external_score,
            suggestions=all_refs,
        )

    used_refs = _filter_used(all_refs, cited_ids) if cited_ids else []
    if not used_refs:
        used_refs = list(all_refs)
        for r in used_refs:
            r.usedInAnswer = False

    has_internal = len(ev["internal_refs"]) > 0
    evidence_status = "sufficient" if has_internal and ev["external_refs"] else "partial"
    confidence_pct = combined_confidence(
        internal_conf.score, external_score, conflicts=False, unsupported=0
    )
    if confidence_pct < 40 and (ev["internal_refs"] or ev["external_refs"]):
        confidence_pct = 55

    return StrictAnswerResponse(
        requestId=request_id,
        answer=answer,
        confidenceScore=confidence_pct,
        status=status,
        references=used_refs,
        verification=VerificationSummary(
            evidenceStatus=evidence_status,
            unsupportedClaimsRemoved=0,
            conflictsDetected=False,
            internalRagConfidence=internal_conf.score,
            externalEvidenceConfidence=round(external_score, 4),
        ),
        ui=UiHints(),
    )


def stream_answer_strict(
    question: str,
    *,
    library: Optional[str] = None,
    use_web: bool = True,
    history: Optional[List[Dict[str, str]]] = None,
) -> Iterable[Dict[str, Any]]:
    """Generator that yields evidence + answer chunks for SSE.

    Yields events of the form {"event": "<name>", "data": {...}}:
      - "meta"       : { requestId, library }
      - "references" : [Reference dicts]
      - "delta"      : { text }
      - "done"       : { confidenceScore, status, references (used) }
      - "error"      : { message }
    """
    request_id = str(uuid.uuid4())
    if not question or not question.strip():
        yield {"event": "error", "data": {"message": "Empty question"}}
        return

    try:
        ev = _gather_evidence(question, library=library, use_web=use_web)
    except Exception as exc:
        logger.exception(f"Stream evidence gathering failed: {exc}")
        yield {"event": "error", "data": {"message": str(exc)}}
        return

    yield {"event": "meta", "data": {"requestId": request_id, "library": ev["library"]}}
    yield {
        "event": "references",
        "data": [r.model_dump() for r in ev["all_refs"]],
    }

    if not ev["all_refs"]:
        yield {
            "event": "done",
            "data": {
                "confidenceScore": 0,
                "status": "insufficient_evidence",
                "references": [],
                "answer": "I do not have sufficient verified evidence to answer this question.",
            },
        }
        return

    collected: List[str] = []
    try:
        for delta in stream_synthesize_from_sources(
            question,
            internal_chunks=ev["internal_chunks"],
            references=ev["all_refs"],
            history=history or [],
        ):
            if not delta:
                continue
            collected.append(delta)
            yield {"event": "delta", "data": {"text": delta}}
    except Exception as exc:
        logger.exception(f"Stream synthesis failed: {exc}")
        yield {"event": "error", "data": {"message": str(exc)}}
        return

    full_text = "".join(collected).strip()

    # Best-effort: try to detect cited ids by scanning brackets in the streamed answer.
    import re as _re
    cited_ids = list({m for m in _re.findall(r"\[([A-Za-z0-9_\-]+)\]", full_text)})
    used_refs = _filter_used(ev["all_refs"], cited_ids) if cited_ids else list(ev["all_refs"])

    internal_conf = ev["internal_conf"]
    external_score = ev["external_score"]
    has_internal = len(ev["internal_refs"]) > 0
    has_external = len(ev["external_refs"]) > 0
    evidence_status = "sufficient" if has_internal and has_external else "partial"
    confidence_pct = combined_confidence(
        internal_conf.score, external_score, conflicts=False, unsupported=0
    )
    if confidence_pct < 40 and (has_internal or has_external):
        confidence_pct = 55
    status = "answered" if full_text else "insufficient_evidence"

    yield {
        "event": "done",
        "data": {
            "requestId": request_id,
            "answer": full_text,
            "confidenceScore": confidence_pct,
            "status": status,
            "references": [r.model_dump() for r in used_refs],
            "verification": {
                "evidenceStatus": evidence_status,
                "internalRagConfidence": internal_conf.score,
                "externalEvidenceConfidence": round(external_score, 4),
                "conflictsDetected": False,
                "unsupportedClaimsRemoved": 0,
            },
        },
    }
