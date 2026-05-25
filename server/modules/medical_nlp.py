"""Medical-term normalization using scispaCy + UMLS EntityLinker.

Lazy-loads `en_core_sci_lg` and attaches the `scispacy_linker` pipe with the UMLS
knowledge base. Used to rewrite noisy user queries (e.g. "hart atack medisin")
into UMLS-canonical phrasing before retrieval / external connector calls.

The module fails open: if scispaCy or its model is not installed, every call
returns the input unchanged and logs a single warning.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Dict, List

from logger import logger

_NLP = None
_DISABLED = False
_LOCK = threading.Lock()

_LINKER_NAME = os.getenv("SCISPACY_LINKER", "umls")
_LINKER_THRESHOLD = float(os.getenv("SCISPACY_LINKER_THRESHOLD", "0.7"))
_MODEL_NAME = os.getenv("SCISPACY_MODEL", "en_core_sci_lg")


def _load():
    global _NLP, _DISABLED
    if _DISABLED or _NLP is not None:
        return _NLP
    with _LOCK:
        if _DISABLED or _NLP is not None:
            return _NLP
        try:
            import spacy
            import scispacy  # noqa: F401
            from scispacy.linking import EntityLinker  # noqa: F401
        except Exception as exc:
            logger.warning(
                f"scispaCy not installed ({exc}); medical normalization disabled"
            )
            _DISABLED = True
            return None
        try:
            nlp = spacy.load(_MODEL_NAME)
        except Exception as exc:
            logger.warning(
                f"scispaCy model '{_MODEL_NAME}' not loadable ({exc}); "
                "medical normalization disabled"
            )
            _DISABLED = True
            return None
        try:
            nlp.add_pipe(
                "scispacy_linker",
                config={
                    "resolve_abbreviations": True,
                    "linker_name": _LINKER_NAME,
                    "threshold": _LINKER_THRESHOLD,
                },
            )
        except Exception as exc:
            logger.warning(
                f"scispaCy EntityLinker unavailable ({exc}); continuing with NER only"
            )
        _NLP = nlp
        logger.info(
            f"scispaCy loaded: model={_MODEL_NAME} linker={_LINKER_NAME} "
            f"threshold={_LINKER_THRESHOLD}"
        )
        return _NLP


def extract_medical_entities(text: str) -> List[Dict[str, Any]]:
    """Return medical entities detected in text, each with optional UMLS link."""
    nlp = _load()
    if nlp is None or not text:
        return []
    try:
        doc = nlp(text)
    except Exception as exc:
        logger.warning(f"scispaCy failed on input: {exc}")
        return []
    try:
        linker = nlp.get_pipe("scispacy_linker")
    except Exception:
        linker = None
    out: List[Dict[str, Any]] = []
    for ent in doc.ents:
        item: Dict[str, Any] = {
            "text": ent.text,
            "start": ent.start_char,
            "end": ent.end_char,
            "canonical": None,
            "cui": None,
            "score": None,
        }
        kb_ents = getattr(ent._, "kb_ents", None) or []
        if kb_ents and linker is not None:
            cui, score = kb_ents[0]
            kb_entity = linker.kb.cui_to_entity.get(cui)
            if kb_entity is not None:
                item["canonical"] = kb_entity.canonical_name
                item["cui"] = cui
                item["score"] = float(score)
        out.append(item)
    return out


def normalize_query(text: str) -> Dict[str, Any]:
    """Rewrite spans in `text` to their UMLS canonical names where confidently linked.

    Returns:
        { "original": str, "normalized": str, "entities": [ ... ] }
    Falls back to the original text on any error or when scispaCy is unavailable.
    """
    if not text or not text.strip():
        return {"original": text, "normalized": text, "entities": []}

    entities = extract_medical_entities(text)
    if not entities:
        return {"original": text, "normalized": text, "entities": []}

    normalized = text
    changed = False
    for ent in sorted(entities, key=lambda e: e["start"], reverse=True):
        canon = ent.get("canonical")
        if not canon:
            continue
        span = text[ent["start"] : ent["end"]]
        if canon.strip().lower() == span.strip().lower():
            continue
        normalized = normalized[: ent["start"]] + canon + normalized[ent["end"] :]
        changed = True

    if changed:
        logger.info(
            f"medical-nlp: normalized '{text[:80]}' -> '{normalized[:80]}'"
        )
    return {"original": text, "normalized": normalized, "entities": entities}
