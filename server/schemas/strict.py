from datetime import datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

ABSTENTION_SENTENCE = "I do not have sufficient verified evidence to answer this question."

StrictAnswerStatus = Literal[
    "answered",
    "partial",
    "insufficient_evidence",
    "conflicting_evidence",
]

EvidenceStatus = Literal["sufficient", "partial", "insufficient", "conflicting"]

SourceTier = Literal["A", "B", "C"]


class Reference(BaseModel):
    id: Optional[str] = None
    title: str
    source: str
    sourceType: Optional[str] = None
    url: str
    confidenceScore: int = 0
    publishedAt: Optional[str] = None
    retrievedAt: Optional[str] = None
    keyFindings: List[str] = Field(default_factory=list)
    usedInAnswer: bool = True
    credibilityTier: Optional[SourceTier] = None


class VerificationSummary(BaseModel):
    evidenceStatus: EvidenceStatus = "insufficient"
    unsupportedClaimsRemoved: int = 0
    conflictsDetected: bool = False
    internalRagConfidence: float = 0.0
    externalEvidenceConfidence: float = 0.0


class UiHints(BaseModel):
    leftPanelTitle: str = "References"
    emptyReferencesMessage: str = "No verified sources found"


class StrictAnswerResponse(BaseModel):
    requestId: str
    mode: Literal["strict"] = "strict"
    answer: str
    confidenceScore: int = 0
    status: StrictAnswerStatus
    references: List[Reference] = Field(default_factory=list)
    verification: VerificationSummary = Field(default_factory=VerificationSummary)
    ui: UiHints = Field(default_factory=UiHints)


def abstention_response(
    request_id: str,
    *,
    internal_confidence: float = 0.0,
    external_confidence: float = 0.0,
    conflicts: bool = False,
    evidence_status: EvidenceStatus = "insufficient",
    suggestions: Optional[List[Reference]] = None,
) -> StrictAnswerResponse:
    refs = list(suggestions or [])
    for r in refs:
        r.usedInAnswer = False
    return StrictAnswerResponse(
        requestId=request_id,
        answer=ABSTENTION_SENTENCE,
        confidenceScore=0,
        status="conflicting_evidence" if evidence_status == "conflicting" else "insufficient_evidence",
        references=refs,
        verification=VerificationSummary(
            evidenceStatus=evidence_status,
            unsupportedClaimsRemoved=0,
            conflictsDetected=conflicts,
            internalRagConfidence=internal_confidence,
            externalEvidenceConfidence=external_confidence,
        ),
        ui=UiHints(
            emptyReferencesMessage=(
                "No related sources found on PubMed for this query."
                if not refs
                else "Suggested reading (verifier could not confirm a direct answer)"
            )
        ),
    )


def now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
