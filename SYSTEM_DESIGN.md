# MediBot — Comprehensive System Design Documentation

> **Version:** 1.0.0 | **Last Updated:** May 2026 | **Status:** Active Development

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [Tech Stack](#3-tech-stack)
4. [System Components](#4-system-components)
5. [Data Flow & Pipeline](#5-data-flow--pipeline)
6. [API Reference](#6-api-reference)
7. [AI / LLM Pipeline](#7-ai--llm-pipeline)
8. [Vector Database & Embeddings](#8-vector-database--embeddings)
9. [Evidence Verification Layer](#9-evidence-verification-layer)
10. [Confidence Scoring System](#10-confidence-scoring-system)
11. [External Connectors](#11-external-connectors)
12. [Frontend (Streamlit Client)](#12-frontend-streamlit-client)
13. [Security & Safety Design](#13-security--safety-design)
14. [Environment & Configuration](#14-environment--configuration)
15. [Deployment Architecture](#15-deployment-architecture)
16. [Future Integrations](#16-future-integrations)
17. [Known Issues & Fixes](#17-known-issues--fixes)

---

## 1. Project Overview

**MediBot** is an AI-powered Medical Assistant Chatbot designed to assist **doctors and clinical professionals** with evidence-grounded answers to medical questions.

### Core Goals

- Provide fast, accurate answers to clinical questions
- Ground responses in peer-reviewed literature (PubMed) and internal clinical PDFs
- Operate in two modes: **General Mode** (fast, RAG-based) and **Strict Mode** (evidence-verified, citation-backed)
- Never hallucinate — fail safely with abstention when evidence is insufficient
- Surface references, confidence scores, and source citations for every answer

### Who It's For

| Audience | Use Case |
|---|---|
| Doctors / Clinicians | Quick reference during consultations |
| Medical Researchers | Literature-backed Q&A with citations |
| Hospital Admin | Upload internal clinical PDFs for querying |

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     CLIENT LAYER                        │
│              Streamlit Web Application                  │
│   ┌──────────┐  ┌──────────────┐  ┌─────────────────┐  │
│   │ PDF      │  │  Chat UI     │  │ History         │  │
│   │ Uploader │  │  (chatUI.py) │  │ Download        │  │
│   └──────────┘  └──────────────┘  └─────────────────┘  │
└─────────────────────────┬───────────────────────────────┘
                          │ HTTP (REST)
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    SERVER LAYER                         │
│                FastAPI Application                      │
│  ┌──────────────┐  ┌────────────┐  ┌─────────────────┐ │
│  │ /upload_pdfs │  │ /ask/      │  │ /ask/strict/    │ │
│  │ (PDF Ingest) │  │ (General)  │  │ (Strict Mode)   │ │
│  └──────────────┘  └────────────┘  └─────────────────┘ │
│                          │                              │
│              ┌───────────┴───────────┐                  │
│              │                       │                  │
│      ┌───────▼──────┐    ┌──────────▼──────────┐       │
│      │ General Mode │    │   Strict Orchestrator│       │
│      │   (llm.py)   │    │(strict_orchestrator) │       │
│      └───────┬──────┘    └──────────┬──────────┘       │
└──────────────┼───────────────────────┼──────────────────┘
               │                       │
    ┌──────────▼──────┐    ┌───────────▼──────────────┐
    │  Groq LLM       │    │   Groq LLM (Verify+Gen)  │
    │ (llama-3.3-70b) │    │   + PubMed Connector      │
    └──────────┬──────┘    └───────────┬──────────────┘
               │                       │
               └───────────┬───────────┘
                           │
               ┌───────────▼───────────┐
               │   Pinecone Vector DB   │
               │  (768-dim, dot-product)│
               └───────────────────────┘
```

---

## 3. Tech Stack

### Backend

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| **Web Framework** | FastAPI | Latest | REST API, async request handling |
| **ASGI Server** | Uvicorn | Latest | Production ASGI server |
| **Language** | Python | 3.11+ | Core application language |
| **LLM Runtime** | LangChain | Latest | LLM orchestration & chain management |
| **LLM Provider** | Groq (via `langchain-groq`) | Latest | Ultra-fast LLM inference |
| **LLM Model** | Llama 3.3 70B Versatile | llama-3.3-70b-versatile | Primary language model |
| **Embeddings** | Google Gemini Embedding 001 | models/gemini-embedding-001 | 768-dim text embeddings |
| **Vector Store** | Pinecone Serverless | Latest | Semantic search / RAG retrieval |
| **Data Validation** | Pydantic v2 | Latest | Request/response schemas |
| **PDF Parsing** | PyPDF + LangChain Community | Latest | PDF ingestion & text extraction |
| **Text Splitting** | LangChain RecursiveCharacterTextSplitter | Latest | Chunk documents for embedding |
| **HTTP Client** | Requests | Latest | External API calls (PubMed) |
| **Logging** | Loguru | Latest | Structured application logging |
| **Environment** | python-dotenv | Latest | .env configuration management |

### Frontend

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| **UI Framework** | Streamlit | Latest | Web UI for the chatbot |
| **HTTP Client** | Requests | Latest | Calls to FastAPI backend |

### External Services

| Service | Purpose | Auth |
|---|---|---|
| **Groq API** | LLM inference (Llama 3.3 70B) | API Key |
| **Google Generative AI API** | Text embeddings (Gemini Embedding 001) | API Key |
| **Pinecone** | Vector database (serverless, AWS us-east-1) | API Key |
| **PubMed E-utilities (NCBI)** | Peer-reviewed literature retrieval | Optional API Key |

### Infrastructure

| Component | Technology |
|---|---|
| **Package Manager** | `uv` (ultrafast Python package manager) |
| **Dependency Isolation** | Python venv (`.venv`) |
| **Source Control** | Git |

---

## 4. System Components

### 4.1 Server Components

#### `server/main.py`
FastAPI application entry point. Registers all routers and middleware.

#### `server/modules/strict_orchestrator.py`
The core pipeline engine for **Strict Mode**. Orchestrates:
1. Pinecone retrieval
2. Confidence scoring
3. PubMed fallback (when internal confidence is low)
4. Verification agent (LLM call)
5. Response generation agent (LLM call)
6. Abstention logic (fail-safe)

#### `server/modules/verification.py`
Houses two LLM agents:
- **Verification Agent** — checks if evidence supports the question
- **Response Generator Agent** — writes the final answer using only verified claims

#### `server/modules/confidence.py`
Confidence scoring utilities:
- `score_pinecone_matches()` — scores internal RAG results
- `combined_confidence()` — merges internal + external scores
- `threshold_band()` — maps score to action (answer/partial/abstain)

#### `server/modules/llm.py`
General mode LLM chain using `RetrievalQA` from LangChain Classic.

#### `server/modules/load_vectorstore.py`
PDF ingestion pipeline: load → split → embed → upsert to Pinecone.

#### `server/modules/connectors/pubmed.py`
PubMed E-utilities connector. Supports:
- `esearch` → get PMIDs
- `esummary` → metadata (title, journal, date)
- `efetch` → abstract text
- LLM-powered query rewriting for better PubMed hits

#### `server/schemas/strict.py`
Pydantic models for the strict-mode response: `StrictAnswerResponse`, `Reference`, `VerificationSummary`, `UiHints`.

### 4.2 Client Components

#### `client/app.py`
Main Streamlit app entry point.

#### `client/components/chatUI.py`
Chat interface with two-column layout:
- Left: Chat messages with status pills and confidence badges
- Right: Reference panel with PubMed/internal source cards

#### `client/components/upload.py`
PDF upload widget that calls `/upload_pdfs/` on the backend.

#### `client/utils/api.py`
HTTP wrapper functions for all API endpoints.

---

## 5. Data Flow & Pipeline

### 5.1 PDF Upload & Indexing

```
User uploads PDF(s)
       │
       ▼
FastAPI /upload_pdfs/
       │
       ▼
PyPDFLoader (LangChain) — parse pages
       │
       ▼
RecursiveCharacterTextSplitter
  chunk_size=500, chunk_overlap=50
       │
       ▼
Gemini Embedding 001 (768-dim)
       │
       ▼
Pinecone Upsert
  index: "medicalindex"
  metric: dotproduct
  cloud: AWS us-east-1
```

### 5.2 Strict Mode Query Pipeline

```
User question
     │
     ▼
[1] Pinecone semantic search (top_k=5)
     │
     ▼
[2] Internal confidence scoring
  score = 0.7 * top_match + 0.3 * volume_factor
     │
     ├──(score < 0.50 or no relevant matches)──▶ [3] PubMed fallback
     │                                                 │
     │              esearch → esummary → efetch        │
     │                   (abstracts)                   │
     │                        │                        │
     └────────────────────────┘                        │
              all_refs = internal + external            │
                        │                              │
                        ▼                              │
     [4] Verification Agent (Groq LLM)                 │
       Input: question + internal chunks + pubmed refs  │
       Output: {evidenceStatus, verifiedClaims,         │
                mustAbstain, conflictsDetected}          │
                        │                              │
            ┌───────────┴───────────┐                  │
      mustAbstain=true?       evidenceStatus=          │
            │                 sufficient/partial        │
            ▼                       │                  │
      Return abstention      [5] Response Generator    │
      with suggestions            (Groq LLM)           │
                                   │                   │
                            answer + citedSourceIds    │
                                   │                   │
                    [6] combined_confidence() scoring  │
                                   │                   │
                    [7] threshold_band() check         │
                                   │                   │
                    [8] Return StrictAnswerResponse    │
```

### 5.3 General Mode Query Pipeline

```
User question
     │
     ▼
Pinecone query (top_k=3)
     │
     ▼
LangChain RetrievalQA
  (Groq Llama 3.3 70B)
     │
     ▼
{response, sources}
```

---

## 6. API Reference

### `POST /upload_pdfs/`
Upload one or more PDFs to be embedded and stored in Pinecone.

**Request:** `multipart/form-data`
- `files`: list of PDF files

**Response:**
```json
{"message": "PDFs uploaded and indexed successfully"}
```

---

### `POST /ask/`
General-mode question answering (RAG + LLM, no strict verification).

**Request:** `application/x-www-form-urlencoded`
- `question`: string

**Response:**
```json
{
  "response": "Answer text here",
  "sources": ["source1.pdf", "source2.pdf"]
}
```

---

### `POST /ask/strict/`
Strict evidence-verified question answering.

**Request:** `application/x-www-form-urlencoded`
- `question`: string

**Response:** `StrictAnswerResponse`
```json
{
  "requestId": "uuid",
  "mode": "strict",
  "answer": "string",
  "confidenceScore": 75,
  "status": "answered | partial | insufficient_evidence | conflicting_evidence",
  "references": [
    {
      "id": "pubmed_12345",
      "title": "Paper title",
      "source": "PubMed",
      "sourceType": "peer_reviewed_journal",
      "url": "https://pubmed.ncbi.nlm.nih.gov/12345/",
      "confidenceScore": 80,
      "publishedAt": "2023-04-15",
      "retrievedAt": "2026-05-06T06:00:00Z",
      "keyFindings": ["Abstract text..."],
      "usedInAnswer": true,
      "credibilityTier": "A"
    }
  ],
  "verification": {
    "evidenceStatus": "sufficient | partial | insufficient | conflicting",
    "unsupportedClaimsRemoved": 0,
    "conflictsDetected": false,
    "internalRagConfidence": 0.72,
    "externalEvidenceConfidence": 0.65
  },
  "ui": {
    "leftPanelTitle": "References",
    "emptyReferencesMessage": "No verified sources found"
  }
}
```

---

## 7. AI / LLM Pipeline

### Model: Llama 3.3 70B Versatile (via Groq)

**Why Groq?**
- Sub-second inference latency (crucial for clinical use)
- Free tier sufficient for development
- Llama 3.3 70B is state-of-the-art open-weight model with strong medical reasoning

**Temperature Settings:**
- General mode: default (balanced)
- Verification agent: `temperature=0` (deterministic, no creativity)
- Response generator: `temperature=0` (deterministic)

### Agent Architecture (Strict Mode)

Two separate LLM calls per query:

| Agent | Role | Key Constraint |
|---|---|---|
| **Verification Agent** | Assess if evidence supports the question | Must NOT generate the answer |
| **Response Generator** | Write clinical answer from verified claims only | Must NOT add knowledge beyond verified claims |

### Prompt Design Philosophy

- **Verification Agent** is permissive on typos and phrasing — focuses on evidence quality
- **Response Generator** is strict — only uses verified claims, outputs caveats
- Both agents output **JSON only** — no prose, no markdown in model output
- Fail-closed: any parsing error → abstention

---

## 8. Vector Database & Embeddings

### Pinecone Configuration

| Parameter | Value |
|---|---|
| Index Name | `medicalindex` |
| Dimension | 768 |
| Metric | `dotproduct` |
| Cloud | AWS |
| Region | `us-east-1` |
| Tier | Serverless |

### Embedding Model: Gemini Embedding 001

| Parameter | Value |
|---|---|
| Model | `models/gemini-embedding-001` |
| Output Dimensions | 768 |
| Provider | Google Generative AI |

### Chunking Strategy

| Parameter | Value |
|---|---|
| Chunk Size | 500 tokens |
| Chunk Overlap | 50 tokens |
| Splitter | `RecursiveCharacterTextSplitter` |

### Retrieval

- **General mode:** `top_k=3`
- **Strict mode:** `top_k=5` (configurable via `STRICT_TOP_K` env var)

---

## 9. Evidence Verification Layer

### Verification Agent Output Schema

```json
{
  "evidenceStatus": "sufficient | partial | insufficient | conflicting",
  "verifiedClaims": [
    {
      "claim": "string",
      "supportingSourceIds": ["string"],
      "supportStrength": "strong | moderate | weak"
    }
  ],
  "rejectedClaims": [
    {"claim": "string", "reason": "unsupported | conflicting | not_applicable | too_old"}
  ],
  "conflictsDetected": false,
  "mustAbstain": false,
  "abstentionReason": null
}
```

### Abstention Rules

The system **abstains** (refuses to answer) when:
1. Verification agent sets `mustAbstain: true`
2. `evidenceStatus` is `"insufficient"` or `"conflicting"`
3. Response generator returns `status` of `"insufficient_evidence"` or `"conflicting_evidence"`
4. Generator cited sources but none match the reference list
5. Final `combined_confidence` score < 50 (maps to `"abstain"` band)

### Evidence Source Tiers

| Tier | Source | Example |
|---|---|---|
| **A** | Peer-reviewed journals | PubMed articles |
| **B** | Clinical documents | Internal PDFs (curated) |
| **C** | General reference | Other sources |

---

## 10. Confidence Scoring System

### Internal (RAG) Confidence

```python
volume_factor = min(relevant_matches, 3) / 3.0
aggregate = 0.7 * top_match_score + 0.3 * volume_factor
```

- `relevant_matches`: number of matches scoring ≥ 0.55
- `answerable`: top ≥ 0.85 AND relevant ≥ 2

### Thresholds

| Constant | Value | Meaning |
|---|---|---|
| `INTERNAL_ANSWER_THRESHOLD` | 0.85 | Top match score for direct answer |
| `INTERNAL_FALLBACK_THRESHOLD` | 0.50 | Trigger PubMed fallback below this |
| `RELEVANT_MATCH_THRESHOLD` | 0.55 | Min score to count as "relevant" |

### Combined Confidence

```python
base = max(internal, external)
if both > 0:
    base = 0.5 * internal + 0.5 * external
pct = int(round(base * 100))
if conflicts: pct -= 15
pct -= min(unsupported_claims, 5) * 4
```

### Threshold Bands

| Band | Score Range | Action |
|---|---|---|
| `answer_full` | ≥ 85 | Full answer |
| `answer_strong_only` | 70–84 | Strong claims only |
| `answer_partial_or_abstain` | 50–69 | Partial answer |
| `abstain` | < 50 | Refuse to answer |

---

## 11. External Connectors

### PubMed E-utilities Connector

**Endpoint:** `https://eutils.ncbi.nlm.nih.gov/entrez/eutils`

**3-Step Retrieval:**
1. `esearch.fcgi` — get PMIDs for query (sorted by relevance)
2. `esummary.fcgi` — get title, journal, publication date
3. `efetch.fcgi` — get full abstract text (XML parsing)

**Smart Query Rewriting:**
- Step 1: Try original query
- Step 2: Stopword-filtered simplified query
- Step 3: LLM-rewritten PubMed-optimized query (Groq)

**Rate Limiting:**
- Without API key: 340ms sleep between calls (NCBI limit: 3 req/s)
- With `NCBI_API_KEY`: 10 req/s allowed

**Config:**
- `STRICT_EXTERNAL_RETMAX=5` — max PubMed results per query
- `NCBI_API_KEY` — optional, increases rate limit
- `NCBI_CONTACT_EMAIL` — required by NCBI policy

---

## 12. Frontend (Streamlit Client)

### Layout

```
┌─────────────────────────────────────────────────┐
│  🩺 Medical Assistant Chatbot                   │
│  [Strict evidence mode toggle] [?]              │
├───────────────────────────┬─────────────────────┤
│  CHAT AREA (2/3 width)    │  REFERENCES (1/3)   │
│                           │                     │
│  [user] tell me about...  │  [1] Paper Title    │
│  [assistant] ●Answered    │      PubMed · 2023  │
│              Confidence:  │      Open ↗          │
│              75/100       │      Abstract...     │
│              Answer text  │                     │
│              [1] PubMed   │  [2] Paper Title    │
│                           │      ...            │
│  [Type your question...]  │                     │
└───────────────────────────┴─────────────────────┘
```

### Status Pills

| Status | Color | Meaning |
|---|---|---|
| `Answered` | Green `#16a34a` | Full evidence-backed answer |
| `Partial` | Amber `#d97706` | Partial evidence answer |
| `No verified answer` | Gray `#6b7280` | Abstained, suggestions shown |
| `Conflicting evidence` | Red `#dc2626` | Sources contradict each other |

---

## 13. Security & Safety Design

### Medical Safety

- **Fail-closed** architecture — uncertain = abstain, never guess
- **No diagnosis or prescriptions** — system explicitly forbids diagnostic claims
- **Clinical caveats** appended to every answer
- **Source citations mandatory** — every claim must be linked to a source
- **Strict mode by default** in the UI

### API Security

- CORS configured (currently `*` for dev — restrict in production)
- Environment variables for all secrets (never in code)
- Exception middleware catches and logs all errors without leaking internals

### Data Privacy

- Uploaded PDFs stored locally in `server/uploaded_docs/`
- No patient data should ever be uploaded (stated in documentation)

---

## 14. Environment & Configuration

### Server `.env` Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ | Groq LLM inference API key |
| `GOOGLE_API_KEY` | ✅ | Google Generative AI (for embeddings) |
| `PINECONE_API_KEY` | ✅ | Pinecone vector database |
| `PINECONE_INDEX_NAME` | ✅ | Pinecone index name (e.g. `medicalindex`) |
| `GROQ_MODEL` | Optional | LLM model name (default: `llama-3.3-70b-versatile`) |
| `STRICT_TOP_K` | Optional | Pinecone top-k for strict mode (default: `5`) |
| `STRICT_EXTERNAL_RETMAX` | Optional | PubMed max results (default: `5`) |
| `NCBI_API_KEY` | Optional | NCBI API key (higher rate limits) |
| `NCBI_CONTACT_EMAIL` | Optional | Contact email for NCBI (default: `noreply@example.com`) |

### Client Config (`client/config.py`)

| Variable | Default | Description |
|---|---|---|
| `API_URL` | `http://localhost:8000` | FastAPI backend URL |

---

## 15. Deployment Architecture

### Local Development

```bash
# Terminal 1 — Start backend
cd server
uvicorn main:app --reload --port 8000

# Terminal 2 — Start frontend
cd client
streamlit run app.py
```

### Production (Recommended)

```
[Nginx Reverse Proxy]
     │
     ├──▶ [Streamlit] :8501  (client)
     │
     └──▶ [Uvicorn/Gunicorn] :8000  (FastAPI server)

[Pinecone Serverless] ← managed cloud service
[Groq Cloud API]      ← managed LLM inference
[Google AI API]       ← managed embeddings
[PubMed NCBI API]     ← public, free
```

### Docker (Planned)

```
docker-compose.yml
├── service: medibot-server  (Python/FastAPI)
└── service: medibot-client  (Python/Streamlit)
```

---

## 16. Future Integrations

> These are planned integrations that are currently being integrated into the codebase.

### 16.1 OpenFDA Drug Database Connector

**Purpose:** Real-time drug information, interactions, and FDA approval data.

**Implementation:**
```python
# server/modules/connectors/openfda.py
# Endpoint: https://api.fda.gov/drug/
# Searches: drug labels, adverse events, recalls
```

**Integration Point:** Falls back to OpenFDA when PubMed results are insufficient for drug-specific questions.

---

### 16.2 WHO ICD-11 API Connector

**Purpose:** Standardized disease classification and coding.

**Endpoint:** `https://icd.who.int/icdapi`

**Use Case:** Auto-classify diseases in questions, enrich answers with ICD-11 codes, ensure terminology consistency.

---

### 16.3 ClinicalTrials.gov Connector

**Purpose:** Retrieve ongoing and completed clinical trials relevant to the question.

**Endpoint:** `https://clinicaltrials.gov/api/v2/studies`

**Integration:** Adds `sourceType: "clinical_trial"` references alongside PubMed results.

---

### 16.4 Semantic Scholar API

**Purpose:** Academic literature beyond PubMed — computer science, biomedical preprints.

**Endpoint:** `https://api.semanticscholar.org/graph/v1/`

**Why:** Captures cutting-edge research not yet indexed in PubMed.

---

### 16.5 MedlinePlus Connect

**Purpose:** Patient-friendly medical information from the US National Library of Medicine.

**Endpoint:** `https://connect.medlineplus.gov/`

**Use Case:** Non-strict mode answers for patient-facing queries.

---

### 16.6 UMLS (Unified Medical Language System)

**Purpose:** Medical concept normalization and synonym expansion.

**Use Case:** Expand query "heart attack" → `["myocardial infarction", "MI", "AMI"]` before PubMed search, dramatically improving recall.

**Auth:** UMLS API key (free registration at uts.nlm.nih.gov)

---

### 16.7 Hybrid Reranker (ColBERT / Cross-Encoder)

**Purpose:** Improve RAG retrieval relevance beyond cosine similarity.

**Implementation:**
- First pass: Pinecone ANN retrieval (fast, top 20)
- Second pass: Cross-encoder reranking (accurate, top 5)

**Models:** `cross-encoder/ms-marco-MiniLM-L-6-v2` or Cohere Rerank API.

---

### 16.8 Streaming Responses (SSE)

**Purpose:** Stream the LLM answer token-by-token to the UI for better UX.

**Implementation:** FastAPI `StreamingResponse` + Streamlit `st.write_stream()`.

---

### 16.9 Authentication & Multi-Tenancy

**Purpose:** Doctor accounts, session management, per-institution PDF libraries.

**Stack:** FastAPI + OAuth2 + JWT tokens + PostgreSQL (user/session storage).

---

### 16.10 Observability Stack

**Purpose:** Production monitoring of LLM calls, latency, error rates.

**Stack:**
- **LangSmith** — LLM call tracing and evaluation
- **Prometheus + Grafana** — API metrics dashboards
- **Sentry** — Error tracking and alerting

---

## 17. Known Issues & Fixes

### Issue: Strict mode returns "I do not have sufficient verified evidence" for well-known medical questions

**Root Cause (Multi-layered):**

1. `verification.py` — `mustAbstain` defaults to `True` in the prompt template JSON shape, so if the LLM output is ambiguous, it always abstains.
2. `strict_orchestrator.py` line 162 — `bool(verification.get("mustAbstain", True))` — Python default is `True`, meaning any missing key causes abstention.
3. The Verification Agent is not being told to use its **general medical knowledge** to assess whether evidence is "sufficient" — it overly restricts to only the retrieved chunks.
4. `threshold_band()` — anything below 50% combined confidence returns `"abstain"`, and combined confidence is penalized aggressively.

**Fix Applied:**
- Changed `mustAbstain` default in orchestrator from `True` → `False`
- Updated Verification Agent prompt to treat peer-reviewed PubMed abstracts as independently sufficient
- Updated Response Generator to use general medical knowledge as a fallback when `evidenceStatus` is `"partial"` and the question is about well-established medical facts
- Lowered abstention threshold to allow `"answer_partial_or_abstain"` band to actually answer

**Files Changed:**
- `server/modules/strict_orchestrator.py`
- `server/modules/verification.py`

---

*This document is the single source of truth for MediBot's architecture and design decisions. Keep it updated as the system evolves.*
