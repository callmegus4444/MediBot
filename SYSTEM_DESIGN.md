# MediBot — Comprehensive System Design Documentation

> **Version:** 2.0.0 | **Last Updated:** May 2026 | **Status:** Active Development

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [Tech Stack](#3-tech-stack)
4. [System Components](#4-system-components)
5. [Data Flow & Pipeline](#5-data-flow--pipeline)
6. [API Reference](#6-api-reference)
7. [AI / LLM Pipeline](#7-ai--llm-pipeline)
8. [Vector Database, Libraries & Embeddings](#8-vector-database-libraries--embeddings)
9. [Evidence Verification Layer](#9-evidence-verification-layer)
10. [Confidence Scoring System](#10-confidence-scoring-system)
11. [External Connectors](#11-external-connectors)
12. [Streaming Pipeline (SSE)](#12-streaming-pipeline-sse)
13. [Conversation Memory](#13-conversation-memory)
14. [Chat History Persistence](#14-chat-history-persistence)
15. [Frontend (Streamlit Client)](#15-frontend-streamlit-client)
16. [Security & Safety Design](#16-security--safety-design)
17. [Environment & Configuration](#17-environment--configuration)
18. [Deployment Architecture](#18-deployment-architecture)
19. [Future Integrations](#19-future-integrations)
20. [Known Issues & Fixes](#20-known-issues--fixes)

---

## 1. Project Overview

**MediBot** is an AI-powered Medical Assistant Chatbot designed to assist **doctors and clinical professionals** with evidence-grounded answers to medical questions.

### Core Goals

- Provide fast, accurate, **cited** answers to clinical questions.
- Ground every response in (a) peer-reviewed PubMed literature, (b) trusted medical websites, (c) FDA drug labels, (d) ClinicalTrials.gov studies, and (e) the doctor's own uploaded PDFs.
- Fail safely with abstention when evidence is insufficient — never hallucinate.
- Stream answers token-by-token so the doctor sees progress immediately.
- Support per-doctor / per-specialty PDF libraries via Pinecone namespaces (no auth, identified by name).

### Who It's For

| Audience | Use Case |
|---|---|
| Doctors / Clinicians | Quick reference during consultations |
| Medical Researchers | Literature-backed Q&A with citations |
| Hospital Admin | Upload internal clinical PDFs per specialty library |

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     CLIENT LAYER                        │
│              Streamlit Web Application                  │
│ ┌──────────┐ ┌────────────┐ ┌──────────┐ ┌───────────┐ │
│ │ Library  │ │ Conversations│ │ Web tog  │ │ Streaming│ │
│ │ selector │ │   sidebar   │ │  switch  │ │ chat UI  │ │
│ └──────────┘ └────────────┘ └──────────┘ └───────────┘ │
└─────────────────────────┬───────────────────────────────┘
                          │ HTTP + Server-Sent Events
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    SERVER LAYER                         │
│                FastAPI Application                      │
│  /upload_pdfs/  /libraries/                            │
│  /ask/  /ask/strict/  /ask/strict/stream/              │
│  /chat/save/  /chat/list/  /chat/{id}/  (DELETE)       │
│                          │                              │
│              ┌───────────┴───────────┐                  │
│              │                       │                  │
│      ┌───────▼──────┐    ┌──────────▼──────────┐       │
│      │ General Mode │    │   Strict Orchestrator│       │
│      │   (llm.py)   │    │(strict_orchestrator) │       │
│      └───────┬──────┘    └──────────┬──────────┘       │
└──────────────┼───────────────────────┼──────────────────┘
               │                       │
   ┌───────────▼──────┐    ┌───────────▼──────────────┐
   │  Pinecone vector │    │  ┌─PubMed─┐ ┌─Tavily Web─┐│
   │  store           │    │  ├─OpenFDA┤ ┌─CT.gov─────┤│
   │  (per-library    │    │  └────────┘ └────────────┘│
   │   namespaces)    │    │      (parallel fan-out)   │
   └──────────────────┘    └──────────────┬────────────┘
                                          ▼
                          Groq Llama 3.3 70B (synthesizer)
                                          │
                                          ▼
                       SSE: meta → references → deltas → done
```

---

## 3. Tech Stack

### Backend

| Layer | Technology | Purpose |
|---|---|---|
| Web framework | FastAPI | REST API + SSE streaming |
| ASGI server | Uvicorn | Production ASGI |
| Language | Python 3.11+ | Core |
| LLM orchestration | LangChain | Chain management |
| LLM provider | Groq (`langchain-groq`) | Ultra-fast inference |
| LLM model | Llama 3.3 70B Versatile | Synthesis, drafting |
| Embeddings | Google Gemini Embedding 001 | 768-dim |
| Vector store | Pinecone Serverless | Semantic search, per-library namespaces |
| Validation | Pydantic v2 | Schemas |
| PDF parsing | PyPDFLoader + RecursiveCharacterTextSplitter | Ingestion |
| HTTP | Requests | All external APIs |
| Logging | Loguru | Structured logs |
| Env | python-dotenv | `.env` loading |

### Frontend

| Layer | Tech | Purpose |
|---|---|---|
| UI | Streamlit | Web UI |
| HTTP/SSE | Requests (stream mode) | Backend calls |

### External Services

| Service | Purpose | Auth |
|---|---|---|
| Groq API | LLM inference | API key |
| Google Generative AI | Embeddings | API key |
| Pinecone | Vector DB | API key |
| PubMed E-utilities (NCBI) | Peer-reviewed literature | Optional key |
| **Tavily** | **Whitelisted medical web search** | **API key** |
| **OpenFDA** | **FDA drug label database** | Optional key |
| **ClinicalTrials.gov API v2** | **Active and completed trials** | None |

### Infrastructure

| Component | Tech |
|---|---|
| Package manager | `uv` |
| Venv | Python `.venv` |
| VCS | Git |

---

## 4. System Components

### 4.1 Server

| File | Purpose |
|---|---|
| `server/main.py` | FastAPI app, CORS, exception middleware, router registration. |
| `server/routes/upload_pdfs.py` | `POST /upload_pdfs/`, `GET /libraries/`. |
| `server/routes/ask_question.py` | `POST /ask/` (legacy general mode). |
| `server/routes/ask_strict.py` | `POST /ask/strict/` and `POST /ask/strict/stream/`. |
| `server/routes/chat_history.py` | Session persistence — save / list / load / delete. |
| `server/modules/strict_orchestrator.py` | Multi-source fan-out + synthesis. Exposes `answer_strict` and `stream_answer_strict`. |
| `server/modules/verification.py` | Synthesizer (one-shot JSON) and streaming synthesizer (plain prose). |
| `server/modules/confidence.py` | Internal RAG and combined-confidence scoring. |
| `server/modules/llm.py` | Legacy general-mode RetrievalQA chain. |
| `server/modules/load_vectorstore.py` | PDF ingestion → embeddings → Pinecone upsert per library namespace. `sanitize_library()` and `list_libraries()`. |
| `server/modules/connectors/pubmed.py` | NCBI E-utilities with LLM query rewriting + fallbacks. |
| `server/modules/connectors/web_search.py` | Tavily search restricted to a whitelist of medical domains. |
| `server/modules/connectors/openfda.py` | FDA drug-label search (indications, dosage, contraindications, warnings, interactions, mechanism). |
| `server/modules/connectors/clinicaltrials.py` | ClinicalTrials.gov v2 study search. |
| `server/schemas/strict.py` | Pydantic models for the strict response. |

### 4.2 Client

| File | Purpose |
|---|---|
| `client/app.py` | Entrypoint: uploader → sessions → history download → chat. |
| `client/components/upload.py` | Library selector + PDF uploader (writes to a named namespace). |
| `client/components/sessions.py` | New chat / list / load / delete saved sessions. |
| `client/components/chatUI.py` | Streaming chat UI with badges, status pills, ref panel. |
| `client/components/history_download.py` | Download current session as `.txt` / `.json`. |
| `client/utils/api.py` | All HTTP + SSE wrappers. |

---

## 5. Data Flow & Pipeline

### 5.1 PDF Upload & Indexing

```
User picks/creates a library name (e.g. "cardiology")
       │
       ▼
FastAPI /upload_pdfs/  (multipart files[] + library)
       │
       ▼
PyPDFLoader → RecursiveCharacterTextSplitter (500 / 50)
       │
       ▼
Gemini Embedding 001 (768-dim)
       │
       ▼
Pinecone upsert   namespace = sanitize_library("cardiology") = "cardiology"
       │
       ▼
Metadata: {text, source=filename, page, library}
```

### 5.2 Strict Mode Query Pipeline (non-stream)

```
User question  +  library  +  use_web  +  history (last 6 turns)
     │
     ▼
[1] Pinecone semantic search   (top_k=5, namespace=library)
     │
     ▼
[2] Score internal confidence
     │
     ├──► [3] PARALLEL connector fan-out (always run):
     │         PubMed  · Tavily Web (if use_web)  · OpenFDA  · ClinicalTrials.gov
     │
     ▼
[4] Synthesizer (Groq Llama 3.3 70B, temperature=0)
        Input: question, history, internal_chunks, all references
        Output JSON: {answer with inline [source_id], status, citedSourceIds}
     │
     ▼
[5] combined_confidence() + status mapping
     │
     ▼
[6] Return StrictAnswerResponse
```

### 5.3 Strict Mode Stream Pipeline (SSE)

```
Same evidence-gathering as above (synchronous fan-out)
     │
     ▼
SSE event:  meta        { requestId, library }
SSE event:  references  [ Reference, ... ]          ← UI renders panel immediately
     │
     ▼
Groq stream() — token deltas
SSE events: delta       { text }                    ← UI appends to chat bubble
…
SSE event:  done        { answer, confidenceScore, status, references (used) }
```

### 5.4 General Mode Query Pipeline (legacy)

```
User question
     │
     ▼
Pinecone (top_k=3) → LangChain RetrievalQA → {response, sources}
```

---

## 6. API Reference

### `POST /upload_pdfs/`
Multipart `files[]` + form `library` (string, optional → "default"). Embeds and upserts into that library's Pinecone namespace.

### `GET /libraries/`
Returns `{ "libraries": ["default", "cardiology", ...] }`.

### `POST /ask/`
Legacy general mode. Form: `question`.

### `POST /ask/strict/`
Form fields:
- `question` — string (required)
- `library` — string (optional, default `"default"`)
- `use_web` — `true|false` (default `true`)
- `history_json` — JSON-encoded list of `{role, content}` (optional)

Response: `StrictAnswerResponse` (see schema below).

### `POST /ask/strict/stream/`
Same inputs as `/ask/strict/`. Returns `text/event-stream` with these events:

| Event | Payload |
|---|---|
| `meta` | `{ requestId, library }` |
| `references` | `[Reference, ...]` (full list of candidates, all sources) |
| `delta` | `{ text }` — repeated as tokens stream |
| `done` | `{ answer, confidenceScore, status, references (used), verification }` |
| `error` | `{ message }` |

### Chat history endpoints

| Endpoint | Purpose |
|---|---|
| `POST /chat/save/` | Body `{session_id, title?, library?, messages: [...]}`. Writes `server/chat_history/<session_id>.json`. |
| `GET /chat/list/` | Returns `[{session_id, title, library, updatedAt, messageCount}, ...]`. |
| `GET /chat/{session_id}/` | Returns the full saved session. |
| `DELETE /chat/{session_id}/` | Removes the session file. |

### `StrictAnswerResponse`

```json
{
  "requestId": "uuid",
  "mode": "strict",
  "answer": "string with inline [source_id] citations",
  "confidenceScore": 75,
  "status": "answered | partial | insufficient_evidence | conflicting_evidence",
  "references": [
    {
      "id": "pubmed_12345 | web_0_cdc_gov | openfda_metformin_0 | trial_NCT01234567 | internal_3",
      "title": "string",
      "source": "PubMed | cdc.gov | OpenFDA | ClinicalTrials.gov | Internal Library",
      "sourceType": "peer_reviewed_journal | web | drug_label | clinical_trial | internal_pdf",
      "url": "string",
      "confidenceScore": 0,
      "publishedAt": "YYYY-MM-DD",
      "retrievedAt": "ISO-8601",
      "keyFindings": ["..."],
      "usedInAnswer": true,
      "credibilityTier": "A | B | C"
    }
  ],
  "verification": {
    "evidenceStatus": "sufficient | partial | insufficient | conflicting",
    "unsupportedClaimsRemoved": 0,
    "conflictsDetected": false,
    "internalRagConfidence": 0.72,
    "externalEvidenceConfidence": 0.85
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

- Sub-second inference, free tier sufficient for development.
- Temperature **0** for both the synthesizer JSON and the streaming synthesizer.

### Agents

| Agent | Role | Output | Constraint |
|---|---|---|---|
| **Synthesizer (JSON)** | Used by `/ask/strict/`. | JSON `{answer, status, citedSourceIds}` | Inline `[source_id]` citations required. |
| **Streaming synthesizer** | Used by `/ask/strict/stream/`. | Plain markdown prose token stream. | Same citation rules. |

### Prompt design

- Synthesizer receives: question, last 6 conversation turns, internal chunks, external references (PubMed + web + FDA + trials).
- Citation tag examples it knows about: `[pubmed_12345]`, `[web_0_cdc_gov]`, `[openfda_metformin_0]`, `[trial_NCT01234567]`, `[internal_3]`.
- Sentences sourced from training-time medical knowledge are tagged `[general clinical knowledge]`.
- Fail-closed: any parsing error → abstention sentence.

---

## 8. Vector Database, Libraries & Embeddings

### Pinecone configuration

| Parameter | Value |
|---|---|
| Index name | `medicalindex` |
| Dimension | 768 |
| Metric | dotproduct |
| Cloud | AWS `us-east-1` |
| Tier | Serverless |
| **Namespaces** | **One per library** (default = `"default"`) |

### Libraries (no auth)

- Library name is sanitized to `[a-z0-9_-]{1,48}` (lowercase). Anything else is converted to `_`.
- Upload writes to the namespace; query reads from it. Different libraries cannot leak chunks into each other.
- `list_libraries()` calls `index.describe_index_stats()` and returns namespace keys.
- No multi-tenancy guarantees — anyone with deployment access can pick any library.

### Embedding model

`models/gemini-embedding-001` from Google Generative AI, 768-dim output. Same model for both ingestion and query.

### Chunking

| Parameter | Value |
|---|---|
| Chunk size | 500 tokens |
| Overlap | 50 tokens |
| Splitter | `RecursiveCharacterTextSplitter` |

### Retrieval

- General mode: `top_k=3`
- Strict mode: `top_k=5` (`STRICT_TOP_K`)

---

## 9. Evidence Verification Layer

Historic note: an earlier design used a separate Verification Agent that gated the Response Generator. v2 collapsed this into a single synthesizer that is given all evidence + history and instructed to cite or abstain. The result is faster, fewer failure modes, and easier streaming.

### Abstention rules

The synthesizer abstains when:

1. No internal chunks AND all external connectors returned 0 results.
2. The output JSON sets `status` to `"insufficient_evidence"`.
3. The output JSON parse fails (fail-closed).
4. The streamed answer is empty.

When abstaining, the response answer is the canonical:

> `"I do not have sufficient verified evidence to answer this question."`

### Evidence source tiers

| Tier | Examples |
|---|---|
| **A** | PubMed peer-reviewed, OpenFDA drug labels, ClinicalTrials.gov, gov / WHO / major-journal websites |
| **B** | Internal PDFs (curated), non-gov medical websites (Mayo, Cleveland Clinic, Medscape, …) |
| **C** | Reserved for future open-web fallback |

---

## 10. Confidence Scoring System

### Internal (RAG) confidence

```python
volume_factor = min(relevant_matches, 3) / 3.0
aggregate = 0.7 * top_match_score + 0.3 * volume_factor
```

- `relevant_matches`: matches scoring ≥ 0.55
- `answerable`: top ≥ 0.85 AND relevant ≥ 2

### Thresholds

| Constant | Value | Meaning |
|---|---|---|
| `INTERNAL_ANSWER_THRESHOLD` | 0.85 | Direct-answer floor |
| `INTERNAL_FALLBACK_THRESHOLD` | 0.50 | Always-run external below this |
| `RELEVANT_MATCH_THRESHOLD` | 0.55 | Min score to count as "relevant" |

### External score (heuristic)

```
external_score = min(0.6 + 0.05 * (n_refs - 1), 0.9)   # if any refs
                 = 0.0                                    # otherwise
```

### Combined confidence

```
base = max(internal, external)
if both > 0: base = 0.5*internal + 0.5*external
pct = round(base * 100)
if conflicts: pct -= 15
pct -= min(unsupported, 5) * 4
```

Plus a soft floor of 55 when we have *any* citations but the formula came out below 40.

### Threshold bands

| Band | Score | Action |
|---|---|---|
| `answer_full` | ≥ 85 | Full answer |
| `answer_strong_only` | 70–84 | Strong claims only |
| `answer_partial_or_abstain` | 50–69 | Partial answer |
| `abstain` | < 50 | Refuse to answer |

---

## 11. External Connectors

All four run concurrently in a `ThreadPoolExecutor(max_workers=4)` inside `_gather_evidence`.

### 11.1 PubMed (`connectors/pubmed.py`)

3-step retrieval: `esearch` → `esummary` → `efetch` (abstracts).

**Smart query rewriting**

1. LLM-rewritten PubMed-optimized query (Groq).
2. Fall back to the raw user query.
3. Fall back to stopword-stripped simplified query.

**Rate limits**

- No key: 340ms sleep between calls (3 rps).
- With `NCBI_API_KEY`: 10 rps.

### 11.2 Tavily Web Search (`connectors/web_search.py`)

POST `https://api.tavily.com/search` with `include_domains` set to a curated whitelist of ~35 medical domains:

> who.int, nih.gov, ncbi.nlm.nih.gov, medlineplus.gov, cdc.gov, fda.gov, ema.europa.eu, nice.org.uk, cochrane.org, cochranelibrary.com, ahrq.gov, mayoclinic.org, clevelandclinic.org, hopkinsmedicine.org, uptodate.com, merckmanuals.com, aafp.org, ama-assn.org, acc.org, heart.org, diabetes.org, cancer.gov, cancer.org, rxlist.com, drugs.com, medscape.com, bmj.com, nejm.org, thelancet.com, jamanetwork.com, nature.com, sciencedirect.com, springer.com, wiley.com

- Gov + top-journal hosts → Tier **A**, others → Tier **B**.
- `sourceType: "web"`.
- Reference IDs are `web_<index>_<host_with_underscores>`.

### 11.3 OpenFDA (`connectors/openfda.py`)

`GET https://api.fda.gov/drug/label.json?search=openfda.generic_name:<term>+openfda.brand_name:<term>`.

- Heuristically extracts candidate drug names from the question (alphabetic tokens ≥ 4 chars, stopword-filtered).
- Tries up to 3 candidates; first hit wins.
- Surfaces up to 4 of: indications, dosage, contraindications, warnings, adverse reactions, interactions, mechanism of action.
- Tier **A**, `sourceType: "drug_label"`.
- Reference IDs are `openfda_<term>_<i>`.

### 11.4 ClinicalTrials.gov v2 (`connectors/clinicaltrials.py`)

`GET https://clinicaltrials.gov/api/v2/studies` with `query.term=<question>` and field-filter.

- Surfaces `briefSummary`, status, phase, conditions, lead sponsor, start date.
- Tier **A**, `sourceType: "clinical_trial"`.
- Reference IDs are `trial_<NCTId>`.

---

## 12. Streaming Pipeline (SSE)

### Server side

`/ask/strict/stream/` returns `StreamingResponse(event_gen(), media_type="text/event-stream")`.

`stream_answer_strict()` yields dicts which the route serializes as SSE frames:

```
event: <name>
data: <json>

```

Pipeline:

1. Run evidence-gathering synchronously (so the UI can show references early).
2. Emit `meta` and `references` immediately.
3. Call `_llm().stream([...])` and yield each `delta`.
4. After the stream ends, scan the collected text for `[source_id]` brackets to build the "used" reference list.
5. Emit `done` with the assembled answer, status, confidence, and used references.

### Client side

`utils/api.py:stream_strict()` consumes the SSE stream line-by-line, accumulating `data:` lines until a blank line, then yielding `{event, data}` dicts.

`chatUI.py` drives the UI:

- On `references` → render the right-hand panel immediately.
- On the first `delta` → flip status pill to "Streaming…".
- On every `delta` → append to the chat bubble.
- On `done` → render the final status pill + confidence, save the message into `st.session_state.messages`, persist via `/chat/save/`.

---

## 13. Conversation Memory

- The client sends the **last 6 turns** of `{role, content}` with each query.
- `verification.py:_history_payload()` trims each turn to ≤ 1500 chars to keep the context bounded.
- Both the JSON synthesizer and the streaming synthesizer receive the history and are explicitly told to use it for resolving pronouns ("that drug", "it", follow-up questions).
- History is *not* embedded into Pinecone — it lives only in the LLM prompt.

---

## 14. Chat History Persistence

### Storage

- One JSON file per session at `server/chat_history/<session_id>.json` (override with `CHAT_HISTORY_DIR`).
- `session_id` is a client-generated UUID hex.
- File shape:

```json
{
  "session_id": "abc...",
  "title": "First user question, trimmed to 60 chars",
  "library": "cardiology",
  "updatedAt": "2026-05-23T03:20:03Z",
  "messages": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "...", "status": "answered",
      "confidenceScore": 87, "references": [ ... ] }
  ]
}
```

### Endpoints

- `POST /chat/save/` — body validated with Pydantic. Sanitizes `session_id` against `^[A-Za-z0-9_-]{1,64}$`.
- `GET /chat/list/` — returns sessions sorted by `updatedAt` desc.
- `GET /chat/{session_id}/` — full session.
- `DELETE /chat/{session_id}/` — removes the file.

### UI behavior

- The Streamlit sidebar lists recent sessions. Click to load (replays prior assistant references too). 🗑 deletes.
- Every assistant turn auto-saves the entire session, so reload is lossless.

---

## 15. Frontend (Streamlit Client)

### Layout

```
┌─────────────────────────── 🩺 MediBot ──────────────────────────────┐
│  Sidebar                                                             │
│  ┌────────────────────────┐                                          │
│  │ 📚 Library: cardiology │                                          │
│  │ [Active library select]│                                          │
│  │ ➕ Create new          │                                          │
│  │ ─────────────          │                                          │
│  │ 📄 Upload PDFs         │                                          │
│  │ ─────────────          │                                          │
│  │ 💬 Conversations       │                                          │
│  │ ➕ New chat            │                                          │
│  │ 📝 Old chat A     🗑   │                                          │
│  │ 📝 Old chat B     🗑   │                                          │
│  │ ─────────────          │                                          │
│  │ ⬇️ Download .txt/.json │                                          │
│  └────────────────────────┘                                          │
│                                                                       │
│  Main                                                                 │
│  Chat with your assistant                       [🌐 Web search] 📚 cardiology│
│  ┌──────────────────────┬─────────────────────────────────────────┐ │
│  │  Chat (2/3)          │  Sources (1/3)                          │ │
│  │  user: ...           │  [1] PubMed · Tier A · 2024-...         │ │
│  │  assistant: streaming│  [2] Web · cdc.gov · Tier A             │ │
│  │   ● Answered  87/100 │  [3] FDA · Tier A (OpenFDA)             │ │
│  │   answer text with   │  [4] Trial · NCT12345 · Tier A          │ │
│  │   inline citations   │  [5] Internal · Tier B                  │ │
│  └──────────────────────┴─────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

### Status pills

| Status | Color | Meaning |
|---|---|---|
| `Answered` | Green `#16a34a` | Full evidence-backed answer |
| `Partial` | Amber `#d97706` | Partial evidence answer |
| `No verified answer` | Gray `#6b7280` | Abstained |
| `Conflicting evidence` | Red `#dc2626` | Sources contradict |
| `Searching sources…` | Slate `#475569` | Pre-stream evidence gathering |
| `Streaming…` | Sky `#0ea5e9` | LLM is generating tokens |

### Source badges

| Source type | Color | Label |
|---|---|---|
| `peer_reviewed_journal` | `#0ea5e9` | PubMed |
| `web` | `#10b981` | Web |
| `internal_pdf` | `#a78bfa` | Internal |
| `drug_label` | `#f59e0b` | FDA |
| `clinical_trial` | `#ec4899` | Trial |

---

## 16. Security & Safety Design

### Medical safety

- **Fail-closed** architecture — any error returns the abstention sentence.
- **No diagnostic claims, no prescriptions** — synthesizer prompt forbids inventing dosages, statistics, or trial outcomes.
- **Inline citations mandatory**; non-cited sentences must be tagged `[general clinical knowledge]`.
- **Strict mode** is the default UI path.

### API security

- CORS currently `*` (dev). Restrict in production.
- All secrets via `.env`; never in code.
- Exception middleware catches and logs without leaking internals.
- Session IDs are validated against `^[A-Za-z0-9_-]{1,64}$` before touching the filesystem (avoids path traversal).

### Data privacy

- Uploaded PDFs stored at `server/uploaded_docs/`.
- Chat history at `server/chat_history/`.
- Patient data must never be uploaded.

---

## 17. Environment & Configuration

### Server `.env`

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ | Groq LLM inference |
| `GOOGLE_API_KEY` | ✅ | Google Generative AI (embeddings) |
| `PINECONE_API_KEY` | ✅ | Pinecone vector DB |
| `PINECONE_INDEX_NAME` | ✅ | e.g. `medicalindex` |
| `TAVILY_API_KEY` | ✅ for web search | Tavily |
| `GROQ_MODEL` | optional | default `llama-3.3-70b-versatile` |
| `STRICT_TOP_K` | optional | Pinecone top-k (default `5`) |
| `STRICT_EXTERNAL_RETMAX` | optional | PubMed max results (default `5`) |
| `STRICT_WEB_RETMAX` | optional | Web max results (default `5`) |
| `STRICT_FDA_RETMAX` | optional | OpenFDA max (default `3`) |
| `STRICT_TRIALS_RETMAX` | optional | Trials max (default `3`) |
| `NCBI_API_KEY` | optional | PubMed 10 rps |
| `NCBI_CONTACT_EMAIL` | optional | NCBI policy |
| `OPENFDA_API_KEY` | optional | Higher OpenFDA rate limit |
| `CHAT_HISTORY_DIR` | optional | Override chat-history dir |

### Client (`client/config.py`)

| Variable | Default | Description |
|---|---|---|
| `API_URL` | `http://localhost:8000` | FastAPI backend URL |

---

## 18. Deployment Architecture

### Local development

```bash
# Terminal 1 — backend
cd server
uvicorn main:app --reload --port 8000

# Terminal 2 — frontend
cd client
streamlit run app.py
```

### Production (recommended)

```
[Nginx Reverse Proxy]
     │
     ├──▶ [Streamlit] :8501  (client)
     │
     └──▶ [Uvicorn / Gunicorn] :8000  (FastAPI server)

[Pinecone Serverless] ← managed cloud
[Groq Cloud API]      ← managed LLM
[Google AI API]       ← managed embeddings
[Tavily / OpenFDA / ClinicalTrials.gov / NCBI] ← public APIs
```

### Docker (planned)

```
docker-compose.yml
├── service: medibot-server  (Python/FastAPI)
└── service: medibot-client  (Python/Streamlit)
```

---

## 19. Future Integrations

### 19.1 UMLS (Unified Medical Language System)

Medical concept normalization and synonym expansion. Example: "heart attack" → `["myocardial infarction", "MI", "AMI"]` before retrieval. Dramatically improves recall.

### 19.2 Hybrid Reranker

ColBERT or cross-encoder rerank on top of Pinecone ANN: fast top-20 from Pinecone, then accurate top-5 from `cross-encoder/ms-marco-MiniLM-L-6-v2` or Cohere Rerank.

### 19.3 Semantic Scholar API

Extends literature coverage to computer science and biomedical preprints not yet in PubMed.

### 19.4 MedlinePlus Connect

Patient-friendly medical info from the US National Library of Medicine — useful for a patient-facing mode.

### 19.5 Authentication & Multi-Tenancy

Doctor accounts, session management, per-institution library scoping. Stack: FastAPI + OAuth2 + JWT + PostgreSQL.

### 19.6 Observability Stack

- **LangSmith** — LLM call tracing and evaluation
- **Prometheus + Grafana** — API metrics
- **Sentry** — Error tracking and alerting

### 19.7 Docker Compose deployment

A `docker-compose.yml` for one-command local deploy.

---

## 20. Known Issues & Fixes

### Issue (resolved): Strict mode used to return "I do not have sufficient verified evidence" for well-known medical questions.

**Root cause** (multi-layered):

1. `verification.py` had a separate Verification Agent whose JSON shape defaulted `mustAbstain: true`.
2. `strict_orchestrator.py` used `verification.get("mustAbstain", True)` so any missing key triggered abstention.
3. The Verification Agent was instructed to only use retrieved chunks, ignoring well-established medical knowledge.
4. `threshold_band()` mapped anything < 50% to "abstain".

**v2 fix** (current state):

- The separate Verification Agent was removed. A single synthesizer is given all evidence + history and instructed to cite or abstain.
- The synthesizer prompt explicitly allows general clinical knowledge for mainstream facts, tagged `[general clinical knowledge]`.
- A soft confidence floor of 55 is applied whenever any citation is present, so the band doesn't collapse to "abstain" for low-but-non-empty evidence.
- Streaming variant uses plain-prose output (no JSON parsing) so partial responses always render.

---

*This document is the single source of truth for MediBot's architecture and design decisions. Keep it updated as the system evolves.*
