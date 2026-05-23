# 🩺 MediBot — AI Medical Assistant Chatbot

> Evidence-grounded clinical Q&A for doctors. Searches PubMed, trusted medical websites, the FDA drug-label database, ClinicalTrials.gov, and your own uploaded PDFs — in parallel — then streams a cited answer back.

---

## ✨ What it does

Ask any clinical question. MediBot:

1. Embeds the question with **Gemini Embedding 001** and searches your uploaded PDFs via **Pinecone** (scoped to the active library namespace).
2. Fans out **in parallel** to:
   - **PubMed** (peer-reviewed literature, via NCBI E-utilities)
   - **Tavily Web Search** restricted to a whitelist of trusted medical domains (WHO, CDC, FDA, NIH, NEJM, BMJ, Mayo Clinic, Cochrane, NICE, …)
   - **OpenFDA** (live FDA drug labels — indications, dosage, contraindications, warnings, interactions)
   - **ClinicalTrials.gov v2** (ongoing and completed trials)
3. Feeds all evidence + the last 6 turns of conversation history to a **Groq Llama 3.3 70B** synthesizer.
4. **Streams** the answer back token-by-token over SSE, with inline `[source_id]` citations.
5. Renders distinct source badges (PubMed · Web · Internal · FDA · Trial), credibility tiers, and a final confidence score.
6. Auto-saves the chat to disk so the doctor can revisit prior sessions.

---

## 🧱 Architecture (high level)

```
┌────────────────────────── STREAMLIT CLIENT ──────────────────────────┐
│  Library selector   New chat / load / delete   🌐 Web toggle         │
│  Streaming chat (SSE)        Reference panel with typed badges        │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ HTTP + SSE
                               ▼
┌──────────────────────────── FASTAPI SERVER ──────────────────────────┐
│   /upload_pdfs/   /libraries/   /ask/strict/   /ask/strict/stream/   │
│   /chat/save/     /chat/list/   /chat/{id}/   (DELETE chat)          │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
   ┌─── Pinecone (per-library namespaces) ────────────────────────────┐
   │                                                                   │
   │   ┌── PubMed ──┐  ┌── Tavily web ──┐  ┌── OpenFDA ──┐  ┌── CT.gov ┐ │
   │   │ E-utils    │  │ whitelist 35   │  │ drug labels │  │ studies   │ │
   │   └────────────┘  └────────────────┘  └─────────────┘  └───────────┘ │
   └────────────────────────── all in parallel ─────────────────────────┘
                               │
                               ▼
              Groq Llama 3.3 70B  →  streamed prose with [source_id]
```

See [`SYSTEM_DESIGN.md`](./SYSTEM_DESIGN.md) for the full design.

---

## 🚀 Quick start

### 1. Server

```bash
cd server
uv venv && .venv/Scripts/activate   # macOS/Linux: source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env                # then fill in keys (see below)
uvicorn main:app --reload --port 8000
```

### 2. Client

```bash
cd client
uv venv && .venv/Scripts/activate
uv pip install -r requirements.txt
streamlit run app.py
```

Open **http://localhost:8501**.

### 3. Environment (`server/.env`)

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | ✅ | Llama 3.3 70B inference |
| `GOOGLE_API_KEY` | ✅ | Gemini embeddings (768-dim) |
| `PINECONE_API_KEY` | ✅ | Vector store |
| `PINECONE_INDEX_NAME` | ✅ | e.g. `medicalindex` |
| `TAVILY_API_KEY` | ✅ for web search | Whitelisted medical web search |
| `GROQ_MODEL` | optional | default `llama-3.3-70b-versatile` |
| `STRICT_TOP_K` | optional | Pinecone top-k (default `5`) |
| `STRICT_EXTERNAL_RETMAX` | optional | PubMed max results (default `5`) |
| `STRICT_WEB_RETMAX` | optional | Web max results (default `5`) |
| `STRICT_FDA_RETMAX` | optional | OpenFDA max results (default `3`) |
| `STRICT_TRIALS_RETMAX` | optional | ClinicalTrials.gov max (default `3`) |
| `NCBI_API_KEY` | optional | Raises PubMed rate limit to 10 rps |
| `NCBI_CONTACT_EMAIL` | optional | Contact for NCBI policy |
| `OPENFDA_API_KEY` | optional | Raises OpenFDA rate limit |
| `CHAT_HISTORY_DIR` | optional | Override server chat-history dir |

---

## 🔌 API

| Method | Path | Description |
|---|---|---|
| `POST` | `/upload_pdfs/` | Multipart upload `files[]` + form `library` (string). Embeds and upserts into the library's Pinecone namespace. |
| `GET`  | `/libraries/` | Lists existing library namespaces. |
| `POST` | `/ask/` | General-mode (legacy) RAG answer. |
| `POST` | `/ask/strict/` | Strict evidence-grounded answer. Form fields: `question`, `library`, `use_web` (bool), `history_json` (JSON list of `{role, content}`). |
| `POST` | `/ask/strict/stream/` | Same inputs as `/ask/strict/`; returns SSE stream with events `meta` → `references` → `delta`* → `done`. |
| `POST` | `/chat/save/` | Body `{session_id, title?, library?, messages: [...]}`. |
| `GET`  | `/chat/list/` | List saved sessions (id, title, library, updatedAt, messageCount). |
| `GET`  | `/chat/{session_id}/` | Load a saved session. |
| `DELETE` | `/chat/{session_id}/` | Delete a saved session. |

Strict response shape (non-stream):

```json
{
  "requestId": "uuid",
  "mode": "strict",
  "answer": "string with inline [source_id] citations",
  "confidenceScore": 0-100,
  "status": "answered | partial | insufficient_evidence | conflicting_evidence",
  "references": [
    {
      "id": "pubmed_12345 | web_0_cdc_gov | openfda_metformin_0 | trial_NCT01234567 | internal_3",
      "title": "...",
      "source": "PubMed | cdc.gov | OpenFDA | ClinicalTrials.gov | Internal Library",
      "sourceType": "peer_reviewed_journal | web | drug_label | clinical_trial | internal_pdf",
      "url": "...",
      "credibilityTier": "A | B | C",
      "publishedAt": "YYYY-MM-DD",
      "keyFindings": ["..."]
    }
  ],
  "verification": {
    "evidenceStatus": "sufficient | partial | insufficient | conflicting",
    "internalRagConfidence": 0.0,
    "externalEvidenceConfidence": 0.0
  }
}
```

---

## 🌐 Trusted-domain whitelist (web search)

WHO, NIH, NCBI/PMC, MedlinePlus, CDC, FDA, EMA, NICE, Cochrane, AHRQ, Mayo Clinic, Cleveland Clinic, Hopkins Medicine, UpToDate, Merck Manuals, AAFP, AMA, ACC, Heart.org, Diabetes.org, Cancer.gov, Cancer.org, RxList, Drugs.com, Medscape, BMJ, NEJM, The Lancet, JAMA Network, Nature, ScienceDirect, Springer, Wiley.

Government and major-journal domains get Tier **A**, the rest Tier **B**.

---

## 📚 Libraries (no auth)

Each "library" is an isolated Pinecone namespace. The sidebar lets the doctor:

- Pick an existing library
- Create a new one (e.g. `cardiology`, `pediatrics`)
- Upload PDFs into it
- Query only that library's documents

No authentication — libraries are shared across anyone with access to the deployment. Add auth in front (e.g. nginx + basic auth) for production.

---

## 💬 Chat history

- Auto-saved server-side as `server/chat_history/<session_id>.json` after every assistant turn.
- Sidebar lists prior sessions; click to load, 🗑 to delete.
- Each conversation stores the last assistant message's references too, so they re-render when you reload an old chat.
- Also downloadable as `.txt` or `.json` from the sidebar.

---

## 🛡 Safety design

- **Fail-closed**: any retrieval/LLM error → abstention sentence, never a guess.
- **No diagnostic claims**; the synthesizer prompt forbids inventing dosages, statistics, or trial outcomes.
- **Source citations mandatory** inline (`[source_id]`). General-knowledge sentences are tagged `[general clinical knowledge]`.
- **Strict mode only** in the new UI (general mode endpoint stays for legacy callers).

---

## 🧩 Tech stack

| Layer | Tech |
|---|---|
| LLM | Groq Llama 3.3 70B Versatile |
| Embeddings | Google Gemini Embedding 001 (768-dim) |
| Vector store | Pinecone Serverless (dotproduct, AWS us-east-1) |
| Web search | Tavily (whitelisted domains) |
| Drug data | OpenFDA `/drug/label.json` |
| Trials | ClinicalTrials.gov API v2 |
| Backend | FastAPI + Uvicorn |
| Streaming | Server-Sent Events |
| Frontend | Streamlit |
| Package mgr | `uv` |

---

## 🗺 Roadmap

Already shipped: web search, OpenFDA, ClinicalTrials.gov, conversation memory, streaming, libraries, persistent history.

Still planned (see `SYSTEM_DESIGN.md` §16):

- UMLS query expansion ("heart attack" → MI / AMI)
- Cross-encoder reranker on top of Pinecone ANN
- Authentication / multi-tenancy
- LangSmith + Prometheus observability

---

## 📜 License

MIT.
