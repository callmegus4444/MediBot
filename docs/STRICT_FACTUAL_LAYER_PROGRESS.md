# MediBot Strict Factual Answering Layer - Progress So Far

Date: 2026-05-01

Repository: `d:\ALL PROJECTS\medibuddy`

Status: design artifact added. No runtime backend or frontend code has been changed yet.

## Progress Log

| Date | Change | Files/Areas |
| --- | --- | --- |
| 2026-05-01 | Reviewed the existing project structure and current RAG path. | `server/main.py`, `server/routes/ask_question.py`, `server/modules/llm.py`, `server/modules/query_handlers.py`, `server/modules/load_vectorstore.py`, `client/components/chatUI.py` |
| 2026-05-01 | Verified official integration surfaces for PubMed/NCBI, ClinicalTrials.gov, CDC, FDA, WHO, EMA, JAMA RSS, and UpToDate EHR integration guidance. | Official documentation links listed at the end of this document. |
| 2026-05-01 | Added production design for the second-layer factual verification system. | `docs/STRICT_FACTUAL_LAYER_PROGRESS.md` |
| 2026-05-04 | Implemented Phase 1 (strict response contract + confidence gate), a slice of Phase 3 (PubMed connector), Phase 4 (verification + fail-closed response generator), and Phase 5-light (references panel + strict mode toggle in Streamlit). All inside the existing FastAPI/Streamlit stack — Node gateway scaffold deferred. | `server/schemas/strict.py`, `server/modules/confidence.py`, `server/modules/connectors/pubmed.py`, `server/modules/verification.py`, `server/modules/strict_orchestrator.py`, `server/routes/ask_strict.py`, `server/main.py`, `client/utils/api.py`, `client/components/chatUI.py` |

## Current System Baseline

The current app is a Python RAG application:

- Backend: FastAPI.
- Frontend: Streamlit.
- Vector DB: Pinecone.
- Embeddings: Google Generative AI embeddings, `models/embedding-001`.
- LLM: Groq `llama3-70b-8192` through LangChain.
- Current `/ask/` behavior:
  - embeds the raw question,
  - queries Pinecone with `top_k=3`,
  - wraps matches in LangChain `Document` objects,
  - sends them to a `RetrievalQA` chain,
  - returns `{ response, sources }`.

Current gap: the existing prompt says "use only context", but there is no explicit confidence gate, source verification layer, external trusted-source fallback, source normalization, claim-level validation, queueing, caching, or UI reference panel contract.

## Objective

Add a second layer that answers only when evidence is available and verified. The practical target is not "the LLM never makes a mistake" as an abstract claim. The target is stricter and testable:

- every clinical claim in the final answer must map to retrieved evidence,
- unsupported claims are removed or cause abstention,
- the system uses only allowlisted medical sources,
- the response includes source references and confidence,
- if verification fails, the system returns exactly:

```text
I do not have sufficient verified evidence to answer this question.
```

## Production Architecture

Recommended architecture: add a Node.js/TypeScript verification gateway beside the existing FastAPI RAG service.

This avoids a risky rewrite while still letting BullMQ, Redis, worker pools, and React integration live in the preferred Node stack.

```text
Doctor UI
  |
  v
React App
  |
  v
Node.js Factual Gateway
  |-- Auth, tenant, audit, rate limit
  |-- Query Parser Agent
  |-- Internal RAG Retrieval Agent
  |     |-- Existing FastAPI RAG adapter or direct Pinecone adapter
  |-- Confidence Gate
  |-- External Search Agent
  |     |-- PubMed/NCBI connector
  |     |-- ClinicalTrials.gov connector
  |     |-- CDC connector
  |     |-- WHO connector
  |     |-- FDA/openFDA connector
  |     |-- EMA ePI connector
  |     |-- Licensed/link-only connectors for UpToDate, JAMA, NEJM, Medscape
  |-- Verification Agent
  |-- Response Generator Agent
  |
  v
JSON answer contract
  |-- center panel answer
  |-- left panel references
  |-- confidence score
  |-- audit metadata

Shared Infrastructure
  |-- Redis: BullMQ, rate limits, query/source cache
  |-- Postgres: audit, normalized source metadata, answer runs
  |-- Pinecone: internal PDF vectors and optional external source vectors
  |-- Object storage: permitted raw public snapshots, hashes, crawl logs
  |-- Observability: OpenTelemetry, structured logs, metrics, alerts
```

### Why Node Gateway Instead Of Rewriting FastAPI

- The existing FastAPI server can remain the internal RAG service.
- BullMQ is native to the Node ecosystem.
- TypeScript gives strong schemas for agent outputs and UI contracts.
- Migration can be incremental:
  - Phase 1: Node gateway calls existing `/ask/` or Pinecone directly.
  - Phase 2: add external source retrieval.
  - Phase 3: move response generation and UI to strict contract.
  - Phase 4: deprecate old direct `/ask/` for doctor-facing strict mode.

## Agent Flow Diagram

```text
1. Doctor asks question
   |
   v
2. Query Parser Agent
   - extracts medical terms, symptoms, diseases, drug names, population, intent
   - produces structured query JSON
   |
   v
3. RAG Retrieval Agent
   - queries existing Pinecone internal PDF chunks
   - computes internal confidence from similarity, chunk count, source quality
   |
   +--> If internal confidence >= threshold:
   |      |
   |      v
   |   Verification Agent
   |      - checks whether answerable claims are supported by retrieved chunks
   |      |
   |      v
   |   Response Generator Agent
   |      - answer with internal references
   |
   +--> If internal confidence < threshold:
          |
          v
       External Search Agent
          - searches only allowlisted trusted sources
          - normalizes title, source, URL, abstract/summary, key findings
          |
          v
       Verification Agent
          - ranks evidence
          - rejects unsupported or conflicting claims
          - computes confidence
          |
          v
       Response Generator Agent
          - answer with references or abstain
```

## Strict Response Contract

All strict-mode responses should use one contract, even when no answer is available.

```json
{
  "requestId": "uuid",
  "mode": "strict",
  "answer": "Direct clinical answer or abstention sentence.",
  "confidenceScore": 82,
  "status": "answered",
  "references": [
    {
      "id": "src_001",
      "title": "Article or official page title",
      "source": "PubMed",
      "sourceType": "peer_reviewed_journal",
      "url": "https://pubmed.ncbi.nlm.nih.gov/...",
      "confidenceScore": 91,
      "publishedAt": "2025-10-12",
      "retrievedAt": "2026-05-01T08:30:00Z",
      "keyFindings": [
        "Short evidence-grounded finding."
      ],
      "usedInAnswer": true
    }
  ],
  "verification": {
    "evidenceStatus": "sufficient",
    "unsupportedClaimsRemoved": 0,
    "conflictsDetected": false,
    "internalRagConfidence": 0.84,
    "externalEvidenceConfidence": 0.78
  },
  "ui": {
    "leftPanelTitle": "References",
    "emptyReferencesMessage": "No verified sources found"
  }
}
```

If no verified sources are found:

```json
{
  "requestId": "uuid",
  "mode": "strict",
  "answer": "I do not have sufficient verified evidence to answer this question.",
  "confidenceScore": 0,
  "status": "insufficient_evidence",
  "references": [],
  "verification": {
    "evidenceStatus": "insufficient",
    "unsupportedClaimsRemoved": 0,
    "conflictsDetected": false,
    "internalRagConfidence": 0,
    "externalEvidenceConfidence": 0
  },
  "ui": {
    "leftPanelTitle": "References",
    "emptyReferencesMessage": "No verified sources found"
  }
}
```

## Confidence System

Confidence is not the model's self-rated certainty. It is computed from evidence.

Suggested scoring:

```text
confidence =
  source_count_score
+ source_credibility_score
+ evidence_consistency_score
+ retrieval_relevance_score
+ freshness_score
- conflict_penalty
- unsupported_claim_penalty
- high_risk_claim_penalty
```

Recommended thresholds:

| Range | Behavior |
| --- | --- |
| 85-100 | Answer directly with references. |
| 70-84 | Answer only claims strongly supported by evidence. Include concise uncertainty where clinically necessary. |
| 50-69 | Prefer partial answer only if it is safely bounded. Otherwise abstain. |
| 0-49 | Abstain with the required sentence. |

Internal RAG confidence should use:

- top Pinecone similarity score,
- score gap between top match and lower matches,
- number of relevant chunks above threshold,
- whether chunks come from credible ingested documents,
- whether retrieved chunks contain direct answer-bearing evidence,
- freshness/edition metadata if available.

Important: calibrate thresholds using a test set of real medical questions. Start conservative.

## Source Ranking

| Tier | Source Type | Examples | Notes |
| --- | --- | --- | --- |
| A | Peer-reviewed indexed literature | PubMed abstracts, PMC open access records, JAMA/NEJM records via PubMed when indexed | Do not scrape paywalled full text. |
| A | Official regulator/public health guidance | CDC, WHO, FDA/openFDA, EMA | Strong for public health, safety, labeling, recalls, official guidance. |
| B | Clinical trial registry | ClinicalTrials.gov | Good for study existence/status; not the same as proven clinical efficacy. |
| B | Licensed clinical decision support | UpToDate via licensed integration only | Use only through permitted contractual/API/EHR integration. |
| C | Medical platforms | Medscape, only if licensed or permitted | Useful supporting source, not primary evidence for high-risk claims. |

## Agent Prompts

All agent calls should use `temperature: 0`, JSON schema validation, retry on invalid JSON, and hard fail after bounded retries. Do not let agents access arbitrary browsing tools. They consume only inputs supplied by deterministic connectors.

### 1. Query Parser Agent Prompt

```text
SYSTEM:
You are the Query Parser Agent for a strict medical evidence system used by doctors.

Your job is to convert the doctor's question into structured medical search intent.

Rules:
- Do not answer the question.
- Do not infer facts not present in the question.
- Do not diagnose.
- Extract only what is explicitly stated or clinically necessary for search formulation.
- If a field is unknown, use null or an empty array.
- Identify red flags or emergency intent if explicitly present.
- Return valid JSON only.

USER:
Doctor question:
{{question}}

Return this JSON shape:
{
  "normalizedQuestion": "string",
  "intent": "diagnosis|treatment|drug_safety|guideline|prognosis|trial_lookup|epidemiology|mechanism|other",
  "population": {
    "age": "string|null",
    "sex": "string|null",
    "pregnancyStatus": "string|null",
    "comorbidities": ["string"],
    "setting": "outpatient|inpatient|icu|emergency|unknown"
  },
  "clinicalEntities": {
    "symptoms": ["string"],
    "diseases": ["string"],
    "drugs": ["string"],
    "procedures": ["string"],
    "labs": ["string"],
    "devices": ["string"]
  },
  "searchQueries": {
    "internalRagQuery": "string",
    "pubMedQuery": "string",
    "governmentQuery": "string",
    "clinicalTrialsQuery": "string"
  },
  "evidenceRequirements": {
    "preferredStudyTypes": ["systematic_review", "randomized_trial", "guideline", "labeling", "registry"],
    "needsRecentEvidence": true,
    "highRiskClinicalAnswer": true
  },
  "safetyFlags": {
    "emergency": false,
    "patientSpecificMedicalAdvice": false,
    "requiresHumanClinicianJudgment": true
  }
}
```

### 2. RAG Retrieval Agent Prompt

Use deterministic retrieval first. If an LLM is used to assess retrieved chunks, use this prompt.

```text
SYSTEM:
You are the Internal RAG Retrieval Agent.

Your job is to evaluate whether retrieved internal document chunks contain enough evidence to answer the doctor's question.

Rules:
- Do not answer beyond the retrieved chunks.
- Do not use outside knowledge.
- Identify only direct supporting evidence.
- If evidence is indirect, incomplete, stale, or unrelated, mark it as insufficient.
- Return valid JSON only.

USER:
Structured query:
{{structuredQueryJson}}

Retrieved chunks:
{{retrievedChunksJson}}

Return:
{
  "internalRagConfidence": 0.0,
  "answerableFromInternalDocs": false,
  "supportingChunks": [
    {
      "chunkId": "string",
      "documentTitle": "string|null",
      "page": "string|null",
      "relevanceScore": 0.0,
      "keyFinding": "string",
      "directlySupportsAnswer": true
    }
  ],
  "missingEvidence": ["string"],
  "reason": "string"
}
```

### 3. External Search Agent Prompt

The actual search must be done by source connectors, not by the LLM. This agent plans and normalizes connector results.

```text
SYSTEM:
You are the External Search Agent for a strict medical evidence system.

Allowed sources only:
- PubMed/NCBI
- JAMA Network metadata/RSS or licensed access only
- NEJM metadata/linking or PubMed-indexed records only
- UpToDate licensed integration only
- CDC
- WHO
- Medscape only if licensed or explicitly permitted
- ClinicalTrials.gov
- FDA/openFDA
- EMA

Rules:
- Do not request arbitrary web search.
- Do not scrape paywalled or login-protected content.
- Do not use source text if rights do not permit it.
- Prefer APIs over scraping.
- Prefer peer-reviewed literature and official government/regulator sources.
- Return valid JSON only.

USER:
Structured query:
{{structuredQueryJson}}

Connector results:
{{connectorResultsJson}}

Return:
{
  "normalizedSources": [
    {
      "title": "string",
      "source": "PubMed|CDC|WHO|FDA|EMA|ClinicalTrials.gov|JAMA|NEJM|UpToDate|Medscape",
      "sourceType": "peer_reviewed_journal|government|regulator|clinical_trial_registry|clinical_platform|licensed_cds",
      "url": "string",
      "publishedAt": "string|null",
      "retrievedAt": "string",
      "abstractOrSummary": "string|null",
      "keyFindings": ["string"],
      "credibilityTier": "A|B|C",
      "rightsStatus": "api_public|open_public|metadata_only|licensed_only|unknown",
      "usableForAnswer": true
    }
  ],
  "sourcesRejected": [
    {
      "url": "string|null",
      "reason": "not_allowlisted|paywalled|irrelevant|insufficient_metadata|license_restricted|fetch_failed"
    }
  ]
}
```

### 4. Verification Agent Prompt

```text
SYSTEM:
You are the Verification Agent for a strict clinical answering system.

Your job is to decide whether the evidence is sufficient to support a factual answer.

Rules:
- Do not generate the final answer.
- Verify each proposed claim against evidence.
- A claim is supported only if at least one source directly supports it.
- Prefer two independent sources for treatment, diagnosis, safety, or guideline claims.
- If sources conflict, mark conflict and lower confidence.
- If evidence is missing, stale, indirect, or not clinically applicable, mark insufficient.
- Do not fill gaps with medical knowledge.
- Return valid JSON only.

USER:
Doctor question:
{{question}}

Structured query:
{{structuredQueryJson}}

Internal evidence:
{{internalEvidenceJson}}

External evidence:
{{externalEvidenceJson}}

Return:
{
  "evidenceStatus": "sufficient|partial|insufficient|conflicting",
  "verifiedClaims": [
    {
      "claim": "string",
      "supportingSourceIds": ["string"],
      "supportStrength": "strong|moderate|weak",
      "clinicalApplicability": "direct|indirect|unclear"
    }
  ],
  "rejectedClaims": [
    {
      "claim": "string",
      "reason": "unsupported|conflicting|not_applicable|source_not_trusted|too_old"
    }
  ],
  "confidenceScore": 0,
  "conflicts": [
    {
      "topic": "string",
      "sourceIds": ["string"],
      "description": "string"
    }
  ],
  "mustAbstain": true,
  "abstentionReason": "string|null"
}
```

### 5. Response Generator Agent Prompt

```text
SYSTEM:
You are the Response Generator Agent for MediBot strict mode.

Your audience is doctors.

Rules:
- Use only verified claims supplied by the Verification Agent.
- Do not add medical facts from memory.
- Do not guess.
- Do not provide unsupported diagnosis, dosing, treatment, or safety claims.
- Do not write storytelling or filler.
- Use concise clinical language.
- If mustAbstain is true or evidenceStatus is insufficient/conflicting, the answer must be exactly:
  "I do not have sufficient verified evidence to answer this question."
- Return valid JSON only.

USER:
Doctor question:
{{question}}

Verification result:
{{verificationJson}}

References:
{{referencesJson}}

Return:
{
  "answer": "string",
  "confidenceScore": 0,
  "status": "answered|partial|insufficient_evidence|conflicting_evidence",
  "references": [
    {
      "title": "string",
      "source": "string",
      "url": "string",
      "confidenceScore": 0,
      "keyFindings": ["string"]
    }
  ],
  "clinicalCaveats": ["string"]
}
```

## Backend Architecture - Node.js Preferred

Recommended stack:

- Runtime: Node.js 20+ with TypeScript.
- HTTP API: Fastify or NestJS.
- Queue: BullMQ.
- Cache/queue backend: Redis.
- Relational DB: Postgres.
- ORM/query builder: Prisma, Drizzle, or Kysely.
- Vector DB: keep existing Pinecone.
- Validation: Zod for request, response, and agent schema validation.
- LLM adapter: provider-neutral wrapper so Groq/OpenAI/other models can be swapped.
- Observability: OpenTelemetry, pino structured logs, Prometheus metrics.

Suggested module layout:

```text
server-node/
  src/
    app.ts
    config/
    api/
      chat.routes.ts
      health.routes.ts
    orchestration/
      strict-answer.orchestrator.ts
      confidence.service.ts
    agents/
      query-parser.agent.ts
      rag-retrieval.agent.ts
      external-search.agent.ts
      verification.agent.ts
      response-generator.agent.ts
      prompts/
    retrieval/
      pinecone.adapter.ts
      existing-fastapi-rag.adapter.ts
    connectors/
      pubmed.connector.ts
      clinicaltrials.connector.ts
      cdc.connector.ts
      who.connector.ts
      fda.connector.ts
      ema.connector.ts
      journal-rss.connector.ts
      licensed-source.connector.ts
    queues/
      bullmq.ts
      processors/
    cache/
      redis-cache.ts
    db/
      schema.ts
      repositories/
    logging/
    security/
    types/
```

Main endpoints:

```text
POST /v1/chat/strict
GET  /v1/chat/strict/:requestId
GET  /v1/chat/strict/:requestId/events
GET  /v1/references/:sourceId
GET  /health
```

Synchronous behavior:

- Try cache.
- Try internal RAG.
- If internal evidence is sufficient, return answer immediately.
- If external search is needed and expected to exceed a short timeout, return `202 Accepted` plus `requestId`, then stream progress through SSE.

## BullMQ + Redis Queue Design

Queues:

| Queue | Purpose | Retry |
| --- | --- | --- |
| `strict-chat` | Orchestrates a full strict answer run. | 1 retry, no duplicate active job per `requestId`. |
| `rag-retrieval` | Queries Pinecone or existing FastAPI RAG service. | 2 retries with short backoff. |
| `external-search` | Fans out to trusted source connectors. | 2 retries, source-aware backoff. |
| `source-fetch` | Fetches allowed API records, RSS metadata, or permitted pages. | 3 retries with exponential backoff. |
| `source-normalize` | Converts connector output to normalized source schema. | 1 retry. |
| `verification` | Runs claim/evidence verification. | 1 retry. |
| `answer-generation` | Produces final strict JSON response. | 1 retry. |
| `audit-log` | Writes audit events and analytics. | 5 retries, durable. |

Recommended BullMQ settings:

```ts
{
  attempts: 3,
  backoff: { type: "exponential", delay: 1000 },
  removeOnComplete: { age: 86400, count: 10000 },
  removeOnFail: { age: 604800 },
  timeout: 30000
}
```

Rate limit examples:

```text
PubMed/NCBI without API key: <= 3 requests/second.
PubMed/NCBI with API key: <= 10 requests/second.
ClinicalTrials.gov: use conservative global throttling and backoff.
CDC/WHO/FDA/EMA: per-connector throttles with jitter and circuit breakers.
JAMA/NEJM/UpToDate/Medscape: no scraping unless licensed/contractually permitted.
```

Coordination:

- Start with BullMQ + Redis locks and idempotency keys.
- Use Kubernetes leader election or Redis Redlock if a single scheduled worker must run.
- Add etcd/ZooKeeper only if you introduce multi-region coordination, complex leader election, or many independently scheduled crawlers. It is not needed for the first production version.

## Caching Design

Redis keys:

```text
query:v1:{tenantId}:{normalizedQuestionHash}
source:v1:{source}:{sourceRecordId}
pubmed:v1:{pmid}
ctgov:v1:{nctId}
rate:{source}:{window}
job:{requestId}:status
```

Suggested TTLs:

| Cache | TTL | Notes |
| --- | --- | --- |
| Exact answered query | 6-24 hours | Shorter for treatment/safety questions. |
| PubMed metadata/abstract | 7 days | PubMed records are stable but can be updated. |
| CDC/WHO guidance pages | 24 hours to 7 days | Use shorter TTL during active outbreaks. |
| openFDA labeling/enforcement | 12-24 hours | Drug safety freshness matters. |
| ClinicalTrials.gov study | 12-24 hours | Recruitment status changes. |
| Failed source fetch | 5-30 minutes | Prevent retry storms. |

Cache invalidation:

- include source `retrievedAt` in every answer,
- expire high-risk clinical answers quickly,
- invalidate when connector version or ranking logic changes,
- do not cache PHI-bearing raw user input; hash normalized query text.

## Scraping Strategy - Safe And Scalable

Policy:

1. API first.
2. RSS or metadata second.
3. Public-page scraping only when terms permit it.
4. Never scrape login-protected, paywalled, or explicitly disallowed pages.
5. Never bypass access controls.
6. Store only what rights allow. Prefer metadata, short summaries, URL, checksum, retrieval timestamp, and connector logs.

Allowed implementation details:

- Maintain a strict source allowlist.
- Use `robots.txt` and terms-of-use checks before adding any scraper.
- Identify with a clear user agent and contact email.
- Use per-domain concurrency limits.
- Use exponential backoff and circuit breakers.
- Deduplicate by DOI, PMID, NCT ID, canonical URL, or content hash.
- Extract title, abstract/summary, source, URL, date, and key findings.
- Do not train models on scraped copyrighted content.

Recommended source strategy:

| Source | Strategy |
| --- | --- |
| PubMed | Use NCBI E-utilities: `esearch`, `esummary`, `efetch`. |
| PubMed Central | Use NCBI/PMC APIs for open access records when needed. |
| JAMA Network | Prefer PubMed-indexed records and official RSS metadata. Licensed full text only if contracted. |
| NEJM | Do not scrape. Use PubMed records, links, and licensed access if available. |
| UpToDate | Do not scrape. Use licensed EHR/HL7 Infobutton or vendor-approved integration. |
| CDC | Use CDC Content Services API, CDC Open Data, and public official pages where permitted. |
| WHO | Use GHO OData/Athena APIs and official publications/pages. |
| FDA | Use openFDA for labeling, adverse events, enforcement, device, and safety datasets. |
| EMA | Use ePI/FHIR API for EU medicine product information where applicable. |
| Medscape | Use only if licensed/approved. Otherwise skip. |
| ClinicalTrials.gov | Use v2 REST API. Treat registry data as trial status evidence, not efficacy proof. |

## API Suggestions

Primary integrations:

- PubMed/NCBI E-utilities
  - Best for biomedical literature search and abstracts.
  - Use NCBI API key in production for higher rate limits.
  - Pipeline: `esearch` -> `esummary`/`efetch` -> normalize PMIDs, titles, abstracts, publication dates, journal, DOI.

- ClinicalTrials.gov API v2
  - Best for trial lookup, recruitment status, conditions, interventions, locations, outcomes.
  - Pipeline: query by condition/intervention -> fetch study records by NCT ID -> summarize status and outcomes carefully.

- openFDA
  - Best for drug labels, adverse events, enforcement/recalls, device datasets.
  - Do not treat adverse event reports as causality without context.

- CDC Content Services and data APIs
  - Best for public health guidance, syndication content, datasets, CDC topics.

- WHO Global Health Observatory APIs
  - Best for global health indicators, country/regional data, official WHO datasets.

- EMA ePI API
  - Best for EU product information documents such as summary of product characteristics and package leaflets.

Conditional/licensed integrations:

- UpToDate
  - Use only vendor-approved EHR/HL7 Infobutton or enterprise integration.
  - Skip if no licensed access.

- JAMA Network
  - Use RSS for metadata discovery and PubMed for indexed article abstracts.
  - Full text only through licensed/contracted access.

- NEJM
  - Do not scrape. Use PubMed records and direct links unless licensed access is available.

- Medscape
  - No public API assumed. Use only with permission/licensing.

## Database Design

Postgres tables:

```text
tenants
  id, name, plan, created_at

users
  id, tenant_id, email, role, created_at

chat_sessions
  id, tenant_id, user_id, created_at, updated_at

chat_messages
  id, session_id, role, content_hash, redacted_content, created_at

strict_answer_runs
  id, request_id, tenant_id, user_id, session_id,
  normalized_question_hash, status, confidence_score,
  internal_rag_confidence, external_evidence_confidence,
  latency_ms, created_at, completed_at

structured_queries
  id, run_id, intent, entities_json, search_queries_json, safety_flags_json

source_documents
  id, source, source_type, title, url, canonical_id,
  doi, pmid, nct_id, published_at, retrieved_at,
  rights_status, credibility_tier, content_hash

source_findings
  id, source_document_id, run_id, key_finding, relevance_score, used_in_answer

verification_claims
  id, run_id, claim, status, support_strength,
  supporting_source_ids_json, rejection_reason

api_usage_events
  id, tenant_id, source, endpoint, status_code, latency_ms, created_at

source_failures
  id, run_id, source, failure_type, message, retry_count, created_at

audit_events
  id, tenant_id, user_id, action, request_id, metadata_json, created_at
```

Pinecone metadata for internal PDF chunks:

```json
{
  "tenantId": "tenant_123",
  "documentId": "doc_123",
  "documentTitle": "DIABETES.pdf",
  "sourceType": "internal_pdf",
  "page": 12,
  "chunkIndex": 44,
  "text": "chunk text",
  "uploadedAt": "2026-05-01T08:00:00Z",
  "contentHash": "sha256"
}
```

Optional Pinecone metadata for external source chunks:

```json
{
  "source": "PubMed",
  "sourceType": "peer_reviewed_journal",
  "pmid": "12345678",
  "doi": "10.xxxx/yyyy",
  "title": "Article title",
  "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
  "publishedAt": "2025-09-01",
  "retrievedAt": "2026-05-01T08:30:00Z",
  "credibilityTier": "A",
  "rightsStatus": "api_public"
}
```

## Frontend Integration Plan - React

Target layout:

```text
+-------------------------------------------------------------+
| Header: MediBot Strict Mode                                 |
+----------------------+--------------------------------------+
| Left Reference Panel | Main Answer Panel                    |
|                      |                                      |
| Source cards         | Doctor question                      |
| - title              | Answer                               |
| - source badge       | Confidence badge                     |
| - link               | Clinical caveats                     |
| - confidence         |                                      |
|                      |                                      |
| Empty state:         | Loading states:                      |
| No verified sources  | Parsing, internal RAG, external      |
| found                | search, verification, answer         |
+----------------------+--------------------------------------+
```

React components:

```text
StrictChatPage
ReferencePanel
ReferenceCard
AnswerPanel
ConfidenceBadge
VerificationStatus
QuestionInput
SourceDrawer
```

Recommended TypeScript types:

```ts
type StrictAnswerStatus =
  | "answered"
  | "partial"
  | "insufficient_evidence"
  | "conflicting_evidence"
  | "processing";

type Reference = {
  id?: string;
  title: string;
  source: string;
  sourceType?: string;
  url: string;
  confidenceScore: number;
  keyFindings: string[];
  publishedAt?: string | null;
  retrievedAt?: string | null;
};

type StrictAnswerResponse = {
  requestId: string;
  mode: "strict";
  answer: string;
  confidenceScore: number;
  status: StrictAnswerStatus;
  references: Reference[];
};
```

UI rules:

- Left panel always exists in strict mode.
- If `references.length === 0`, show `No verified sources found`.
- Source links open in a new tab with clear source label.
- Confidence must be visible near the answer.
- Do not hide abstention behind generic error UI.
- Stream progress states during external search.

Current Streamlit bridge:

- Short term: use `st.columns([1, 2])` to show references left and answer center.
- Long term: replace Streamlit with React for richer async/SSE behavior.

## Failure Handling Strategy

| Failure | Behavior |
| --- | --- |
| Internal RAG low confidence | Trigger external trusted-source search. |
| No internal or external evidence | Return required abstention sentence and empty references. |
| Source API timeout | Retry with backoff; if still failing, mark source unavailable and continue with other sources. |
| Rate limit | Backoff, queue delay, circuit breaker per source. |
| Paywall/login detected | Reject source content; keep link metadata only if allowed. |
| Conflicting evidence | Either abstain or answer only non-conflicting verified claims. Lower confidence. |
| LLM invalid JSON | Retry with schema reminder; after limit, fail closed. |
| LLM adds unsupported claim | Verification rejects claim; response generator cannot use it. |
| Queue worker crash | BullMQ retry plus idempotent request IDs. |
| Redis unavailable | Return degraded service response; do not run unthrottled source fetching. |
| Postgres unavailable | Stop strict mode or buffer audit events only if compliance allows. |
| PHI detected | Redact logs, avoid cache of raw query, enforce tenant isolation. |

Fail-closed default:

```text
I do not have sufficient verified evidence to answer this question.
```

## Logging And Audit

Track:

- request ID,
- tenant/user/session,
- normalized query hash,
- detected entities,
- internal retrieval scores,
- external sources queried,
- source URLs and IDs used,
- rejected sources and reasons,
- verification confidence,
- final answer status,
- latency by stage,
- failures and retries.

Do not log raw PHI unless the product has the required compliance controls and a clear retention policy. Prefer redacted content plus hashes.

## Security And Compliance Notes

Because this is intended for real doctors:

- Add authentication and role-based access control.
- Add tenant isolation for vectors, database records, queues, and cache keys.
- Treat chat content as potentially sensitive health information.
- Encrypt data in transit and at rest.
- Maintain audit logs.
- Define retention/deletion policies.
- Add source licensing review before enabling any scraper.
- Add a clinical safety review process before production use.

## Implementation Phases

Phase 1 - Contract and confidence gate:

- Add strict response schema.
- Add internal RAG scoring.
- Return source metadata with page/document info.
- Add abstention on low internal confidence.

Phase 2 - Node gateway and queues:

- Add TypeScript Node service.
- Add Redis and BullMQ.
- Add `POST /v1/chat/strict`.
- Call existing FastAPI/Pinecone for internal retrieval.

Phase 3 - External source connectors:

- PubMed first.
- ClinicalTrials.gov second.
- FDA/CDC/WHO/EMA next.
- Add licensed-only placeholders for UpToDate/JAMA/NEJM/Medscape.

Phase 4 - Verification:

- Add claim extraction and source mapping.
- Add confidence formula.
- Add conflict detection.
- Add fail-closed response generator.

Phase 5 - React UI:

- Build left reference panel.
- Build main answer panel.
- Add confidence and progress states.
- Support SSE for async external search.

Phase 6 - Evaluation and safety:

- Build medical QA regression set.
- Add unsupported-claim tests.
- Add source outage tests.
- Add rate-limit tests.
- Review with clinicians before production rollout.

## Acceptance Criteria

- The system never returns a clinical answer with zero verified references.
- Empty evidence returns exactly the required abstention sentence.
- Every answer includes `confidenceScore`.
- Every reference includes title, source, link, and confidence score.
- External search uses only allowlisted connectors.
- Scrapers are disabled by default unless legal/terms review approves them.
- Queue jobs are idempotent by `requestId`.
- Logs contain enough data to audit which evidence produced each answer.

## Implementation Notes (2026-05-04)

What was actually built in this session, with the rationale for the deviations:

- **Stayed in the existing FastAPI service.** The original plan recommends a Node/TypeScript gateway alongside the FastAPI RAG service. Building that gateway in one session would deliver scaffolding without working behavior. Instead, the strict layer was implemented in the existing Python service so the doctor-facing flow runs end-to-end today. The Node gateway remains a valid Phase 2 target; the modules added here (`schemas/strict.py`, `modules/confidence.py`, `modules/verification.py`, `modules/strict_orchestrator.py`) translate cleanly into the proposed TypeScript module layout.
- **New endpoint, not a replacement.** `/ask/strict/` is added beside the existing `/ask/`. Default chat keeps working unchanged. The Streamlit UI now exposes a "Strict evidence mode" toggle so doctor-mode and casual mode coexist.
- **Strict response contract.** `server/schemas/strict.py` implements the `StrictAnswerResponse`, `Reference`, `VerificationSummary`, and `UiHints` models from the design doc, plus the canonical `ABSTENTION_SENTENCE` and an `abstention_response()` helper so every fail-closed path returns the same shape.
- **Confidence gate.** `server/modules/confidence.py` scores Pinecone retrievals from top match + relevant-chunk count, exposes the 85/70/50 threshold bands, and computes a combined internal+external confidence with penalties for conflicts and unsupported claims.
- **Internal retrieval.** The orchestrator queries Pinecone with `top_k=5` (configurable via `STRICT_TOP_K`). Internal chunks are turned into both verification payloads and `Reference` objects so internal-only answers still populate the left panel.
- **External fallback (PubMed).** `server/modules/connectors/pubmed.py` calls NCBI E-utilities `esearch` -> `esummary`, supports `NCBI_API_KEY` and `NCBI_CONTACT_EMAIL` env vars, throttles to ~3 req/s when no key is present, and returns normalized `Reference` objects. Smoke-tested against the live API: returns peer-reviewed records for clinical queries. ClinicalTrials.gov, openFDA, CDC/WHO/EMA connectors follow the same shape and can drop into the same orchestrator.
- **Verification + response generation.** `server/modules/verification.py` uses the existing Groq Llama model with `temperature=0` and JSON-only system prompts. The verification agent decides `mustAbstain`, `evidenceStatus`, conflicts, and rejected claims; the response generator only writes an answer when verification cleared it. Any non-JSON or schema-mismatch response causes a fail-closed abstention.
- **Fail-closed everywhere.** The orchestrator returns the canonical abstention sentence (with `status: insufficient_evidence` or `conflicting_evidence`) for: empty query, retrieval errors, no internal/external evidence, verification mustAbstain, response generator failure, no cited references, or final confidence below the abstain band.
- **UI references panel.** `client/components/chatUI.py` switches to a `st.columns([1, 2])` layout: left column is the references panel with source cards (title, source, link, per-source confidence bar), right column is the chat. Strict mode adds status + confidence badges and surfaces `unsupportedClaimsRemoved` and `conflictsDetected` from the verification summary.

### Environment variables introduced

| Var | Purpose | Default |
| --- | --- | --- |
| `NCBI_API_KEY` | Higher PubMed rate limit (10 req/s) | unset (3 req/s) |
| `NCBI_CONTACT_EMAIL` | NCBI compliance contact in E-utilities params | `noreply@example.com` |
| `STRICT_TOP_K` | Pinecone top_k for internal retrieval in strict mode | `5` |
| `STRICT_EXTERNAL_RETMAX` | Max PubMed records per query | `5` |
| `GROQ_MODEL` | Override the Groq model name for verification/response | `llama3-70b-8192` |

### Smoke tests run

- `schemas.strict.abstention_response("test").model_dump()` returns the canonical fail-closed shape with the exact abstention sentence.
- `modules.confidence.score_pinecone_matches([...])` produces sensible bands across high, low, and empty cases.
- `modules.connectors.pubmed.search("metformin type 2 diabetes", retmax=2)` returns two normalized PubMed references against the live NCBI API.
- AST-parsed all new files and the modified `main.py` to confirm no syntax errors.

### Not yet implemented

- Node/TypeScript gateway (`server-node/`), BullMQ + Redis, Postgres audit tables.
- ClinicalTrials.gov, openFDA, CDC, WHO, EMA, JAMA RSS, UpToDate, NEJM, Medscape connectors.
- React frontend (still on Streamlit).
- SSE / `202 Accepted` async path for long external searches.
- Auth, tenant isolation, audit logging, rate limiting.
- Calibrated thresholds against a real medical QA test set.

## Official Documentation Links Used

- NCBI APIs and PubMed E-utilities: https://www.ncbi.nlm.nih.gov/home/develop/api/
- NCBI API key/rate guidance: https://support.nlm.nih.gov/knowledgebase/article/KA-05317/en-us
- ClinicalTrials.gov API v2 announcement: https://www.nlm.nih.gov/pubs/techbull/ma24/ma24_clinicaltrials_api.html
- CDC Content Services API reference: https://tools.cdc.gov/api/docs/info.aspx
- FDA openFDA: https://www.fda.gov/science-research/health-informatics-fda/openfda
- WHO Global Health Observatory: https://www.who.int/data/gho/
- EMA ePI API details: https://epi.developer.ema.europa.eu/api-details
- UpToDate EHR integration: https://www.wolterskluwer.com/en/solutions/uptodate/about/ehr-integration
- JAMA RSS feeds: https://jamanetwork.com/journals/jama/pages/rss
- NEJM Group terms and scraping restrictions: https://www.nejmgroup.org/legal/terms-of-use.htm
