# System Architecture & Technical Design: Hiver Support Agent

## 1. System Overview

The **Hiver Support Agent** is an evaluation-first, modular AI customer support agent tailored for Apple customer service interactions using the Customer Support on Twitter dataset (data/raw/twcs.csv).

The system ingests raw multi-party customer support tweets, reconstructs coherent conversation dialogue threads via deterministic breadth-first search (BFS) starting from customer roots, classifies inquiries into an 8-class empirical taxonomy, retrieves historically resolved precedent conversations using an in-memory Okapi BM25 engine, generates grounded draft responses citing precedent, and applies a multi-trigger escalation routing engine to decide between automated response and human operator escalation.

---

# Implementation Plan: Hiver Support Agent

**Branch**: `001-support-agent` | **Date**: 2026-09-12 | **Spec**: [SPECIFICATION.md](SPECIFICATION.md)
 
**Input**: Feature specification from [SPECIFICATION.md](SPECIFICATION.md) and dataset profile from [DATASET_PROFILE.md](DATASET_PROFILE.md).

---

## Summary

Build an evaluation-first, modular AI customer-support agent focusing on AppleSupport using the Customer Support on Twitter dataset (`data/raw/twcs.csv`). The system reconstructs dialogue threads via deterministic breadth-first search from customer clean roots, classifies inquiries into an 8-class empirical taxonomy, retrieves historically resolved conversations via an in-memory Okapi BM25 engine, generates grounded draft responses citing precedent, and makes calibrated, explainable auto-handle vs. escalation decisions. The implementation prioritizes rigorous evaluation (200 curated evaluation benchmark examples with zero leakage) and single-machine reproducibility under 15 minutes.

---

## Technical Context

**Language/Version**: Python 3.10+ (tested on Python 3.13)  
**Primary Dependencies**: `pydantic` (schemas & validation), `scikit-learn` (baseline models & calibration), `google-generativeai` (Gemini API for draft generation & judging), `click` (CLI interface), `pandas` (tabular processing), `pytest` (test suite)  
**Storage**: File-based local storage (`data/processed/` JSONL/Parquet) and in-memory serialized BM25 index (pure Python, zero external database daemon)  
**Testing**: `pytest` (unit tests, contract schema validation, integration tests)  
**Target Platform**: Single developer workstation (Windows / macOS / Linux)  
**Project Type**: Modular CLI tool and Python library (`src/`)  
**Performance Goals**:
- Full evaluation harness execution: $< 15$ minutes across 200 golden examples
- End-to-end single-ticket inference: $< 500$ ms latency
- In-memory retrieval latency: $< 5$ ms per query across 63,800 historical conversations
**Constraints**:
- Absolute zero conversation-level eval leakage (cryptographically and set-theoretically audited)
- Dataset immutability (`data/raw/twcs.csv` strictly read-only)
- Byte-identical output reproducibility under fixed seeds
**Scale/Scope**:
- 2,811,774 raw tweet records
- 74,571 reconstructed AppleSupport conversations
- 63,842 retrieval corpus conversations (pre-2017-11-01)
- 200 authentic post-split evaluation benchmark examples (rule-assisted curated ground truth)

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Gate Status | Compliance Strategy & Verification |
|---|:---:|---|
| **I. Eval Before Optimization** | **PASS** | Evaluation harness (`src/eval/`) and golden benchmark built first; stub baseline passing before AI tuning. |
| **II. Reproducibility & Determinism** | **PASS** | Ingestion pipeline uses fixed random seeds (`seed=42`), SHA-256 hash logging, and temperature 0.0. |
| **III. Grounded Generation** | **PASS** | `DraftReply` models mandate citation of retrieved `source_conversation_id`s; ungrounded claims prohibited. |
| **IV. Explicit Uncertainty & Escalation** | **PASS** | `EscalationRouter` enforces multi-trigger fallback, confidence thresholds, and fail-safe human escalation default. |
| **V. No Eval Leakage** | **PASS** | Strict temporal split boundary (2017-11-01); automated pre-eval set-intersection assertion halts on overlap. |
| **VI. Modular Architecture** | **PASS** | Decoupled modules (`pipeline`, `intent`, `retrieval`, `generator`, `escalation`, `eval`) with Pydantic contracts. |
| **VII. Observability of Decisions** | **PASS** | Every decision emits a structured JSON decision trace with confidence, triggers, and evidence excerpts. |
| **VIII. Simplicity Over Infrastructure** | **PASS** | Pure Python in-memory BM25 index; zero external vector DB, Docker containers, or distributed queues. |
| **IX. Fast Reproducible Eval** | **PASS** | End-to-end evaluation runs in $< 15$ minutes; includes random & majority-class baselines. |
| **X. Decision Log Discipline** | **PASS** | Comprehensive decision log maintained across all architectural, prompt, and threshold choices. |

---

## Project Structure

### Documentation Structure

```text
docs/
├── SPECIFICATION.md          # Feature requirements & user stories
├── ARCHITECTURE.md           # System architecture & research decisions
├── DATASET_PROFILE.md        # Kaggle TWCS statistical analysis
├── DATA_MODEL.md             # Core schemas & Pydantic models
├── DECISION_LOG.md           # Architectural decision records
├── EVALUATION_REPORT.md      # Evaluation methodology & results
├── QUICKSTART.md             # Setup & validation runbook
└── contracts/
    ├── CLI_CONTRACT.md       # CLI syntax & exit codes
    └── PIPELINE_SCHEMAS.json # JSON Schema validation contracts
```

### Source Code Layout

```text
src/
├── __init__.py
├── cli.py                   # Main CLI entrypoint (ingest, evaluate, handle, audit)
├── config.py                # Configuration loader and threshold dataclasses
├── data/
│   ├── __init__.py
│   ├── ingestion.py         # Streaming CSV reader & BFS conversation reconstruction
│   ├── normalizer.py        # HTML decoding, token normalization, text cleaning
│   ├── splitter.py          # Temporal partitioner & manifest generator
│   └── human_benchmark.py   # Sampling, validation, freezing for human test benchmark
├── intent/
│   ├── __init__.py
│   ├── taxonomy.py          # 8-class taxonomy definitions & keyword patterns
│   └── classifier.py        # Intent classifier (rule/few-shot) with confidence calibration
├── retrieval/
│   ├── __init__.py
│   ├── bm25.py              # Pure Python in-memory Okapi BM25 indexer & ranker
│   └── engine.py            # Evidence retriever with min-score filtering
├── generator/
│   ├── __init__.py
│   ├── prompts.py           # Grounded response prompt templates
│   ├── llm_client.py        # Provider client (Gemini API + Offline Mock fallback)
│   └── reply_agent.py       # Grounded draft reply generator with citation metadata
├── escalation/
│   ├── __init__.py
│   └── router.py            # Multi-trigger rule engine & confidence threshold router
└── eval/
    ├── __init__.py
    ├── harness.py           # Evaluation orchestrator & baseline runner
    ├── metrics.py           # F1, MRR@5, Precision@5, False-Auto-Handle, Kappa
    ├── judge.py             # LLM-as-judge groundedness & alignment scorer
    └── report.py            # Structured JSON & Markdown evaluation report generator

tests/
├── unit/
│   ├── test_ingestion.py        # Conversation BFS reconstruction unit tests
│   ├── test_normalizer.py       # Text cleaning & HTML decoding tests
│   ├── test_intent.py           # Intent classification & calibration tests
│   ├── test_retrieval.py        # In-memory BM25 indexer & ranking tests
│   ├── test_escalation.py       # Multi-trigger escalation logic tests
│   └── test_human_benchmark.py  # Sampling, validation, duplicate & leakage tests
├── contract/
│   ├── test_schemas.py                  # Pydantic & JSON Schema validation tests
│   ├── test_cli.py                      # CLI flag, argument & exit code tests
│   └── test_human_benchmark_contract.py # Human benchmark schema & annotator contract tests
└── integration/
    ├── test_leakage.py                  # Automated split manifest zero-overlap test
    ├── test_pipeline.py                 # End-to-end single-ticket inference pipeline test
    ├── test_eval_harness.py             # Full evaluation harness run on golden test fixtures
    └── test_human_benchmark_workflow.py # Human benchmark lifecycle (sample->validate->freeze)
```

---

## Complexity Tracking

*No violations of Constitution Principles detected.* All design choices favor minimal operational complexity:
- In-memory BM25 selected over standalone vector DBs (Principle VIII).
- Pure Python and standard libraries prioritized over heavy orchestration frameworks.
- File-based Parquet/JSONL artifacts selected over client-server databases.


---

## Technical Research & Design Decisions

# Technical Research: Hiver Support Agent

**Feature Branch**: `001-support-agent`  
**Date**: 2026-09-12  
**Status**: Complete  

This document resolves all technical decisions, architecture strategies, and implementation trade-offs deferred from the feature specification ([SPECIFICATION.md](SPECIFICATION.md)), adhering strictly to the core architectural principles.

---

## 1. Research Area 1: Temporal Train/Eval Split Boundary (Assumption A-04)

### Context & Problem
Constitution Principle V and `spec.md` (FR-D-006, A-04) mandate a zero-leakage conversation-level split. In customer support, real-world deployments use historical resolutions to assist future incoming requests. Thus, a temporal split reflects real operational conditions far better than random shuffling.

### Empirical Evidence from Dataset Profile
- AppleSupport conversation timestamps in `data/raw/twcs.csv` span from **November 2012 to November 30, 2017**.
- Total clean AppleSupport conversations: **74,571**.
- Timestamp distribution:
  - Pre-2017-11-01: ~63,800 conversations (~85.5% of corpus).
  - 2017-11-01 to 2017-11-30: ~10,700 conversations (~14.5% of corpus).

### Decision
- **Retrieval / Development Corpus Partition**: All conversations whose root tweet timestamp is **strictly before 2017-11-01 00:00:00 UTC** (~63,800 conversations).
- **Evaluation Candidate Pool**: All conversations whose root tweet timestamp is **on or after 2017-11-01 00:00:00 UTC** (~10,700 conversations).
- **Curated Evaluation Benchmark**: A carefully sampled, rule-assisted curated benchmark of **200 authentic evaluation examples** drawn strictly from the Evaluation Candidate Pool, stratified across all 8 intent categories and escalation conditions with documented rule-based ground truth criteria.

### Rationale
1. **Zero Contamination**: The temporal boundary is strictly monotonic; no conversation in the evaluation benchmark can have occurred prior to the retrieval corpus boundary.
2. **Operational Realism**: Simulates production where the agent uses precedent up to October 31, 2017 to handle tickets arriving in November 2017.
3. **Volume Adequacy**: Retrieval partition provides dense lexical and semantic coverage for retrieval indexing, while post-split conversations provide a rich pool for selecting 200 high-signal evaluation examples.

### Alternatives Considered
- *Random Conversation-Level Split (80/20)*: Rejected because random splitting allows later conversations into the retrieval corpus to answer earlier ones, causing temporal data leakage (retrieving future knowledge).
- *Stratified Hash Splitting*: Rejected for the same temporal leakage reason.

---

## 2. Research Area 2: Intent Taxonomy Definition (Assumption A-06)

### Context & Problem
FR-I-001 and FR-I-002 require a small, explainable, mutually exclusive, and exhaustive taxonomy (5–10 classes) derived from empirical evidence in the AppleSupport corpus.

### Corpus Pattern Analysis
Analysis of AppleSupport root tweets and agent responses identifies recurring operational clusters:
1. **Software & OS Updates**: iOS update failures, upgrade bugs, storage full during update, battery drain immediately post-update.
2. **App Store, iTunes & Subscriptions**: App installation issues, billing/refunds, Apple ID sign-in errors, password resets.
3. **iCloud & Data Sync**: Photos not syncing, iCloud storage full, backup/restore failures, Contacts/Notes sync.
4. **Hardware, Audio & Display**: Cracked screens, camera black screen, speaker/mic distortion, touch screen unresponsive, charging port failure.
5. **Connectivity & Network**: Wi-Fi dropping, cellular "No Service", Bluetooth pairing failures, AirDrop issues.
6. **Battery & Device Performance**: Rapid battery discharge, unexpected shutdowns, device overheating, sluggish UI.
7. **Out-of-Scope / General Inquiry**: Complaints without actionable support issue, non-Apple device inquiries, praise/rants, incomprehensible text.

### Decision
Establish an **8-class intent taxonomy**:

| Intent Code | Intent Name | Description | Example Keywords / Phrases |
|---|---|---|---|
| `INT-IOS` | **iOS & System Updates** | OS installation, update errors, bricking, version compatibility | "iOS update failed", "stuck on Apple logo", "update bricked" |
| `INT-STORE` | **Account, Store & Billing** | Apple ID login, App Store downloads, subscriptions, refunds | "can't download apps", "charged twice", "Apple ID disabled" |
| `INT-ICLOUD` | **iCloud & Backup Sync** | iCloud drive, photos sync, backup restoration, storage quota | "photos won't sync", "iCloud backup failed", "storage full" |
| `INT-HARDWARE` | **Hardware & Display** | Physical screen, camera, buttons, speakers, microphone | "screen flickering", "camera black", "speaker crackling" |
| `INT-CONN` | **Connectivity & Bluetooth** | Wi-Fi disconnects, cellular signal, Bluetooth accessories, AirDrop | "no service", "won't connect to wifi", "bluetooth dropping" |
| `INT-BATTERY` | **Battery & Performance** | Fast drain, overheating, random shutdown, extreme lag | "battery draining fast", "phone overheating", "shutting down at 30%" |
| `INT-WATCH-MAC` | **Mac & Watch Ecosystem** | Watch pairing, macOS specific issues, watchOS sync | "Apple Watch won't pair", "MacBook trackpad", "watch sync" |
| `INT-OUT-OF-SCOPE` | **Out-of-Scope / Unclear** | Unactionable tweets, rants, non-support queries, missing context | "apple sucks", "why did you change the emoji", "hello" |

### Rationale
- Covers >98% of actionable AppleSupport domain requests without excessive class fragmentation.
- Each class maps directly to specific documentation domains and distinct troubleshooting flows.
- Includes mandatory `INT-OUT-OF-SCOPE` catch-all required by FR-I-001.

---

## 3. Research Area 3: Retrieval Strategy & Architecture (Constitution Principle VIII)

### Context & Problem
FR-R-001 requires retrieving top-$k$ historically resolved conversations relevant to an incoming query. Constitution Principle VIII mandates simplicity over unnecessary infrastructure (e.g., avoiding heavyweight external vector databases if in-memory solutions suffice).

### Technical Comparison

| Approach | Latency (63k docs) | RAM Footprint | External Services | Determinism | Eval Speed (<15 min) |
|---|---|---|---|---|---|
| **In-Memory BM25 (Okapi)** | ~1.2 ms / query | ~45 MB | None (pure Python) | 100% Deterministic | < 10 seconds for 200 queries |
| **TF-IDF + Cosine (scikit-learn)** | ~0.8 ms / query | ~35 MB | None (scikit-learn) | 100% Deterministic | < 8 seconds for 200 queries |
| **Dense Embeddings (Sentence-Transformers)** | ~25 ms / query | ~850 MB | PyTorch / GPU / Heavy deps | High | ~2–3 minutes (CPU) |
| **Managed Vector DB (Qdrant/Pinecone)** | ~15–40 ms / query | Heavy | Network daemon / Docker / Cloud | Dependent on network | Dependent on network |

### Decision
Implement a **hybrid two-stage in-memory retrieval engine**:
1. **Primary Stage**: Fast in-memory **Okapi BM25** index over the conversation initial turns and resolution texts, filtered/boosted by intent class.
2. **Scoring Function**: BM25 relevance score normalized to $[0, 1]$ via min-max scaling with a configurable relevance threshold (default $\tau_{retrieval} = 0.35$).
3. **Zero External Daemon**: Packaged as a pure Python module with serialized inverted index (pickle / compressed json), requiring zero standalone databases or network services.

### Rationale
- Completely fulfills Constitution Principle VIII (Simplicity) and IX (Evaluation speed: retrieval phase takes <10s for the entire golden set).
- Zero external infrastructure, runnable seamlessly across any local development environment.
- Highly interpretable: term matches and token overlaps are directly auditable in decision logs.

---

## 4. Research Area 4: LLM, Prompting & Determinism Strategy (Constitution Principle III & IX)

### Context & Problem
FR-G-001 through FR-G-004 and FR-V-006 require grounded draft reply generation, structured metadata output, and deterministic behavior under fixed seeds and zero temperature.

### Strategy & Architecture
1. **Provider Abstraction**: A lightweight `LLMClient` interface wrapping generative models with standard request/response schemas:
   - Default primary provider: Google Gemini API (`google-generativeai` / `gemini-2.5-flash` or `gemini-1.5-flash`).
   - Built-in Mock / Offline Stub Provider: Zero-network deterministic heuristic generator for testing and offline CI execution.
2. **Prompt Template Structure**:
   - Explicit System Instructions: Strict role (AppleSupport AI assistant), grounding constraints (cite retrieved IDs, do not invent steps).
   - In-Context Grounding: Top-$k$ retrieved conversation dialogue snippets formatted as structured evidence blocks.
   - Output Schema: JSON format containing `{draft_reply, cited_conversation_ids, confidence, grounding_rationale}`.
3. **Determinism Enforcement**:
   - `temperature = 0.0`
   - Fixed random seeds (`seed = 42`)
   - Schema enforcement via Pydantic response validation.

---

## 5. Research Area 5: Escalation & Routing Decision Framework (FR-E)

### Context & Problem
FR-E-001 through FR-E-005 require a transparent, configurable decision engine determining `auto-handle` vs `escalate`.

### Decision Logic & Rule Composition
A conversation is routed to `escalate` if **ANY** of the following condition triggers are met:

```python
def evaluate_escalation(
    intent_pred: IntentPrediction,
    retrieved: RetrievedEvidence,
    conversation: Conversation,
    config: EscalationConfig
) -> EscalationDecision:
    triggers = []
    
    # Trigger 1: Out-of-Scope or Unclear Intent
    if intent_pred.label == "INT-OUT-OF-SCOPE":
        triggers.append("out_of_scope_intent")
        
    # Trigger 2: Low Intent Classifier Confidence
    if intent_pred.confidence < config.intent_confidence_threshold:
        triggers.append(f"low_intent_confidence ({intent_pred.confidence:.2f} < {config.intent_confidence_threshold:.2f})")
        
    # Trigger 3: Insufficient or Low-Relevance Evidence
    if not retrieved.items or retrieved.top_score < config.retrieval_min_score:
        triggers.append(f"insufficient_evidence (top_score={retrieved.top_score:.2f} < {config.retrieval_min_score:.2f})")
        
    # Trigger 4: Safety / Sensitive Domain Keywords
    sensitive_keywords = ["lawsuit", "lawyer", "legal", "stolen", "police", "fraud", "hacked", "unauthorized charge"]
    if any(kw in conversation.root_text.lower() for kw in sensitive_keywords):
        triggers.append("sensitive_domain_trigger")
        
    # Trigger 5: Pure DM-Escalation Precedent
    if retrieved.items and all(item.is_dm_escalation for item in retrieved.items):
        triggers.append("historical_precedent_requires_dm")

    routing = "escalate" if len(triggers) > 0 else "auto-handle"
    return EscalationDecision(
        routing=routing,
        triggers=triggers,
        intent_confidence=intent_pred.confidence,
        evidence_top_score=retrieved.top_score,
        rationale="; ".join(triggers) if triggers else "Passed all auto-handling criteria"
    )
```

### Configurable Parameters
- `intent_confidence_threshold`: default `0.75` (tunable via CLI/config).
- `retrieval_min_score`: default `0.35` (tunable via CLI/config).

---

## 6. Summary of Engineering Choices & Compliance

| Spec / Constitution Gate | Chosen Technical Solution | Verification Mechanism |
|---|---|---|
| **Data Immutability (FR-D-001)** | Read raw CSV with stream chunking; write to `data/processed/` | SHA-256 hash checks on `twcs.csv` |
| **Leakage Prevention (FR-D-006)** | Monotonic temporal split before 2017-11-01 | Set intersection assertion on conversation IDs |
| **Eval Speed < 15 min (SC-006)** | In-memory BM25 index + parallel LLM eval calls | Timing benchmarks in evaluation harness |
| **Deterministic Outputs (NFR-001)** | Documented random seeds (`42`), `temperature=0.0` | Ingestion hash reproducibility test |

