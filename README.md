# Hiver Support Agent

An evaluation-first, modular AI customer support agent designed for Apple customer service interactions using the Customer Support on Twitter dataset (`data/raw/twcs.csv`).

The system ingests multi-turn customer support conversations, classifies inquiries into an 8-class empirical taxonomy, retrieves historically resolved precedent conversations using an in-memory Okapi BM25 engine, drafts grounded replies citing precedent, and applies a multi-trigger escalation routing engine to decide between automated handling and human operator escalation.

---

## 1. Overview

The **Hiver Support Agent** provides an end-to-end pipeline for handling inbound customer support queries with explicit uncertainty calibration and grounded generation.

- **Customer Support Use Case**: High-volume triage, automated resolution of routine technical issues (e.g., battery health, iOS update procedures, iCloud sync), and fail-safe routing of complex, ambiguous, or high-risk inquiries (e.g., account security, billing disputes, legal concerns) to human agents.
- **Brand Focus (`AppleSupport`)**: Selected based on empirical dataset analysis demonstrating high clean conversation volume (74,571 threads), dense troubleshooting link citations (75.4%), and realistic escalation boundaries.

---

## 2. Architecture

The codebase is structured into testable, decoupled modules under `src/`:

```
src/
├── data/          # Streaming CSV ingestion, BFS thread reconstruction, temporal partitioning
├── intent/        # 8-class intent taxonomy & calibrated intent classification
├── retrieval/     # In-memory Okapi BM25 indexer & evidence retriever
├── generator/     # Grounded reply agent & LLM client abstraction (Gemini + Offline Mock)
├── escalation/    # Multi-trigger rule engine & confidence-based routing router
├── eval/          # Evaluation harness, metrics computation, LLM-as-judge scoring
├── pipeline.py    # End-to-end ticket processing orchestrator
├── config.py      # App configuration and threshold settings
├── models.py      # Core Pydantic data schemas & contracts
└── cli.py         # Unified CLI interface
```

### Module Breakdown:
1. **Ingestion & Conversation Reconstruction (`src/data/`)**: Parses raw tweets and reconstructs complete conversation dialogue trees using deterministic breadth-first search (BFS) starting from customer roots.
2. **Intent Classification (`src/intent/`)**: Classifies queries into an 8-class taxonomy (`INT-IOS`, `INT-STORE`, `INT-ICLOUD`, `INT-HARDWARE`, `INT-CONN`, `INT-BATTERY`, `INT-WATCH-MAC`, `INT-OUT-OF-SCOPE`) with confidence scores and supporting excerpts.
3. **Historical Retrieval Engine (`src/retrieval/`)**: In-memory Okapi BM25 index over pre-split historical conversations. Retrieves top-$k$ relevant precedent threads without requiring external vector databases.
4. **Grounded Reply Generator (`src/generator/`)**: Generates draft responses strictly conditioned on retrieved precedent conversations and links, citing evidence IDs.
5. **Escalation Router (`src/escalation/`)**: Evaluates safety triggers (sensitive keywords, low confidence, out-of-scope intent, missing evidence, DM precedent) to decide between `auto-handle` and `escalate`.
6. **Evaluation Harness (`src/eval/`)**: Automated evaluation suite executing against benchmark datasets with strict zero-leakage assertions.

---

## 3. Dataset & Data Partitions

The system operates on the **Customer Support on Twitter (TWCS)** dataset:

- **Raw Data**: `data/raw/twcs.csv` (2,811,774 rows, 516.5 MB).
- **Temporal Split Boundary**: Cutoff at **`2017-11-01 00:00:00 UTC`**.
  - **Retrieval Corpus (`data/processed/retrieval_corpus.jsonl`)**: 63,842 conversations with root timestamps strictly prior to 2017-11-01.
  - **Evaluation Candidate Pool (`data/processed/eval_candidate_pool.jsonl`)**: 10,729 post-split conversations.
- **Held-Out Human Benchmark (`data/processed/human_benchmark_unannotated.jsonl`)**: 200 authentic conversations sampled with fixed seed `42` from the post-split candidate pool and frozen with a cryptographic SHA-256 manifest (`data/processed/human_benchmark_manifest.json`).
- **No-Leakage Principle**: Enforces strict conversation-level disjointness between the retrieval corpus, development golden set, and human evaluation benchmark. Pre-evaluation audit commands raise a hard failure (`exit code 2`) if any overlap is detected.

---

## 4. Setup & Installation

### Requirements
- **Python**: `>= 3.10` (tested on Python 3.13)
- **OS**: Windows / macOS / Linux

### Installation
```bash
# 1. Clone repository and navigate to folder
cd hiver-support-agent

# 2. Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Environment Variables
For LLM-assisted generation and LLM judge evaluation:
```bash
# Set Gemini API Key (Optional - offline deterministic mock fallback runs automatically if unset)
export GEMINI_API_KEY="your-api-key-here"
# On Windows PowerShell:
$env:GEMINI_API_KEY="your-api-key-here"
```
> **Note**: Do not commit or hardcode real API keys. If `GEMINI_API_KEY` is not provided, the system seamlessly uses its built-in offline mock provider.

---

## 5. Running the Application

The application is controlled via the CLI (`python -m src.cli` or installed command `hiver-agent`).

### 5.1 Dataset Ingestion & Preprocessing
```bash
# Reconstruct conversations and generate temporal partitions
python -m src.cli ingest \
  --raw-path data/raw/twcs.csv \
  --output-dir data/processed \
  --split-date 2017-11-01 \
  --brand AppleSupport
```

### 5.2 Data Leakage Audit
```bash
# Verify zero conversation ID overlap across partitions
python -m src.cli audit-leakage \
  --manifest data/processed/split_manifest.json \
  --golden-set data/processed/golden_set.jsonl \
  --human-benchmark data/processed/human_benchmark_unannotated.jsonl
```

### 5.3 Single Ticket Inference
```bash
# Run end-to-end inference on a single customer inquiry
python -m src.cli handle \
  --text "My iPhone 7 battery is draining from 100% to zero in 3 hours after updating iOS. Any tips?" \
  --json
```

### 5.4 Running Evaluation Harness
```bash
# Run full evaluation against authoritative human benchmark
python -m src.cli evaluate \
  --benchmark human \
  --manifest data/processed/split_manifest.json \
  --retrieval-corpus data/processed/retrieval_corpus.jsonl \
  --output-report reports/human_evaluation_report.json

# Run evaluation against development golden set
python -m src.cli evaluate \
  --benchmark golden \
  --manifest data/processed/split_manifest.json \
  --retrieval-corpus data/processed/retrieval_corpus.jsonl \
  --output-report reports/golden_evaluation_report.json
```

### 5.5 Running Tests
```bash
# Execute unit, integration, and contract tests
python -m pytest
```

---

## 6. Authoritative Evaluation Results

Below are the authoritative final evaluation metrics computed on the **held-out human evaluation benchmark** (200 authentic post-split AppleSupport conversations, Run ID: `EVAL-RUN-1789407471`):

| Metric | Measured Score | Baseline Score | Evaluation Standard |
|---|:---:|:---:|:---:|
| **Intent Macro F1** | **0.501056** | 0.092271 | 8-Class Macro Average |
| **Retrieval MRR@5** | **0.849394** | 0.000000 | Mean Reciprocal Rank @ 5 |
| **Retrieval Precision@5** | **1.000000** | 0.000000 | Precision @ 5 |
| **Reply Groundedness Rate** | **1.000000** | 0.500000 | LLM Judge (Score $\ge 3$) |
| **False-Auto-Handle Rate** | **0.627907** | 0.500000 | Safety Target $\le 0.10$ |
| **Escalation Precision** | **0.205128** | 0.500000 | Precision of Escalation Flag |
| **Escalation Recall** | **0.372093** | 0.500000 | Recall of Escalation Need |
| **LLM Judge Agreement** | **0.778689** | N/A | Human vs Judge Agreement |
| **Execution Runtime** | **46.30s** | $< 900$s | Full 200-sample benchmark |

### Production Readiness Notice & Safety Limitations
> **CRITICAL NOTICE**: While retrieval accuracy (MRR@5: `0.849`) and reply groundedness (`1.000`) meet high performance bars, **routing safety remains the primary limitation** (False-Auto-Handle Rate: `0.628`, Escalation Precision: `0.205`). Consequently, **the system is NOT production-ready for unsupervised customer-facing deployment** and must operate in an agent-assist / human-in-the-loop workflow.

---

## 7. Project Documentation

Comprehensive documentation is organized under the [`docs/`](docs/) directory:

- [**docs/SPECIFICATION.md**](docs/SPECIFICATION.md): Complete requirements, user stories, and acceptance criteria.
- [**docs/ARCHITECTURE.md**](docs/ARCHITECTURE.md): System design, data flow diagrams, technical research, and constitution principles.
- [**docs/DATASET_PROFILE.md**](docs/DATASET_PROFILE.md): Statistical profile and audit of the Kaggle TWCS dataset.
- [**docs/DATA_MODEL.md**](docs/DATA_MODEL.md): Core Pydantic schemas, entity definitions, and lifecycle states.
- [**docs/DECISION_LOG.md**](docs/DECISION_LOG.md): Architectural decision records (ADRs) with rationale and rejected alternatives.
- [**docs/EVALUATION_REPORT.md**](docs/EVALUATION_REPORT.md): Detailed evaluation methodology, failure mode breakdown, and error analysis.
- [**docs/QUICKSTART.md**](docs/QUICKSTART.md): Step-by-step developer guide and validation scenarios.
- [**docs/contracts/CLI_CONTRACT.md**](docs/contracts/CLI_CONTRACT.md): CLI specifications, exit codes, and output schemas.
- [**docs/contracts/PIPELINE_SCHEMAS.json**](docs/contracts/PIPELINE_SCHEMAS.json): Machine-readable JSON schemas for pipeline interfaces.
