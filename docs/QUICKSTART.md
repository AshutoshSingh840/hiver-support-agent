# Quickstart Validation Guide: Hiver Support Agent

**Feature Branch**: `001-support-agent`  
**Date**: 2026-09-12  
**Status**: Complete  

This guide provides step-by-step instructions for setting up the environment, running the deterministic data pipeline, executing the complete automated evaluation suite, and validating end-to-end agent inference.

---

## 1. Prerequisites & Environment Setup

### System Requirements
- Python 3.10+ (tested on Python 3.13)
- Windows / macOS / Linux
- Raw dataset placed at `data/raw/twcs.csv` (SHA-256 verified)

### Installation
```bash
# 1. Clone & enter project repository
cd hiver-support-agent

# 2. Create and activate virtual environment (optional but recommended)
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# 3. Install core dependencies
pip install -r requirements.txt
```

---

## 2. Validation Scenario 1: Deterministic Ingestion & Leakage Check

Run the dataset ingestion pipeline to reconstruct conversation threads, generate temporal partitions, and verify zero eval leakage.

```bash
# Execute ingestion
python -m src.cli ingest \
  --raw-path data/raw/twcs.csv \
  --output-dir data/processed \
  --split-date 2017-11-01 \
  --brand AppleSupport

# Verify zero data leakage
python -m src.cli audit-leakage \
  --retrieval-manifest data/processed/retrieval_manifest.json \
  --golden-set data/processed/golden_set.jsonl
```

**Expected Outcome**:
- Output logs report exactly 74,571 AppleSupport conversations reconstructed.
- Retrieval index built with 63,842 historical conversations.
- Audit leakage command exits with code `0` and outputs `[PASS] 0 overlapping conversation IDs`.

---

## 3. Validation Scenario 2: Single-Command Automated Evaluation

Run the entire evaluation suite across all headline metrics (intent classification, retrieval relevance, reply groundedness, escalation accuracy, baseline comparisons, and failure analysis).

```bash
# Execute full evaluation harness against curated development benchmark
python -m src.cli evaluate \
  --golden-set data/processed/golden_set.jsonl \
  --retrieval-corpus data/processed/retrieval_corpus.jsonl \
  --manifest data/processed/split_manifest.json \
  --output-report reports/evaluation_report.json \
  --escalation-threshold 0.75
```

**Expected Outcome**:
- Evaluation completes in $< 15$ minutes on a single machine.
- Console outputs a structured metric summary table:
  - Intent Macro F1 $> 0.70$ (vs majority baseline)
  - Retrieval MRR@5 $> 0.50$ (vs random baseline)
  - False-Auto-Handle rate $\le 10\%$
  - LLM-judge agreement rate $\ge 70\%$
- Detailed evaluation report saved to `reports/evaluation_report.json` containing $\ge 5$ failure mode traces.

---

## 3.1 Independent Human-Validated Benchmark Workflow

```bash
# 1. Sample 200 unannotated candidate records from post-split pool
python -m src.cli sample-human-benchmark \
  --count 200 \
  --seed 42 \
  --output data/processed/human_benchmark_unannotated.jsonl

# 2. Audit 3-way disjointness across partitions
python -m src.cli audit-leakage \
  --manifest data/processed/split_manifest.json \
  --golden-set data/processed/golden_set.jsonl \
  --human-benchmark data/processed/human_benchmark_unannotated.jsonl

# 3. Validate completed human benchmark (after manual labeling)
python -m src.cli validate-human-benchmark \
  --benchmark-file data/processed/human_benchmark.jsonl

# 4. Freeze completed benchmark into immutable cryptographic manifest
python -m src.cli freeze-human-benchmark \
  --benchmark-file data/processed/human_benchmark.jsonl \
  --output-manifest data/processed/human_benchmark_manifest.json \
  --version v1.0-human
```


---

## 4. Validation Scenario 3: End-to-End Single Ticket Inference

Submit a sample customer query to inspect the full trace: intent classification, evidence retrieval, escalation decision, and draft reply generation.

```bash
# Test an auto-handle battery inquiry
python -m src.cli handle \
  --text "My iPhone 7 battery is draining from 100% to zero in 3 hours after updating iOS. Any tips?" \
  --json
```

**Expected Outcome**:
- Returns structured JSON containing:
  - `intent.label`: `INT-BATTERY` (confidence $\ge 0.85$)
  - `retrieval.evidence`: Top-3 AppleSupport battery resolution threads
  - `escalation.routing`: `auto-handle`
  - `draft_reply.text`: Grounded advice referencing battery health settings

```bash
# Test a sensitive escalation trigger
python -m src.cli handle \
  --text "Someone hacked into my Apple ID and charged $400 to my card. I'm taking legal action!" \
  --json
```

**Expected Outcome**:
- `escalation.routing`: `escalate`
- `escalation.triggers`: `["sensitive_domain_trigger", "security_escalation"]`
- `draft_reply`: `null` (safely routed to human support)

---

## 5. Automated Test Suite Execution

Run unit, contract, and integration tests across all modules:

```bash
# Run all test suites
pytest tests/ -v
```
