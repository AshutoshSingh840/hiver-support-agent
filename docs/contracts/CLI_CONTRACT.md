# CLI Interface Contract: Hiver Support Agent

**Feature Branch**: `001-support-agent`  
**Date**: 2026-09-12  
**Status**: Complete  

This contract defines the command-line interface, argument syntax, environment variables, exit codes, and standard output formats for the Hiver Support Agent CLI (`hiver-agent` or `python -m src.cli`).

---

## 1. Global Flags & Conventions

```bash
python -m src.cli [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS] [ARGS]
```

### Global Options
- `--config PATH`: Path to custom YAML configuration file (default: `config/default.yaml`).
- `--verbose / -v`: Enable detailed debug logging to stderr.
- `--json`: Output machine-readable JSON to stdout instead of formatted text tables.
- `--seed INT`: Set explicit pseudo-random seed for determinism (default: `42`).

### Exit Codes
- `0`: Success.
- `1`: General runtime or validation error.
- `2`: Data leakage detected (hard failure).
- `3`: Missing prerequisites or input files.

---

## 2. Command Specifications

### 2.1 `ingest` — Ingest & Reconstruct Dataset

Reconstructs conversation threads from `data/raw/twcs.csv`, applies filters, performs temporal partitioning, and builds the retrieval index.

```bash
python -m src.cli ingest \
  --raw-path data/raw/twcs.csv \
  --output-dir data/processed \
  --split-date 2017-11-01 \
  --brand AppleSupport
```

**Options**:
- `--raw-path PATH` (required): Path to raw `twcs.csv`.
- `--output-dir PATH` (default: `data/processed`): Output directory for reconstructed parquet/jsonl files.
- `--split-date TEXT` (default: `2017-11-01`): Temporal cutoff date `YYYY-MM-DD`.
- `--brand TEXT` (default: `AppleSupport`): Target brand handle.

**Output (JSON format)**:
```json
{
  "status": "success",
  "input_file": "data/raw/twcs.csv",
  "input_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "total_raw_rows": 2811774,
  "reconstructed_conversations": 74571,
  "retrieval_corpus_size": 63842,
  "eval_candidate_pool_size": 10729,
  "excluded_broken_threads": 3862,
  "runtime_sec": 14.82,
  "manifest_path": "data/processed/split_manifest.json"
}
```

---

### 2.2 `evaluate` — Run Full Automated Evaluation Pipeline

Executes the complete evaluation suite against the 150–250 example golden set and generates the evaluation report.

```bash
python -m src.cli evaluate \
  --golden-set data/processed/golden_set.jsonl \
  --retrieval-index data/processed/retrieval_index.bm25 \
  --output-report reports/evaluation_report.json \
  --escalation-threshold 0.75 \
  --llm-provider gemini
```

**Options**:
- `--golden-set PATH` (required): Path to curated evaluation benchmark dataset.
- `--retrieval-index PATH`: Path to serialized BM25 retrieval index.
- `--output-report PATH` (default: `reports/evaluation_report.json`): Destination path for report.
- `--escalation-threshold FLOAT` (default: `0.75`): Confidence threshold for auto-handling.
- `--llm-provider [gemini|mock]` (default: `gemini`): Generative model provider.

**Exit Behavior**:
- Fails with exit code `2` if pre-run conversation ID leakage is detected.
- Emits human-readable summary table to stdout; saves structured JSON report to `--output-report`.

---

### 2.3 `handle` — Single Conversation End-to-End Inference

Takes a single customer tweet or conversation dialogue, determines intent, retrieves precedent, decides escalation, and generates a draft reply if auto-handled.

```bash
python -m src.cli handle \
  --text "My iPhone 8 battery drains completely in 2 hours after updating to iOS 11. What should I do?" \
  --json
```

**Output (JSON format)**:
```json
{
  "conversation_id": 9999001,
  "intent": {
    "label": "INT-BATTERY",
    "confidence": 0.94,
    "supporting_excerpt": "battery drains completely in 2 hours after updating"
  },
  "retrieval": {
    "top_k": 3,
    "evidence": [
      {
        "source_conversation_id": 115854,
        "relevance_score": 0.88,
        "resolution_text": "Let's help with your battery. Check Battery Health in Settings > Battery..."
      }
    ]
  },
  "escalation": {
    "routing": "auto-handle",
    "triggers": [],
    "rationale": "High intent confidence (0.94) and high relevance evidence available (0.88)"
  },
  "draft_reply": {
    "text": "We'd like to help get your battery running smoothly. Go to Settings > Battery > Battery Health to check maximum capacity. If you recently updated, apps may be re-indexing in the background.",
    "cited_evidence_ids": [115854]
  },
  "latency_ms": 342.1
}
```

---

### 2.4 `audit-leakage` — Standalone Leakage Verification

Runs an automated set-intersection check between the retrieval corpus, curated development benchmark, and human test benchmark (3-way disjointness).

```bash
python -m src.cli audit-leakage \
  --manifest data/processed/split_manifest.json \
  --golden-set data/processed/golden_set.jsonl \
  --human-benchmark data/processed/human_benchmark_unannotated.jsonl
```

**Exit Codes**:
- `0`: Zero overlap across all partitions (Pass).
- `2`: Overlap detected (Fail).

---

### 2.5 `sample-human-benchmark` — Deterministic Human Candidate Sampling

Samples 200 authentic post-split AppleSupport conversations for human annotation with unpopulated ground-truth fields.

```bash
python -m src.cli sample-human-benchmark \
  --manifest data/processed/split_manifest.json \
  --golden-set data/processed/golden_set.jsonl \
  --output data/processed/human_benchmark_unannotated.jsonl \
  --count 200 \
  --seed 42
```

---

### 2.6 `validate-human-benchmark` — Human Benchmark Integrity Validator

Verifies that all 200 human benchmark records have complete manual labels, valid non-automated `annotator_id`, and zero leakage.

```bash
python -m src.cli validate-human-benchmark \
  --benchmark-file data/processed/human_benchmark.jsonl \
  --manifest data/processed/split_manifest.json \
  --golden-set data/processed/golden_set.jsonl
```

---

### 2.7 `freeze-human-benchmark` — Cryptographic Benchmark Freezer

Freezes validated human benchmark and outputs an immutable SHA-256 manifest.

```bash
python -m src.cli freeze-human-benchmark \
  --benchmark-file data/processed/human_benchmark.jsonl \
  --raw-path data/raw/twcs.csv \
  --manifest data/processed/split_manifest.json \
  --golden-set data/processed/golden_set.jsonl \
  --output-manifest data/processed/human_benchmark_manifest.json \
  --version v1.0-human
```

