# Hiver Support Agent

An evaluation-first, modular AI customer support agent designed for Apple customer service interactions using the Customer Support on Twitter dataset (`data/raw/twcs.csv`).

The system ingests multi-turn customer support conversations, classifies inquiries into an 8-class empirical taxonomy with calibrated confidence, retrieves historically resolved precedent conversations using an in-memory Okapi BM25 engine, drafts grounded replies citing evidence IDs, and computes a continuous probabilistic routing risk score ($R(x) \in [0, 1]$) to decide between automated handling and human operator escalation.

---

## 1. Overview & Core Capabilities

The **Hiver Support Agent** provides an end-to-end, scientifically evaluated pipeline for high-volume customer support triage:
- **Intent Classification**: 8-class empirical taxonomy (`INT-IOS`, `INT-STORE`, `INT-ICLOUD`, `INT-HARDWARE`, `INT-CONN`, `INT-BATTERY`, `INT-WATCH-MAC`, `INT-OUT-OF-SCOPE`) with calibrated continuous confidence.
- **Historically Grounded Reply Drafting**: Okapi BM25 candidate retrieval strictly conditions draft responses on verified resolution precedent and official Apple documentation links.
- **Continuous Routing Risk Scoring**: Computes an interpretable, continuous risk score $R(x) \in [0.0, 1.0]$ via Noisy-OR probability aggregation across safety keywords, legal risk, fraud, hardware danger, bricking, language detection, intent ambiguity, retrieval evidence deficit, and customer frustration cues.
- **Safety-First Threshold Selection**: Operating threshold $\tau = 0.45$ selected on the development golden set to maximize safety recall ($85.0\%$) and minimize false-auto-handling ($15.0\%$) while maintaining $69.0\%$ automated coverage.
- **3-Way Baseline Comparison**: Benchmarked against both a **Trivial Baseline** (Majority Class + Random Retrieval) and a genuine **Simple ML Baseline** (Supervised TF-IDF + Logistic Regression).
- **Two-Tier Benchmark Evaluation**: Evaluated against a **200-sample hand-labelled Development Golden Set** and validated on a **200-sample frozen Held-Out Human Benchmark** with strict zero-leakage enforcement.

---

## 2. Architecture & Directory Structure

```
hiver-support-agent/
├── data/
│   ├── raw/                 # Raw dataset (twcs.csv) - gitignored
│   └── processed/           # Reconstructed partitions, retrieval corpus, golden & human benchmarks
├── src/
│   ├── data/                # CSV ingestion, BFS thread reconstruction, temporal splitting
│   ├── intent/              # 8-class taxonomy definitions & calibrated classifier
│   ├── retrieval/           # In-memory Okapi BM25 indexer & evidence retriever
│   ├── generator/           # Grounded reply agent (Gemini + Deterministic Offline Mock)
│   ├── escalation/          # Continuous risk scoring router ($R(x)$) & safety triggers
│   ├── eval/                # Evaluation harness, baselines, metrics, LLM-as-a-judge
│   ├── pipeline.py          # End-to-end ticket processing orchestrator
│   ├── config.py            # App configuration dataclasses & threshold settings
│   ├── models.py            # Core Pydantic data schemas & contracts
│   └── cli.py               # Click CLI interface (handle, evaluate, audit-leakage, etc.)
├── docs/                    # Architecture, specification, data model, decision log, evaluation report
├── tests/                   # 85 automated unit, integration, and contract tests
├── reports/                 # JSON evaluation reports and benchmark summaries
├── pyproject.toml           # Project packaging and tool configuration
└── requirements.txt         # Core dependencies (pydantic, scikit-learn, numpy, click, etc.)
```

---

## 3. Evaluation Results & 3-Way Baseline Comparison

### 3.1 Intent Classification Baselines Comparison

| Model / Architecture | Intent Macro F1 (Dev Golden Set) | Intent Macro F1 (Held-Out Human Benchmark) | Deterministic & Local? | Latency |
| :--- | :---: | :---: | :---: | :---: |
| **[1] Trivial Baseline (Majority Class)** | `0.0278` | `0.0923` | Yes | < 0.01s |
| **[2] Simple ML Baseline (TF-IDF + LogReg)** | `0.5803` | `0.3590` | Yes | ~4.2s (train on 4k corpus) |
| **[3] Hiver Support Agent (Hybrid Model)** | **`0.8027`** | **`0.5025`** | Yes | < 0.05s / inference |

---

### 3.2 Headline Evaluation Scorecard

| Headline Metric | Dev Golden Set Score | Held-Out Human Benchmark Score | Target Threshold | Operational Status |
| :--- | :---: | :---: | :---: | :---: |
| **Intent Macro F1** | **`0.8027`** | `0.5025` | $\ge 0.700$ | **PASS** (Dev) / Limitation (Held-Out) |
| **Retrieval MRR@5** | **`0.8783`** | **`0.8510`** | $\ge 0.500$ | **PASS (Consistently Strong)** |
| **Retrieval Precision@5** | **`1.0000`** | **`1.0000`** | $\ge 0.600$ | **PASS** |
| **Reply Groundedness Rate** | **`1.0000`** (100%) | **`1.0000`** (100%) | $\ge 0.850$ | **PASS (Zero Hallucinations)** |
| **False-Auto-Handle Rate (Safety)** | **`0.1500`** (15.0%) | `0.6279` (62.8%) | $\le 0.100$ | **FAIL (Key Production Limitation)** |
| **Escalation Precision** | **`0.5484`** (54.8%) | `0.1765` (17.7%) | $\ge 0.750$ | Trade-off for Safety Sensitivity |
| **Escalation Recall** | **`0.8500`** (85.0%) | `0.3721` (37.2%) | $\ge 0.800$ | **PASS** (Dev) / Limitation (Held-Out) |
| **Auto-Handle Traffic Coverage** | **`69.0%`** (138/200) | `54.5%` (109/200) | $\ge 50.0\%$ | **PASS** |
| **LLM Judge Agreement** | **`0.7787`** (77.9%) | **`0.7787`** | $\ge 0.700$ | **PASS** |
| **Evaluation Runtime** | **~37.4s** | **~37.7s** | $\le 900.0\text{s}$ | **PASS** |

---

### 3.3 Routing Threshold Sweep (Safety vs. Coverage Trade-Off)

Continuous risk threshold sweep $\tau \in [0.10, 0.90]$ evaluated on the Development Golden Set:

| Threshold ($\tau$) | False-Auto-Handle Rate | Escalation Precision | Escalation Recall | Auto-Handle Coverage | Escalation F1 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `0.10` - `0.35` | 15.0% | 54.8% | 85.0% | 69.0% | 0.6667 |
| **`0.45` (Selected)** | **15.0%** | **54.8%** | **85.0%** | **69.0%** | **0.6667** |
| `0.60` | 30.0% | 54.9% | 70.0% | 74.5% | 0.6154 |
| `0.70` | 35.0% | 74.3% | 65.0% | 82.5% | 0.6933 |
| `0.80` | 62.5% | 79.0% | 37.5% | 90.5% | 0.5085 |

---

## 4. Production Readiness Notice & Safety Limitations

> [!WARNING]
> **CRITICAL SAFETY NOTICE**: While retrieval accuracy (MRR@5 = 0.851) and reply groundedness (100%) are exceptionally high, **the system is NOT production-ready for unsupervised autonomous deployment**. The held-out false-auto-handle rate of 62.8% on authentic open-ended social tweets proves that single-turn tweets contain sarcasm, implicit failures, and frustration that cannot be safely triaged without multi-turn conversational context or human-in-the-loop oversight.

### What is Misleading About My Headline Number?
- **Class Balance Distortion**: The development set is balanced across 8 classes, whereas live Twitter traffic has a 58.5% iOS skew.
- **Groundedness $\neq$ Safety**: An automated reply can be 100% grounded in official documentation, but replying with a macro to an enraged customer complaining about a known bug destroys customer trust.
- **Single-Turn Blindness**: Twitter customer queries lack preceding thread context, leading to missed escalation cues.

---

## 5. Quickstart & CLI Commands

### 5.1 Installation
```bash
# Clone and setup environment
cd hiver-support-agent
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 5.2 Single Ticket Inference
```bash
# Process a customer ticket through the end-to-end pipeline
python -m src.cli handle \
  --text "My iPhone 7 battery is dying within 2 hours after updating to iOS 11. Any suggestions?"
```

### 5.3 Run Full Evaluation Benchmark
```bash
# Run evaluation on the Development Golden Set
python -m src.cli evaluate --benchmark golden --output-report reports/evaluation_report.json

# Run evaluation on the Frozen Held-Out Human Benchmark (Read-Only)
python -m src.cli evaluate --benchmark human --output-report reports/human_benchmark_report.json
```

### 5.4 Run Full Test Suite
```bash
# Run all 85 unit, integration, and contract tests
python -m pytest
```

### 5.5 Pre-Run Data Leakage Audit
```bash
python -m src.cli audit-leakage \
  --manifest data/processed/split_manifest.json \
  --golden-set data/processed/golden_set.jsonl \
  --human-benchmark data/processed/human_benchmark_unannotated.jsonl
```

---

## 6. Dataset Partitions & Provenance

- **Source Dataset**: Customer Support on Twitter (TWCS), focusing on `@AppleSupport`.
- **Temporal Split Cutoff**: `2017-11-01 00:00:00 UTC` strictly separates past precedent from evaluation candidates.
- **Retrieval Corpus (`data/processed/retrieval_corpus.jsonl`)**: 29,591 pre-split conversations ($\le$ 2017-11-01).
- **Development Golden Set (`data/processed/golden_set.jsonl`)**: 200 hand-labelled examples curated across all 8 intent classes by the author.
- **Held-Out Human Benchmark (`data/processed/human_benchmark_unannotated.jsonl`)**: 200 authentic post-split conversations sampled with fixed seed `42` and frozen with SHA-256 hash `dbc2b1dd...`.

---

## 7. Key Documentation Links

- [**docs/EVALUATION_REPORT.md**](docs/EVALUATION_REPORT.md): Comprehensive evaluation analysis, baseline comparisons, threshold curves, failure modes, and one-week next steps.
- [**docs/DECISION_LOG.md**](docs/DECISION_LOG.md): 14 non-obvious engineering decisions, trade-offs, and design rationales.
- [**docs/ARCHITECTURE.md**](docs/ARCHITECTURE.md): System architecture, data flow diagrams, and module contracts.
- [**docs/SPECIFICATION.md**](docs/SPECIFICATION.md): Feature requirements and acceptance criteria.
