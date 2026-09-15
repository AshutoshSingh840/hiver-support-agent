# Engineering & Architectural Decision Log: Hiver Support Agent

**Specification**: [Feature Specification](SPECIFICATION.md)  
**Architecture Reference**: [Hiver Support Agent Architecture](ARCHITECTURE.md)  
**Evaluation Reference**: [Evaluation Report](EVALUATION_REPORT.md)  

This document logs **14 non-obvious engineering and product design decisions** made across all phases of the project lifecycle, explaining rationale, alternatives evaluated, trade-offs, and verification.

---

### Entry 001: Focus Brand Selection (AppleSupport)
- **Date**: 2026-09-10
- **Decision**: Focus exclusively on `AppleSupport` as the single brand prototype for development and evaluation.
- **Alternatives Considered**: Multi-brand mixture, `AmazonHelp` (high volume, extreme domain variance), `Uber_Support` (short routing queries), `Delta`/`AmericanAir` (frequent schedule lookups requiring real-time external API integration).
- **Evidence & Rationale**:
  - `data/raw/twcs.csv` contains 106,860 AppleSupport outbound messages and 74,571 clean reconstructable conversations.
  - Highest documentation link citation rate (75.4%), enabling rich, grounded historical precedent retrieval.
  - High English proportion (87.5%), allowing clean semantic evaluation.
  - Natural 52.6% DM escalation boundary providing a clean, realistic escalation signal.
- **Trade-offs**: Single ecosystem focus; results may not generalize identically to airlines or logistics without domain tuning.

---

### Entry 002: Monotonic Temporal Train/Eval Split Boundary
- **Date**: 2026-09-12
- **Decision**: Set temporal split cutoff strictly at `2017-11-01 00:00:00 UTC`. Earlier conversations form the historical retrieval corpus; later conversations form the candidate pool for evaluation sets.
- **Alternatives Considered**: Random 80/20 conversation splitting, stratified hash splitting.
- **Evidence & Rationale**:
  - Customer support operates forward in time; retrieval indexing must only have access to precedent resolved in the past.
  - Temporal boundary guarantees zero temporal contamination (no future conversations answering past inquiries).
  - Yields 29,591 pre-split retrieval conversations and a disjoint post-split candidate pool.
- **Trade-offs**: Evaluation set is drawn from post-November 2017 traffic, reflecting real seasonal software update distributions (e.g., iOS 11 launch).

---

### Entry 003: 8-Class Empirical Intent Taxonomy
- **Date**: 2026-09-12
- **Decision**: Define an 8-class empirical taxonomy (`INT-IOS`, `INT-STORE`, `INT-ICLOUD`, `INT-HARDWARE`, `INT-CONN`, `INT-BATTERY`, `INT-WATCH-MAC`, `INT-OUT-OF-SCOPE`).
- **Alternatives Considered**: 3-class coarse taxonomy (Software, Hardware, Account), 25-class granular taxonomy.
- **Evidence & Rationale**:
  - Balances operational distinctness with high classification precision.
  - Directly reflects empirical problem clusters in the AppleSupport corpus.
  - Includes mandatory out-of-scope catch-all to safely capture rants and non-support social queries.
- **Trade-offs**: Multi-symptom inquiries (e.g., battery drain during iOS update) span two classes; resolved by precedence rules and ambiguity confidence flagging.

---

### Entry 004: Intent-Aware In-Memory Okapi BM25 Retrieval Engine
- **Date**: 2026-09-12
- **Decision**: Implement a pure Python in-memory Okapi BM25 indexing and retrieval engine with intent-aware candidate ranking.
- **Alternatives Considered**: Standalone vector DBs (Qdrant, Milvus, Pinecone), dense neural embeddings (Sentence-Transformers with PyTorch).
- **Evidence & Rationale**:
  - In-memory BM25 index on historical corpus requires ~45 MB RAM and achieves ~1.2 ms per query.
  - Full 200-item evaluation batch completes retrieval in $< 1$ second vs minutes on CPU neural models.
  - High explainability: Exact lexical matches and term overlaps are directly inspectable in decision records.
- **Trade-offs**: Does not perform semantic synonym matching for rare unseen vocabulary without exact term overlap, mitigated by intent classification pre-filtering.

---

### Entry 005: Two Concrete Baselines (Trivial & Simple ML)
- **Date**: 2026-09-14
- **Decision**: Formally integrate two reproducible, deterministic baselines into the evaluation harness:
  1. **Trivial Baseline**: Majority-class intent prediction (Macro F1 = `0.0278`) + random corpus retrieval + constant routing.
  2. **Simple ML Baseline**: Supervised TF-IDF word/sub-word n-grams + Logistic Regression ($C=1.0$) trained strictly on pre-split historical conversations (Macro F1 = `0.5803`).
- **Alternatives Considered**: Single trivial baseline; untracked ad-hoc scripts.
- **Evidence & Rationale**:
  - Fulfills the assignment requirement to compare against both trivial and simple baselines.
  - The simple ML baseline trains deterministically in ~4 seconds on pre-split data with zero evaluation leakage.
- **Trade-offs**: Simple baseline relies on silver training labels, but establishes a legitimate benchmark for linear text classification.

---

### Entry 006: Continuous Probabilistic Routing Risk Score ($R(x) \in [0, 1]$)
- **Date**: 2026-09-15
- **Decision**: Replace brittle binary escalation switches with a continuous probabilistic risk score $R(x) \in [0.0, 1.0]$ aggregated via Noisy-OR:
  $$R(x) = 1.0 - \prod_{k} (1.0 - r_k)$$
  incorporating explicit safety triggers, severe failure patterns, non-English detection, intent ambiguity, retrieval evidence deficit, DM precedent, and customer frustration indicators.
- **Alternatives Considered**: Single hard threshold on classifier confidence; pure heuristic if-else cascades; black-box neural classifier.
- **Evidence & Rationale**:
  - Exposes an explicit continuous risk gradient allowing dynamic threshold tuning without changing core logic.
  - Prevents step-function discontinuities where tiny confidence changes flipped routing decisions.
- **Trade-offs**: Requires setting an explicit operational decision threshold ($\tau$) based on safety requirements.

---

### Entry 007: Safety-Oriented Threshold Selection via Empirical Threshold Sweep
- **Date**: 2026-09-15
- **Decision**: Evaluate a continuous threshold sweep ($\tau \in [0.10, 0.90]$) on the development golden set and select $\tau = 0.45$ as the default operating point.
- **Alternatives Considered**: Arbitrary $\tau = 0.50$; tuning threshold against held-out test data.
- **Evidence & Rationale**:
  - At $\tau = 0.45$, the system achieves $85.0\%$ escalation recall and $15.0\%$ false-auto-handle rate while maintaining $69.0\%$ auto-handle coverage.
  - Selection was conducted strictly on the development golden set, preserving zero-leakage test integrity.
- **Trade-offs**: Prioritizes customer safety over aggressive automation coverage.

---

### Entry 008: Grounded Generation & Citation Constraint
- **Date**: 2026-09-12
- **Decision**: Strictly condition `DraftReply` generation on retrieved precedent, requiring explicit evidence ID citations in the output schema and prohibiting unsupported claims.
- **Alternatives Considered**: Unconstrained open-ended LLM generation with post-hoc fact checking.
- **Evidence & Rationale**:
  - Constitution Principle III: Draft replies must be grounded in verified support precedent to eliminate hallucinations.
  - Yields 100% reply groundedness score on auto-handled inquiries.
- **Trade-offs**: If retrieved precedent is sparse, reply generation is suppressed in favor of human escalation.

---

### Entry 009: Dual-Mode Evaluation (LLM-as-a-Judge with Deterministic Offline Fallback)
- **Date**: 2026-09-12
- **Decision**: Implement a 1–5 rubric-based LLM groundedness judge with a deterministic lexical-overlap fallback when API keys are absent.
- **Alternatives Considered**: Human evaluation only, purely lexical BLEU/ROUGE metrics.
- **Evidence & Rationale**:
  - BLEU/ROUGE correlate poorly with factual correctness; LLM judge accurately evaluates semantic grounding.
  - Fallback ensures the pipeline never crashes in offline CI environments or without a Gemini API key.
- **Trade-offs**: LLM judge evaluation introduces minor network latency during full batch evaluation runs.

---

### Entry 010: Two-Tier Benchmark Architecture & Cryptographic Freezing
- **Date**: 2026-09-13
- **Decision**: Architect a strict two-tier evaluation framework:
  1. **Tier 1 (Development Golden Set)**: `data/processed/golden_set.jsonl` (200 hand-labelled examples curated across 8 balanced classes) used for iterative improvement.
  2. **Tier 2 (Frozen Held-Out Human Benchmark)**: `data/processed/human_benchmark_unannotated.jsonl` (200 authentic post-split conversations), cryptographically frozen with SHA-256 manifest.
- **Alternatives Considered**: Single unified benchmark; tuning directly on held-out test data.
- **Evidence & Rationale**:
  - Prevents subtle overfitting and guarantees unbiased out-of-distribution evaluation.
- **Trade-offs**: Requires disciplined separation between development tuning and held-out validation.

---

### Entry 011: Strict Pre-Run Leakage Gate
- **Date**: 2026-09-12
- **Decision**: Implement mandatory automated assertions in the evaluation harness and CLI to verify 0% conversation ID overlap between retrieval corpus and evaluation sets before any benchmark execution.
- **Alternatives Considered**: Manual spot-checks; post-hoc audit scripts.
- **Evidence & Rationale**:
  - Hard failure stops execution if contamination occurs, eliminating data leakage bugs at the root.
- **Trade-offs**: Adds a 50ms set intersection verification step during test initialization.

---

### Entry 012: Pure Python Single-Machine Architecture (<40s Runtime)
- **Date**: 2026-09-12
- **Decision**: Implement all components as lightweight, modular Python modules with standard dependencies (scikit-learn, pydantic, numpy) and zero external container dependencies.
- **Alternatives Considered**: Microservice architecture with Docker Compose, Redis, and FastAPI.
- **Evidence & Rationale**:
  - Enables full 200-sample end-to-end evaluation runs in ~37 seconds on standard developer laptops.
  - Maximizes reliability, reviewability, and reproducibility.
- **Trade-offs**: Single-node architecture; production high-throughput streaming would require message broker orchestration.

---

### Entry 013: Transparent Reporting of Held-Out Routing Safety Limitations
- **Date**: 2026-09-14
- **Decision**: Explicitly document that while retrieval and reply grounding are strong (MRR@5 = 0.851, Groundedness = 1.000), held-out human benchmark routing achieves a 62.8% False-Auto-Handle rate and is **NOT production-safe for unsupervised autonomous deployment**.
- **Alternatives Considered**: Hiding held-out test results; reporting only the development golden set headline numbers.
- **Evidence & Rationale**:
  - Truth-in-engineering: Strong retrieval and groundedness do not automatically imply safety. Sarcastic rants, subtle multi-turn complaints, and implicit failures require explicit escalation caution.
- **Trade-offs**: Transparently reveals engineering limitations rather than claiming false production readiness.

---

### Entry 014: Author-Labelled Golden Set Provenance
- **Date**: 2026-09-15
- **Decision**: Formally document that the 200-example development golden set was curated and hand-labelled by the single project author, prohibiting misleading claims of multi-annotator crowdsourced panels.
- **Alternatives Considered**: Claiming independent external panels; claiming rule-only automated labelling.
- **Evidence & Rationale**:
  - Scientific integrity and accurate provenance tracking.
- **Trade-offs**: Acknowledges single-author annotation perspective while maintaining strict validation standards.
