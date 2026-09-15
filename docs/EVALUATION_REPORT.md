# Empirical Evaluation Report: Hiver Support Agent

**Run ID**: `EVAL-RUN-1789492884`  
**Evaluation Scope**: AppleSupport Customer Inquiries (Customer Support on Twitter Dataset)  
**Development Golden Set**: `data/processed/golden_set.jsonl` (200 hand-labelled examples, balanced across 8 taxonomy classes)  
**Frozen Held-Out Human Benchmark**: `data/processed/human_benchmark_unannotated.jsonl` (200 authentic post-split conversations, Manifest SHA-256: `dbc2b1dd0f71baf41d44d085978de6eb006df9175685c729508c857a09a0ec6b`)  
**Retrieval Corpus**: `data/processed/retrieval_corpus.jsonl` (29,591 pre-split historical conversations prior to 2017-11-01)  
**Pre-Run Leakage Audit**: `[PASSED — Zero conversation ID overlap between retrieval corpus and all evaluation sets]`  
**Execution Runtime**: ~45.2s  
**Test Suite**: `85 passed, 0 failed`

---

## 1. Executive Summary & Baselines Comparison

Three systems are compared against the 200-example development golden set:

1. **Trivial Baseline**: Majority-class intent prediction — always predicts the most frequent label.
2. **Simple ML Baseline**: TF-IDF (1–2 word n-grams, sublinear TF) + Logistic Regression (C=1.0).
3. **Hiver Support Agent**: Rule-based 8-class taxonomy with calibrated confidence scoring, intent-aware BM25 retrieval, and continuous probabilistic routing risk score.

### Simple Baseline Evaluation Methodology — Comparability Caveat

> [!IMPORTANT]
> **The simple baseline and the Hiver Agent scores are NOT a perfectly apples-to-apples comparison.**
>
> **Why not**: The retrieval corpus (29,591 records) has **no ground-truth intent annotations** and cannot be used as a training set without calling the production classifier — which would be circular (the baseline would imitate the classifier rather than learn from human labels). The only available labelled data is the 200-example development golden set.
>
> **What was done**: Deterministic **stratified 5-fold cross-validation** (`StratifiedKFold`, `random_state=42`, no shuffle) over the 200 golden examples. Each fold trains on 160 examples and evaluates on 40. Mean ± std is reported across all 5 folds. The production `IntentClassifier` is **never called** to supply training labels.
>
> **Scope difference**:
> - Simple Baseline: mean ± std over five 40-example held-out folds (160-example training sets)
> - Hiver Agent: scored on all 200 golden examples (no training step)
>
> The baseline is a legitimate simple reference point — the only valid one given the available labelled data — not a direct competitor on the same test set. The gap between 0.5847 and 0.8027 is real but may be slightly inflated by this scope difference.

### Intent Classification 3-Way Comparison

| Model | Intent Macro F1 | Eval Scope | Label Source |
| :--- | :---: | :--- | :--- |
| **[1] Trivial Baseline (Majority Class)** | `0.0278` | All 200 golden examples | N/A |
| **[2] Simple ML Baseline (TF-IDF + LogReg)** | `0.5847 ± 0.0493` (5-fold CV) | 5 × 40-example held-out folds | Human `true_intent` labels only |
| **[3] Hiver Support Agent** | **`0.8027`** | All 200 golden examples | N/A (rule-based, no training) |

*The production classifier's predictions are never used as a training or scoring signal for either baseline.*

---

## 2. Headline Metrics Scorecard (Development Golden Set, n=200, τ=0.45)

| Metric | Score | Baseline | Target | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Intent Macro F1** | **`0.8027`** | `0.0278` | ≥ 0.700 | **PASS** |
| **Retrieval MRR@5** | **`0.8783`** | `0.0000` | ≥ 0.500 | **PASS** |
| **Reply Groundedness Rate (≥ 3/5)** | **`1.0000`** | `0.5000` | ≥ 0.850 | **PASS** |
| **False-Auto-Handle Rate (Safety)** | **`0.3500`** | `0.5000` | ≤ 0.100 | **FAIL** |
| **Escalation Precision** | **`0.7430`** | `0.5000` | ≥ 0.750 | **FAIL (−0.007)** |

> [!WARNING]
> FAH=0.350 at the default τ=0.45 means 14 of 40 true escalation cases were silently auto-handled. At τ=0.10, FAH falls to 0.150. **No threshold achieves FAH ≤ 0.10 on this dataset.** See Section 3 for the full threshold sweep and threshold selection rationale.

### Per-Class Intent F1 (Dev Golden Set)

| Intent Code | Category | F1 | Precision | Recall | Support |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `INT-BATTERY` | Battery & Performance Drain | **`0.913`** | 0.875 | 0.955 | 22 |
| `INT-ICLOUD` | iCloud & Backup Sync | **`0.913`** | 0.875 | 0.955 | 22 |
| `INT-WATCH-MAC` | Mac & Watch Ecosystem | **`0.840`** | 0.955 | 0.750 | 28 |
| `INT-STORE` | Account, Store & Billing | **`0.830`** | 0.786 | 0.880 | 25 |
| `INT-CONN` | Connectivity & Bluetooth | **`0.809`** | 0.826 | 0.792 | 24 |
| `INT-HARDWARE` | Hardware, Audio & Display | **`0.800`** | 0.815 | 0.786 | 28 |
| `INT-IOS` | iOS & System Updates | **`0.721`** | 0.647 | 0.815 | 27 |
| `INT-OUT-OF-SCOPE` | Out-of-Scope / Non-Support | **`0.596`** | 0.700 | 0.519 | 27 |
| **Macro Average** | — | **`0.803`** | **`0.810`** | **`0.806`** | **200** |

---

## 3. Continuous Routing Risk Score & Threshold Sweep

The router uses a Noisy-OR aggregator $R(x) = 1 - \prod_k (1 - r_k)$ over 8 risk channels in two tiers:

**Hard / Safety channels** (r = 0.72–0.92): sensitive keywords, data loss / bricking, non-English, out-of-scope.  
**Soft / Gradient channels** (r = 0.02–0.35): intent confidence deficit, retrieval quality deficit, DM-only precedent, customer frustration.

### Threshold Sweep (Development Golden Set, n=200)

| τ | FAH↓ | Esc Precision↑ | Esc Recall↑ | Coverage↑ | Esc F1 | # Escalated |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0.10`** | **0.150** | 0.548 | **0.850** | 69.0% | 0.667 | 62 |
| `0.20` | 0.200 | 0.696 | 0.800 | 77.0% | **0.744** | 46 |
| `0.30` | 0.350 | 0.743 | 0.650 | 82.5% | 0.693 | 35 |
| `0.35` | 0.350 | 0.743 | 0.650 | 82.5% | 0.693 | 35 |
| `0.40` | 0.350 | 0.743 | 0.650 | 82.5% | 0.693 | 35 |
| **`0.45` (config default)** | 0.350 | 0.743 | 0.650 | 82.5% | 0.693 | 35 |
| `0.50` | 0.350 | 0.743 | 0.650 | 82.5% | 0.693 | 35 |
| `0.55` | 0.350 | 0.743 | 0.650 | 82.5% | 0.693 | 35 |
| `0.60` | 0.350 | 0.743 | 0.650 | 82.5% | 0.693 | 35 |
| `0.70` | 0.350 | 0.743 | 0.650 | 82.5% | 0.693 | 35 |
| `0.80` | 0.650 | 0.824 | 0.350 | 91.5% | 0.491 | 17 |
| `0.90` | 0.675 | 0.867 | 0.325 | 92.5% | 0.473 | 15 |

### Explaining the 0.30–0.70 Plateau

The flat band from τ=0.30 to τ=0.70 is a real property of the score distribution — not a calibration bug. Risk scores cluster in two regions:

- **~83% of queries score ≤ 0.27**: Routine queries with no hard triggers. Soft-signal Noisy-OR combinations max out at ~0.27.
- **~17% of queries score ≥ 0.72**: A hard trigger fires (sensitive keyword, data loss pattern, non-English, out-of-scope with no evidence).
- **No scores fall in [0.28, 0.72]**: This gap is inherent — no single soft-signal combination reaches 0.28, and any hard trigger immediately pushes the score to ≥ 0.72. Thresholds in this range are equally good or bad.

### Threshold Selection — Safety-Oriented Criterion

> [!IMPORTANT]
> **Recommended operating threshold: τ = 0.10** under a safety-first criterion.
>
> **Criterion**: Minimize false-auto-handle rate (FAH) subject to ≥ 50% auto-handle coverage. FAH is the primary safety metric: a false auto-handle means a customer who needed a human received only an automated response.
>
> | τ | FAH | Coverage | Verdict |
> |---|-----|---------|---------|
> | 0.10 | **0.150** | 69.0% | **Recommended** — lowest FAH, adequate coverage |
> | 0.20 | 0.200 | 77.0% | Acceptable secondary choice; FAH up 33% for +8% coverage |
> | 0.30–0.70 | 0.350 | 82.5% | FAH doubles vs 0.10; not acceptable under safety-first |
> | 0.80 | 0.650 | 91.5% | Catastrophic FAH |
>
> **τ = 0.20 is NOT selected because it maximises escalation F1** (0.744). F1 is a symmetric metric that treats missed escalations and unnecessary escalations equally. In a customer support safety context, missed escalations (false auto-handles) carry higher cost.
>
> **The config default remains τ=0.45** (plateau zone) for backward compatibility. Any fresh deployment should use τ=0.10 or τ=0.20 per the above analysis.
>
> **No threshold satisfies FAH ≤ 0.10 on this dataset.** The minimum achievable FAH on the 200-example development set is 0.150 at τ=0.10. On the frozen held-out benchmark FAH is 0.628 irrespective of threshold tuning here.

---

## 4. Confidence Calibration

| Confidence Bucket | Sample Count | Observed Accuracy | Mean Confidence |
| :---: | :---: | :---: | :---: |
| `[0.00, 0.50)` | 0 | — | — |
| `[0.50, 0.70)` | 17 | **47.06%** | 0.5000 |
| `[0.70, 0.85)` | 22 | **63.64%** | 0.8000 |
| `[0.85, 1.00)` | 161 | **85.71%** | 0.9408 |

Accuracy increases monotonically with confidence. The classifier floor of 0.50 (set on ambiguous compound queries) is well-calibrated as the lowest-accuracy bucket.

---

## 5. Frozen Held-Out Human Benchmark (Read-Only)

These metrics were measured on the frozen benchmark **after all development tuning was complete**. The benchmark was **never used** for threshold selection, prompt optimization, or any parameter tuning.

| Metric | Frozen Human Benchmark | Dev Golden Set | Notes |
| :--- | :---: | :---: | :--- |
| **Intent Macro F1** | `0.5025` | `0.8027` | Real tweets: compound symptoms, unconstrained slang |
| **Retrieval MRR@5** | `0.8510` | `0.8783` | Retrieval robust across both sets |
| **Reply Groundedness Rate** | `1.0000` | `1.0000` | No hallucinations on either set |
| **False-Auto-Handle Rate** | **`0.6279`** | `0.3500` (τ=0.45) | **Unsafe for unsupervised deployment** |
| **Escalation Precision** | `0.1765` | `0.7430` | Over-escalates common queries |
| **Escalation Recall** | `0.3721` | `0.6500` | Misses sarcasm / rhetorical rants |

> [!WARNING]
> **Production Safety Notice**: FAH of 62.8% on the held-out benchmark means 63 of every 100 true escalation cases are silently auto-handled. **The system is NOT production-ready for unsupervised autonomous deployment.** Open-ended social tweets contain sarcasm, implicit failures, and frustration that keyword/pattern heuristics cannot reliably detect without multi-turn conversational context.

---

## 6. Top 5 Failure Modes (Real Examples & Hypotheses)

### FM-1: Update Mention Masking Hardware Defect (`intent_misclassification`)
- **Example**: *"After that update my phone keeps pocket dialling, is it just me?"*
- **Expected**: `INT-HARDWARE` | **Predicted**: `INT-IOS`
- **Hypothesis**: "that update" as temporal context triggers `INT-IOS` tokens before the hardware symptom ("pocket dial") accrues enough weight.
- **Remediation**: Dependency parsing to separate temporal markers from functional symptoms.

### FM-2: Rhetorical Rant Missed Escalation (`unsafe_false_auto_handle`)
- **Example**: *"So why tf hasn't @AppleSupport fixed this 'I️'"*
- **Expected**: `escalate` | **Predicted**: `INT-IOS`, `auto-handle`
- **Hypothesis**: Autocorrect bug triggers strong retrieval evidence for a standard workaround, but the customer is expressing brand dissatisfaction.
- **Remediation**: Sentiment classifier to escalate rhetorical complaints regardless of precedent availability.

### FM-3: Prior Store Visit Over-Escalation (`unnecessary_over_escalation`)
- **Example**: *"Went to apple store with call issue. They said '1-2 drops/week is standard.'"*
- **Expected**: `INT-CONN`, `auto-handle` | **Predicted**: `INT-CONN`, `escalate`
- **Hypothesis**: `prior_support_channel_exhausted` fires because of the Apple Store mention, even though the query is a routine connectivity complaint.
- **Remediation**: Distinguish store visit mentions from exhausted technical resolution attempts.

### FM-4: Accessory Authentication Mis-classified as Battery (`intent_misclassification`)
- **Example**: *"How is it that a charge cord from you hits me with 'Not supported'?"*
- **Expected**: `INT-HARDWARE` | **Predicted**: `INT-BATTERY` (0.78)
- **Hypothesis**: "charge cord" triggers battery/charging tokens; accessory authentication error tokens missing from `INT-HARDWARE`.
- **Remediation**: Add "accessory not supported", "unsupported cable" tokens to `INT-HARDWARE`.

### FM-5: Non-Latin Script Out-of-Scope (`intent_misclassification`)
- **Example**: `"11.1?????Twitter??????????????????20????????"` (Japanese/mixed)
- **Expected**: `INT-IOS` | **Predicted**: `INT-OUT-OF-SCOPE`
- **Hypothesis**: Non-ASCII tokens defeat all keyword matches; default falls to out-of-scope.
- **Remediation**: Unicode-normalized transliteration or language-aware token matching.

---

## 7. What is Misleading About My Headline Number?

An engineer reviewing only dev golden set metrics might conclude this system is near-production-ready:
- Intent Macro F1 = **80.3%**
- Retrieval MRR@5 = **87.8%**
- Reply Groundedness = **100%**

### Why These Numbers Are Deceptive

1. **Balanced golden set vs. natural long-tail**: The dev set is artificially balanced (25/class). In authentic Twitter traffic, 58.5% of queries are iOS complaints. Macro F1 conceals degradation on real phrasing.

2. **Retrieval groundedness ≠ routing safety**: 100% groundedness means replies cite retrieved text. But sending a grounded troubleshooting macro to an enraged user demanding accountability is worse than escalating.

3. **Single-turn blindness**: The agent sees only the root tweet. Customer support threads involve DM handoffs, prior store visits, and repeat contacts that are invisible in one 280-character post.

4. **FAH gap**: Dev FAH = 15.0% at τ=0.10 (or 35.0% at τ=0.45). Held-out FAH = 62.8%. Keyword heuristics miss latent escalation cues at scale.

5. **Baseline comparability caveat**: The simple baseline CV score (0.5847 ± 0.0493) is estimated on 5 × 40-example held-out folds; the agent score (0.8027) is on all 200. The gap is real, but the comparison is not perfectly controlled.

---

## 8. What I Would Do With One More Week

1. **Few-shot LLM reranker**: Gemini 2.5 Flash for borderline tickets (confidence 0.50–0.80) to resolve compound-domain ambiguities without adding latency on clear cases.

2. **Multi-turn conversational memory**: Reconstruct prior 3 customer turns and feed into routing. Repeat frustrated replies → auto-escalate.

3. **Active learning dashboard**: FastAPI review queue for tickets in the uncertainty band (R(x) ∈ [0.20, 0.45]). Human triage annotations feed back as silver training data.

4. **Multi-modal OCR**: Over 30% of Apple Support tweets include screenshots. OCR of error dialogs ("Account Disabled", "Cannot Verify Server") provides immediate hard-trigger coverage.

---

## 9. Golden Set Provenance

- **Author-curated benchmark**: 200 examples manually sampled, audited, and labelled by the project author.
- **Sampling**: Uniform stratified sampling (25/class) from post-split conversations (≥ 2017-11-01) with verified non-brand roots.
- **Labelling**: `true_intent` by functional problem domain; `true_escalation` by safety/sensitivity criteria. No automated labelling, crowd-sourcing, or external panels.

---

## 10. Final System Status

| Component | Dev Golden Set | Frozen Human Benchmark |
| :--- | :---: | :---: |
| **Intent Macro F1** | `0.8027` | `0.5025` |
| **Retrieval MRR@5** | `0.8783` | `0.8510` |
| **Reply Groundedness** | `1.0000` | `1.0000` |
| **False-Auto-Handle Rate** | `0.350` (τ=0.45) / `0.150` (τ=0.10) | `0.6279` |
| **Test Suite** | **85 passed, 0 failed** | — |

**Baseline comparison (intent only):**

| Model | F1 | Eval Scope |
| :--- | :---: | :--- |
| Trivial (Majority Class) | `0.0278` | All 200 golden examples |
| Simple ML (TF-IDF + LogReg, 5-fold CV) | `0.5847 ± 0.0493` | 5 × 40-example held-out folds |
| Hiver Agent (Calibrated Keyword) | `0.8027` | All 200 golden examples |

*Baseline and agent scores are not perfectly apples-to-apples — see Section 1 for full methodology.*

**Recommended operating threshold**: τ=0.10 (FAH=0.150, Coverage=69%) under safety-first criterion.  
**No threshold satisfies FAH ≤ 0.10 on this dataset.**  
**Production readiness: NOT CLAIMED.**
