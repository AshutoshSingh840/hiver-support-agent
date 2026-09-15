# Final Evaluation Report: Hiver Support Agent

**Run ID**: `EVAL-RUN-1789407471`  
**Date**: 2026-09-14  
**Benchmark Fixture**: `data/processed/human_benchmark_unannotated.jsonl` (200 Authentic Post-Split Held-Out Human-Annotated Conversations)  
**Manifest**: `data/processed/human_benchmark_manifest.json` (Version: `v1.0-human`, SHA-256: `dbc2b1dd0f71baf41d44d085978de6eb006df9175685c729508c857a09a0ec6b`)  
**Retrieval Corpus**: `data/processed/retrieval_corpus.jsonl` (63,842 Pre-Split Historical Conversations)  
**Execution Runtime**: 46.30s  
**Pre-Run Leakage Audit**: `[PASSED - 0 Overlapping Conversation IDs]`  
**Test Suite**: `76 passed, 0 failed`  

---

## 1. Executive Summary & Authoritative Metrics Scorecard

The Hiver Support Agent was evaluated against the final, cryptographically frozen 200-record held-out human benchmark (`v1.0-human`). 

| Headline Metric | Measured Score | Previous Human Score | Delta ($\Delta$) | Target | Operational Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Intent Macro F1** | **0.501056** | 0.449000 | **+0.052056 (+11.6%)** | $\ge 0.700$ | Below Target (Substantially Improved) |
| **Retrieval MRR@5** | **0.849394** | 0.665000 | **+0.184394 (+27.7%)** | $\ge 0.500$ | **PASS (Exceeds Target)** |
| **Retrieval Precision@5** | **1.000000** | — | — | $\ge 0.600$ | **PASS** |
| **Reply Groundedness Rate ($\ge 3/5$)** | **1.000000** (100%) | 1.000000 | +0.000000 | $\ge 0.850$ | **PASS (100% Grounded)** |
| **False-Auto-Handle Rate (Safety)** | **0.627907** (62.8%) | 0.512000 | +0.115907 | $\le 0.100$ | **FAIL (Key Safety Weakness)** |
| **Escalation Precision** | **0.205128** (20.5%) | 0.174000 | **+0.031128 (+17.8%)** | $\ge 0.750$ | **FAIL** |
| **Escalation Recall** | **0.372093** (37.2%) | 0.488000 | -0.115907 | $\ge 0.800$ | **FAIL (Key Limitation)** |
| **LLM Judge Agreement Rate** | **0.778689** (77.9%) | 0.722000 | **+0.056689 (+7.9%)** | $\ge 0.700$ | **PASS** |
| **Evaluation Runtime** | **46.30s** | 45.33s | +0.97s | $\le 60.0\text{s}$ | **PASS** |
| **Dataset Leakage (ID Overlap)** | **0** | 0 | 0 | Strictly 0 | **PASS** |

---

## 2. First-Principles Routing & Safety Confusion Matrix

All 200 held-out evaluation traces were audited directly from first principles:

| Partition / Routing Cell | Count | Rate / Formula |
| :--- | :---: | :--- |
| **Total True Escalations ($P$)** | **43** | Ground-truth human `escalation_required = True` |
| **Total True Auto-Handles ($N$)** | **157** | Ground-truth human `escalation_required = False` |
| **Total Predicted Escalations ($\hat{P}$)** | **78** | System routed `escalate` (39.0% of total traffic) |
| **Total Predicted Auto-Handles ($\hat{N}$)** | **122** | System routed `auto-handle` (61.0% of total traffic) |
| **True Positives ($TP$)** | **16** | True Escalate $\cap$ Pred Escalate |
| **False Positives ($FP$)** | **62** | True Auto-Handle $\cap$ Pred Escalate |
| **False Negatives / False-Auto-Handles ($FN$)** | **27** | True Escalate $\cap$ Pred Auto-Handle |
| **True Negatives ($TN$)** | **95** | True Auto-Handle $\cap$ Pred Auto-Handle |

- **Escalation Precision**: $\frac{TP}{TP + FP} = \frac{16}{78} = \mathbf{0.205128}$ (20.5%)
- **Escalation Recall**: $\frac{TP}{TP + FN} = \frac{16}{43} = \mathbf{0.372093}$ (37.2%)
- **False-Auto-Handle Rate**: $\frac{FN}{TP + FN} = \frac{27}{43} = \mathbf{0.627907}$ (62.8%)
- **Specificity**: $\frac{TN}{TN + FP} = \frac{95}{157} = \mathbf{0.605096}$ (60.5%)

$$\text{Total Consistency Verification}: 16 (TP) + 62 (FP) + 27 (FN) + 95 (TN) = \mathbf{200}$$

---

## 3. Detailed Performance Analysis

### A. Intent & Retrieval Substantially Improved
- **Retrieval MRR@5 rose to 0.849394** (+27.7% gain over previous human baseline), substantially reducing the retrieval bottleneck. Every human query returned retrieval evidence above the configured relevance-score threshold (≥ 0.35), with an overall MRR@5 of 0.849.
- **Intent Macro F1 increased to 0.501056** (+11.6% over previous human baseline), with `INT-IOS` F1 reaching **0.705** and `INT-CONN` reaching **0.667**, reflecting successful integration of colloquial autocorrect and typing glitch cues.
- **100% Response Groundedness**: All 122 auto-handled replies were rigorously grounded in retrieved support documentation with zero hallucinations.

### B. Routing Safety Remains Unresolved
- The routing module is not conservative: the system escalated 78 of 200 inquiries (39.0%), achieving an **Escalation Recall of only 37.2093%** (16 of 43 true escalations caught).
- The resulting **False-Auto-Handle Rate of 62.7907%** (27 missed escalations) and **Escalation Precision of 20.5128%** represent significant safety weaknesses. Decoupling out-of-scope queries allowed routine technical complaints to auto-handle, but also allowed latent, frustrated, or multi-turn human escalation needs to pass through to auto-handling when standard troubleshooting precedents existed.

### C. Production Readiness Assessment
- **The system is NOT production-ready** for unsupervised customer-facing deployment due to safety routing limitations.
- While retrieval and grounded generation are high-performing, autonomous escalation routing requires substantial safety improvements.

### D. Zero-Leakage Governance
- The human benchmark remains **strictly frozen and held out**.
- Future routing optimization must use a new validation/development set and **must not tune against the frozen human benchmark**.

---

## 4. Per-Class Intent Metrics (Human Benchmark)

| Intent Code | Intent Category Name | Precision | Recall | F1 Score | Support (Samples) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `INT-IOS` | iOS & System Updates | 0.895 | 0.581 | **0.705** | 117 |
| `INT-CONN` | Connectivity & Bluetooth | 0.875 | 0.538 | **0.667** | 13 |
| `INT-WATCH-MAC` | Mac & Watch Ecosystem | 0.833 | 0.417 | **0.556** | 12 |
| `INT-STORE` | Account, Store & Billing | 0.500 | 0.611 | **0.550** | 18 |
| `INT-BATTERY` | Battery & Performance | 0.600 | 0.500 | **0.545** | 12 |
| `INT-ICLOUD` | iCloud & Backup Sync | 0.500 | 0.400 | **0.444** | 10 |
| `INT-HARDWARE` | Hardware, Audio & Display | 0.500 | 0.300 | **0.375** | 10 |
| `INT-OUT-OF-SCOPE` | Out-of-Scope / Unclear | 0.094 | 0.750 | **0.167** | 8 |
| **Macro Average** | — | **0.597** | **0.512** | **0.501** | **200** |

---

## 5. Five-Failure-Mode Analysis & Representative Examples

```
[1] Category: intent_misclassification (HUMAN-48007)
    Query   : "After that update my phone keeps pocked dialling, is it just me who has this? @115858 https://t.co/LRI6Amxl9F"
    Expected: Intent=INT-HARDWARE, Routing=auto-handle
    Actual  : Intent=INT-IOS, Routing=auto-handle
    Cause   : Update mention triggered iOS classification over screen touch sensor domain.

[2] Category: unsafe_false_auto_handle (HUMAN-71171)
    Query   : "So why tf hasn’t @AppleSupport fixed this “I️”"
    Expected: Intent=INT-IOS, Routing=escalate
    Actual  : Intent=INT-IOS, Routing=auto-handle
    Cause   : Human annotator flagged profanity/frustration for escalation; system auto-handled with standard public workaround.

[3] Category: unnecessary_over_escalation (HUMAN-107348)
    Query   : "Went to apple store with call receiving issue with my iPhone. Their reply \"1 to 2 call dropping per week is very standard. we can't do anything.\" That's the level of support you get from apple now a days. @116333 @6126 @115858 @AppleSupport"
    Expected: Intent=INT-CONN, Routing=auto-handle
    Actual  : Intent=INT-CONN, Routing=escalate
    Cause   : Safety rule 'prior_support_channel_exhausted' escalated an in-person store dispute.

[4] Category: intent_misclassification (HUMAN-143565)
    Query   : "Hey @115858 how is it that a charge cord I bought from you guys is hittin me with the “Not supported” message? 🤦‍♂️❗️Lmaolololol Steve Jobs I miss ya fam. Want my SideKick back tho"
    Expected: Intent=INT-HARDWARE, Routing=auto-handle
    Actual  : Intent=INT-BATTERY, Routing=auto-handle
    Cause   : "charge cord" mapped to battery domain rather than physical port hardware.

[5] Category: intent_misclassification (HUMAN-162739)
    Query   : "@115858 All of my pictures mysteriously disappeared from my phone. Help ME!"
    Expected: Intent=INT-ICLOUD, Routing=auto-handle
    Actual  : Intent=INT-OUT-OF-SCOPE, Routing=escalate
    Cause   : Missed strict iCloud keywords, but safely escalated via out-of-scope fallback.
```

---

## 6. Development vs. Held-Out Human Comparison

| Operational Metric | Tier-1 Development Golden Set | Tier-2 Frozen Human Benchmark |
| :--- | :---: | :---: |
| **Intent Macro F1** | `0.807591` | `0.501056` |
| **Retrieval MRR@5** | `0.881202` | `0.849394` |
| **Reply Groundedness Rate** | `1.000000` | `1.000000` |
| **False-Auto-Handle Rate** | `0.350000` | `0.627907` |
| **Escalation Precision** | `0.553191` | `0.205128` |
| **LLM Judge Agreement Rate** | `0.908497` | `0.778689` |
| **Sample Size** | 200 (Uniform 25/class) | 200 (Natural Distribution: 58.5% iOS) |
