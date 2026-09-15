# Feature Specification: Hiver Support Agent

**Feature Branch**: `001-support-agent`

**Created**: 2026-09-12

**Status**: Draft — Awaiting Review

**Input**: Evaluation-first AI customer-support agent; brand selected from
evidence in `docs/DATASET_PROFILE.md`; assignment prototype scope.

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Dataset Ingestion & Conversation Reconstruction (Priority: P1)

A data engineer ingests the raw Customer Support on Twitter dataset
(`data/raw/twcs.csv`, 2,811,774 rows) and produces a clean, versioned
corpus of AppleSupport conversations suitable for retrieval indexing and
evaluation benchmarking.

**Why this priority**: No downstream component (retrieval, classification,
reply generation, evaluation) can function without a correct, leakage-free
conversation corpus. Everything else depends on this step being deterministic
and auditable.

**Independent Test**: Can be tested end-to-end by running the ingestion
pipeline against `data/raw/twcs.csv` and verifying the output corpus counts,
schema, and split manifests, without any AI or LLM components being present.

**Acceptance Scenarios**:

1. **Given** `data/raw/twcs.csv` exists and has not been modified,
   **When** the ingestion pipeline is executed,
   **Then** the pipeline emits exactly 74,571 AppleSupport customer conversations,
   logs the input file hash, output row counts, and runtime, and produces
   identical outputs on repeated runs with the same inputs.

2. **Given** the ingestion pipeline has completed,
   **When** the split manifest is inspected,
   **Then** zero conversation IDs appear in both the development/retrieval
   partition and the golden evaluation partition, and the check is enforced
   automatically before any evaluation run is permitted.

3. **Given** a conversation thread in the raw data has a broken forward
   reference (the `response_tweet_id` pointer targets a tweet not present
   in the dataset),
   **When** the ingestion pipeline encounters that thread,
   **Then** the thread is excluded from the golden evaluation corpus and
   the exclusion is logged with the root tweet ID and reason.

4. **Given** a tweet's `response_tweet_id` contains multiple comma-separated
   values,
   **When** the pipeline reconstructs conversation threads,
   **Then** all valid child IDs are traversed and the thread is correctly
   assembled using a breadth-first traversal from the root tweet outward.

---

### User Story 2 — Intent Classification (Priority: P1)

A support operator wants the system to automatically determine the type of
customer issue from an incoming AppleSupport conversation so that routing
and retrieval can be precisely targeted.

**Why this priority**: Intent classification is the first AI decision in
the pipeline and determines the quality of all downstream steps. Without
a working, evaluated classifier, retrieval and reply generation cannot
be meaningfully grounded.

**Independent Test**: Can be tested by presenting the classifier with raw
conversation text and verifying that it returns one of the defined intent
labels plus a calibrated confidence score, without requiring any retrieval
or generation components.

**Acceptance Scenarios**:

1. **Given** an incoming customer conversation expressing an iPhone
   software malfunction (e.g., app crashing, iOS update failure),
   **When** the intent classifier is invoked,
   **Then** the system returns a single intent label from the approved
   taxonomy, a confidence score in [0, 1], and at least one excerpt from
   the conversation that supports the classification.

2. **Given** an incoming conversation whose content spans two intent
   categories with roughly equal signal (ambiguous query),
   **When** the intent classifier is invoked,
   **Then** the system returns the most probable label, marks confidence
   below the defined ambiguity threshold, and flags the case for human
   review rather than auto-handling.

3. **Given** a message with no recognisable Apple support intent
   (e.g., a general complaint with no actionable support need),
   **When** the intent classifier is invoked,
   **Then** the system returns the "Out-of-Scope / Unclear" label and
   the case is immediately routed to escalation.

4. **Given** the evaluation harness runs on the golden evaluation set
   (200 rule-assisted curated evaluation benchmark examples),
   **When** intent classification predictions are compared to ground-truth
   labels,
   **Then** macro F1 is computed per intent class and reported alongside
   a per-class confusion matrix.

---

### User Story 3 — Retrieval of Historically Resolved Conversations (Priority: P2)

A support operator wants the system to retrieve the most relevant
historically resolved AppleSupport conversations from the retrieval corpus
so that the generated response is grounded in real precedent.

**Why this priority**: Grounded retrieval is what separates this system
from a generic generative response and is required by Constitution
Principle III. It can be built and evaluated independently of reply
generation.

**Independent Test**: Can be tested by submitting a customer query and
verifying that the returned conversations are semantically relevant to
the query intent, ranked by relevance score, and accompanied by the
evidence metadata required by the constitution.

**Acceptance Scenarios**:

1. **Given** a customer query classified as "iCloud Sync Error",
   **When** the retrieval module is invoked,
   **Then** the system returns the top-k (k ≤ 10) historically resolved
   AppleSupport conversations most relevant to iCloud sync errors, each
   accompanied by a relevance score and the conversation's original
   resolution text.

2. **Given** k retrieved conversations are returned,
   **When** their metadata is inspected,
   **Then** each item contains: the source conversation ID, the full
   conversation turn sequence, the relevance/similarity score, the
   temporal range of the conversation, and the assigned intent label.

3. **Given** no historically resolved conversation exceeds the minimum
   relevance threshold for the query,
   **When** the retrieval module is invoked,
   **Then** the system returns an empty result set and signals "no
   sufficient evidence" rather than returning low-relevance conversations
   that would ground a hallucinated reply.

4. **Given** the golden evaluation set is used to assess retrieval quality,
   **When** the retrieval module is evaluated at k = 5,
   **Then** Mean Reciprocal Rank (MRR@5) and Precision@5 are computed
   and reported against the trivial baseline (random retrieval).

---

### User Story 4 — Grounded Draft Reply Generation (Priority: P2)

A support operator reviews an AI-generated draft reply to a customer's
AppleSupport inquiry and can see exactly which retrieved conversations
and evidence fragments were used to produce the draft.

**Why this priority**: Reply generation is the primary user-facing output
of the system. It must be grounded (Constitution Principle III) and
auditable. It depends on intent classification (P1) and retrieval (P2)
being functional.

**Independent Test**: Can be tested by providing the generator with a
fixed intent label, a fixed set of retrieved conversations, and a fixed
customer query, and verifying that the draft reply cites the supplied
evidence and does not introduce unsupported claims.

**Acceptance Scenarios**:

1. **Given** a customer query with classified intent "Battery Drain" and
   three retrieved historically resolved conversations as evidence,
   **When** the reply generator is invoked,
   **Then** the system produces a draft reply that references at least one
   of the three supplied conversations and does not introduce factual
   claims absent from the evidence set.

2. **Given** a generated draft reply,
   **When** its metadata is inspected,
   **Then** the reply record contains: the generated text, the IDs of all
   retrieved conversations used as evidence, the relevance scores of those
   conversations, the intent label, the escalation decision, and the
   escalation rationale.

3. **Given** the reply generator is presented with evidence that contains
   only DM-escalation responses (the agent in the historical conversation
   redirected to private messaging),
   **When** a draft reply is generated,
   **Then** the system generates a response that similarly recommends
   escalation to DM rather than generating a fabricated resolution.

4. **Given** the golden evaluation set is used to assess reply quality,
   **When** an LLM-as-judge scoring prompt is applied to each generated
   draft against the reference reply,
   **Then** the judge returns a 1–5 quality score and the system reports
   mean score, score distribution, and the agreement rate between the
   judge score and reference rating (reported as percentage agreement or
   Cohen's kappa).

---

### User Story 5 — Automated Handling vs. Escalation Decision (Priority: P2)

A support operations lead wants the system to decide, with explicit
reasoning, whether each incoming case can be automatically handled or
should be escalated to a human agent, and to have that threshold be
measurable and adjustable.

**Why this priority**: The escalation decision is the primary safety
gate of the system (Constitution Principle IV). A miscalibrated threshold
risks either ignoring customers or generating inappropriate auto-responses.

**Independent Test**: Can be tested by presenting the escalation router
with a known set of cases across the confidence spectrum and verifying
that the routing decisions match expected escalation behaviour, with the
threshold adjustable without code changes.

**Acceptance Scenarios**:

1. **Given** a customer query where the intent classifier returns a
   confidence score above the configured auto-handle threshold and the
   retrieved evidence is sufficient,
   **When** the escalation router is invoked,
   **Then** the system records an "auto-handle" decision, includes the
   confidence score and the retrieved evidence IDs in the decision record,
   and produces the draft reply.

2. **Given** a customer query where intent confidence falls below the
   configured escalation threshold, or the query matches a known escalation
   trigger (account security, legal, safety),
   **When** the escalation router is invoked,
   **Then** the system records an "escalate" decision with the specific
   reason (low confidence, trigger match, or insufficient evidence) and
   does not produce an automated reply.

3. **Given** the escalation threshold is adjusted to a stricter value
   (higher confidence required for auto-handle),
   **When** the evaluation harness is re-run on the golden evaluation set,
   **Then** the system reports updated escalation precision and recall
   against curated ground-truth escalation decisions, demonstrating that the
   threshold change had the expected directional effect on both metrics.

4. **Given** a case whose correct disposition is escalation (as labelled
   in the golden set) but the system auto-handles it,
   **When** the evaluation harness reports results,
   **Then** the false-auto-handle rate (unsafe auto-handle / total escalation
   cases) is explicitly reported as a primary safety metric, separate from
   overall accuracy.

---

### User Story 6 — Evaluation Harness and Golden Set (Priority: P1)

An evaluator runs the complete automated evaluation pipeline on the
150–250 example golden evaluation set and receives a structured report
covering all headline metrics, baseline comparisons, failure mode analysis,
and LLM-judge agreement statistics.

**Why this priority**: Per Constitution Principle I, the evaluation harness
must exist and pass before any AI-behavior tuning begins. This is the
project's primary quality gate and must be built before retrieval and
generation are optimised.

**Independent Test**: Can be tested independently of the full agent
pipeline by running stub predictions through the harness and verifying
that all metric computations, baseline comparisons, and report generation
work correctly before any AI components are connected.

**Acceptance Scenarios**:

1. **Given** the complete evaluation pipeline is invoked with a single
   command,
   **When** it runs against the golden evaluation set,
   **Then** it completes in under 15 minutes on a standard development
   machine and requires no manual intervention.

2. **Given** the evaluation pipeline completes,
   **When** the output report is inspected,
   **Then** the report contains: intent classification macro F1 per class,
   retrieval MRR@5 and Precision@5, reply quality mean LLM-judge score,
   escalation precision/recall, false-auto-handle rate, baseline comparison
   for all metrics (random baseline and majority-class baseline), and the
   number of evaluation examples per metric.

3. **Given** evaluation results are produced,
   **When** the report's failure mode section is read,
   **Then** at least five distinct failure categories are documented with
   real example cases drawn from the golden evaluation set, each with the
   full decision trace (intent label, retrieved conversations, escalation
   decision, generated draft, and judge score).

4. **Given** the golden evaluation set is constructed,
   **When** an automated leakage check is run,
   **Then** the check reports zero overlap between conversation IDs in
   the golden set and conversation IDs in the retrieval corpus or
   development data, and causes a hard failure if any overlap is detected.

---

### Edge Cases

- What happens when a customer's tweet references another user who is not
  part of the Apple support conversation thread?
- How does the system handle a customer tweet that consists entirely of an
  image URL with no text content?
- What is the behaviour when `response_tweet_id` contains 1,755 comma-separated
  child IDs (observed maximum fan-out in dataset)?
- What is the behaviour when the classified intent is valid but the entire
  retrieval corpus contains zero resolved conversations matching that intent?
- How does the system handle HTML-encoded characters (`&gt;`, `&lt;`, `&amp;`)
  that are present in 100% of raw tweet text records?
- What happens when the same conversation thread contains replies from multiple
  different brand accounts (branching multi-brand response)?
- How does the system behave when an incoming conversation is a continuation
  of a DM escalation (no public text to analyse)?
- What happens when the LLM-as-judge service is unavailable during an
  evaluation run — does the harness fail hard or skip that metric?
- How does the system handle conversations where the `created_at` timestamp
  predates 2013 (sparse early records observed in the dataset)?

---

## Requirements *(mandatory)*

### Functional Requirements

#### FR-D: Data Pipeline

- **FR-D-001**: The system MUST ingest `data/raw/twcs.csv` without
  modifying the source file. All processing MUST write to derived artifact
  paths only.

- **FR-D-002**: The ingestion pipeline MUST reconstruct conversation trees
  from the raw tweet graph using the following deterministic algorithm:
  (a) identify thread roots as tweets where `in_response_to_tweet_id` is
  null AND `inbound = True`; (b) traverse child nodes using
  `response_tweet_id` (handling comma-separated values), including only
  children whose `tweet_id` exists in the dataset; (c) attribute each
  conversation to the brand whose tweet appears first among the root's
  direct children.

- **FR-D-003**: The pipeline MUST filter to AppleSupport conversations:
  all threads where at least one direct child of the customer root is
  authored by `AppleSupport`.

- **FR-D-004**: The pipeline MUST produce a serialised conversation
  format containing, for each conversation: a unique `conversation_id`
  (root tweet ID), an ordered list of dialogue turns each with speaker
  role (`customer` or `agent`), tweet ID, text, and timestamp, plus the
  assigned intent label (once classified), the resolution outcome, and
  the escalation label.

- **FR-D-005**: The pipeline MUST normalise tweet text by decoding HTML
  entities (`&amp;`, `&gt;`, `&lt;`) and preserving anonymised customer
  tokens (e.g., `@115854`) to maintain realistic entity boundaries.

- **FR-D-006**: The pipeline MUST enforce a leakage-free split at the
  conversation level (not the tweet level), producing a development/
  retrieval partition and a golden evaluation partition with a documented
  splitting criterion. A splitting audit MUST verify zero conversation-level
  overlap before any evaluation run.

- **FR-D-007**: The pipeline MUST emit a processing log for each run
  containing: input file hash, output conversation counts per partition,
  excluded thread count and reason codes, and runtime. All random operations
  (sampling, shuffling) MUST use explicit, documented seeds.

- **FR-D-008**: Threads with broken backward references (parent tweet ID
  not present in dataset; 3,862 known cases) MUST be excluded from the
  golden evaluation corpus and flagged in the processing log.

#### FR-I: Intent Classification

- **FR-I-001**: The system MUST classify every incoming customer
  conversation into exactly one label from a small, predefined, exhaustive
  intent taxonomy. The taxonomy MUST include an explicit "Out-of-Scope /
  Unclear" catch-all class.

- **FR-I-002**: The intent taxonomy MUST be defined and documented before
  any classification model or prompt is trained, fine-tuned, or evaluated.
  The taxonomy MUST be derived from evidence in the AppleSupport conversation
  corpus (discovered patterns), not defined arbitrarily.

- **FR-I-003**: The system MUST return a calibrated confidence score in
  [0, 1] for every intent classification output. Confidence calibration
  MUST be reported as part of the evaluation (e.g., reliability diagram or
  Expected Calibration Error).

- **FR-I-004**: The intent classifier MUST produce a structured decision
  record for every prediction containing: the input text excerpt, the
  predicted label, the confidence score, and the supporting evidence
  excerpt(s) from the conversation.

- **FR-I-005**: Intent classification MUST be independently testable
  without requiring the retrieval or reply-generation components to
  be present.

- **FR-I-006**: A majority-class baseline (always predict the most
  frequent intent label in the training set) MUST be included in all
  intent classification evaluations.

#### FR-R: Retrieval

- **FR-R-001**: Given an incoming conversation and its classified intent,
  the system MUST retrieve the top-k (configurable, default k = 5,
  maximum k = 10) historically resolved AppleSupport conversations most
  relevant to the query from the retrieval corpus.

- **FR-R-002**: Each retrieved conversation MUST be accompanied by a
  structured evidence record containing: the source conversation ID, the
  full conversation turn sequence, the relevance score, the intent label,
  and the temporal range of the original conversation.

- **FR-R-003**: If no retrieved conversation exceeds a configurable
  minimum relevance threshold, the system MUST return an empty result
  set with a "no sufficient evidence" signal rather than returning
  low-relevance results.

- **FR-R-004**: The retrieval corpus MUST contain only conversations from
  the development/retrieval partition; conversations from the golden
  evaluation partition MUST be excluded.

- **FR-R-005**: A random-retrieval baseline MUST be included in all
  retrieval evaluations for calibration.

- **FR-R-006**: The retrieval module MUST be independently testable
  without requiring the reply-generation component.

#### FR-G: Grounded Reply Generation

- **FR-G-001**: The system MUST generate a draft support reply for each
  auto-handled case, grounded in the retrieved historical conversations.
  The reply MUST NOT introduce factual claims, product policies, or
  procedural steps absent from the retrieved evidence set.

- **FR-G-002**: Every generated draft reply MUST be accompanied by
  structured metadata containing: the generated text, the IDs of all
  retrieved conversations used as evidence, the relevance scores, the
  intent label, the escalation decision, and the escalation rationale.

- **FR-G-003**: When the retrieved evidence consists entirely of
  DM-escalation responses, the generated reply MUST recommend escalation
  to a private channel rather than fabricate a resolution.

- **FR-G-004**: The reply generator MUST be independently testable by
  supplying a fixed intent label, fixed retrieved conversations, and
  a fixed customer query; the output MUST be deterministic given a
  fixed random seed or temperature setting.

#### FR-E: Escalation Router

- **FR-E-001**: For every processed case, the system MUST produce a
  binary routing decision: `auto-handle` or `escalate`.

- **FR-E-002**: The system MUST escalate when any of the following
  conditions is met: (a) intent confidence falls below the configured
  threshold; (b) the query matches a known escalation trigger category
  (account security, legal, safety, hardware fault requiring physical
  inspection); (c) retrieved evidence is empty (no sufficient evidence
  signal from retrieval); (d) the classified intent is "Out-of-Scope /
  Unclear".

- **FR-E-003**: Default behaviour for unknown or ambiguous cases MUST be
  escalation, never silent auto-handling.

- **FR-E-004**: Every escalation decision MUST include a structured record
  containing: the routing outcome, the specific trigger condition(s) met,
  the intent confidence value, and the evidence considered.

- **FR-E-005**: The escalation threshold MUST be adjustable without code
  changes, and its adjustment MUST produce measurable directional changes
  in escalation precision and recall on the evaluation set.

#### FR-V: Evaluation Harness

- **FR-V-001**: The system MUST provide a complete automated evaluation
  pipeline executable via a single command with no manual steps.

- **FR-V-002**: The evaluation pipeline MUST complete in under 15 minutes
  on a standard development machine across the full golden evaluation set.

- **FR-V-003**: The evaluation pipeline MUST compute and report all of
  the following headline metrics:
  - Intent classification: per-class precision, recall, macro F1, and
    confusion matrix.
  - Retrieval quality: MRR@5 and Precision@5.
  - Reply quality: mean LLM-as-judge score (1–5 scale), score distribution.
  - LLM-judge agreement: percentage agreement or Cohen's kappa between
    judge scores and reference ratings.
  - Escalation quality: precision, recall, and false-auto-handle rate
    (unsafe auto-handle / total true-escalation cases).

- **FR-V-004**: All metrics MUST be reported alongside: the baseline
  value (random and majority-class baselines), the number of evaluation
  examples used, and a results table suitable for reproduction.

- **FR-V-005**: The evaluation report MUST include a failure mode analysis
  documenting at least five distinct failure categories with full decision
  traces drawn from real golden-set examples.

- **FR-V-006**: Evaluation results MUST be deterministic. Non-determinism
  in LLM components MUST be mitigated via fixed temperature or majority
  voting, and the mitigation strategy MUST be documented.

- **FR-V-007**: The pipeline MUST run an automated leakage check and
  cause a hard failure if any conversation ID appears in both the golden
  evaluation set and the retrieval corpus.

#### FR-L: Decision Log

- **FR-L-001**: A decision log MUST be maintained containing 10–15
  non-obvious engineering and product decisions made during the project.

- **FR-L-002**: Each decision log entry MUST contain: the decision, the
  alternatives considered, the evidence or reasoning driving the choice,
  and the expected trade-offs.

- **FR-L-003**: Decisions MUST be recorded at the time they are made.
  The following decisions MUST appear in the log: brand selection rationale,
  intent taxonomy design choices, retrieval strategy, similarity threshold,
  escalation threshold, evaluation metric selection, and prompt engineering
  choices.

### Non-Functional Requirements

- **NFR-001 — Reproducibility**: All pipeline stages MUST produce identical
  outputs given identical inputs. Random operations MUST use documented seeds.

- **NFR-002 — Auditability**: Every AI decision MUST emit a structured
  decision record queryable for post-hoc debugging and evaluation.

- **NFR-003 — No Eval Leakage**: Zero conversation-level overlap between
  the golden evaluation set and the retrieval/development corpus is a
  hard constraint enforced automatically.

- **NFR-004 — Simplicity**: The full pipeline MUST be runnable on a single
  machine. No distributed infrastructure, cloud databases, or orchestration
  frameworks may be introduced without a documented, measured justification.

- **NFR-005 — Language Scope**: The system MUST operate exclusively on
  English-language conversations. Non-English records in the raw dataset
  MUST be detected and excluded during ingestion (relevant to the 12.5%
  non-ASCII records in the full corpus; AppleSupport was selected partly
  for its 87.5% ASCII/English rate, higher than the corpus-wide average).

- **NFR-006 — Dataset Immutability**: `data/raw/twcs.csv` MUST NOT be
  modified by any pipeline or application component. All processing MUST
  write to derived artifact paths.

- **NFR-007 — Evaluation First**: The evaluation harness MUST be
  implemented and passing against stub predictions before any AI component
  is tuned or optimised (Constitution Principle I).

### Key Entities

- **RawTweet**: A single record from `data/raw/twcs.csv` identified by
  `tweet_id` (sequential integer), authored by `author_id` (anonymised
  numeric ID for customers, brand handle for agents), with direction flag
  `inbound`, timestamp `created_at`, free-text `text`, backward pointer
  `in_response_to_tweet_id`, and forward pointer(s) `response_tweet_id`
  (comma-separated, up to 1,755 observed values).

- **Conversation**: A reconstructed support thread rooted at a customer
  clean root tweet. Identified by `conversation_id` (root tweet ID).
  Contains an ordered list of `DialogueTurn`s, an attributed `brand_id`
  (the brand of the first direct brand reply), and a derived `turn_count`.

- **DialogueTurn**: One message within a Conversation. Contains `tweet_id`,
  `speaker` (customer | agent), `text` (HTML-decoded), and `created_at`.

- **IntentLabel**: One element of the exhaustive intent taxonomy.
  Defined with a human-readable name, a description, and at least three
  positive examples drawn from the corpus. Includes the "Out-of-Scope /
  Unclear" catch-all.

- **IntentPrediction**: The classifier's output for a given Conversation.
  Contains `label` (IntentLabel), `confidence` (float in [0, 1]), and
  `evidence_excerpt` (text).

- **RetrievedEvidence**: A set of historically resolved Conversations
  returned by the retrieval module. Each item carries `conversation_id`,
  `relevance_score`, `intent_label`, and `temporal_range`.

- **DraftReply**: The generated agent response for an auto-handled case.
  Contains `text`, `evidence_ids` (list of source conversation IDs),
  `relevance_scores`, `intent_label`, `escalation_decision`, and
  `escalation_rationale`.

- **EscalationDecision**: The routing outcome for a case. Contains
  `routing` (`auto-handle` | `escalate`), `trigger_conditions` (list),
  `intent_confidence`, and `evidence_considered`.

- **GoldenExample**: One rule-assisted curated evaluation record. Contains a
  Conversation, a curated `intent_label`, a curated
  `escalation_label`, and optionally a curated reference reply.
  Stored separately from the retrieval corpus with zero ID overlap enforced.

- **EvaluationReport**: The structured output of one evaluation run.
  Contains per-metric results (with baselines), the evaluation set size,
  at least five failure mode analyses with decision traces, LLM-judge
  agreement statistics, and a leakage audit result.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001 — Intent Classification Accuracy**: The intent classifier
  achieves macro F1 > 0.70 on the golden evaluation set across all
  defined intent classes (excluding the catch-all), measured against
  rule-assisted curated ground truth. Reported with per-class breakdown and confusion matrix.

- **SC-002 — Retrieval Relevance**: The retrieval module achieves
  MRR@5 > 0.50 on the golden evaluation set, measured against ground-truth
  intent relevance. Must exceed the random retrieval baseline
  by ≥ 0.20 absolute.

- **SC-003 — Reply Groundedness**: At least 85% of generated draft
  replies receive an LLM-judge groundedness score of ≥ 3 out of 5,
  where a score of 3 indicates the reply contains no unsupported factual
  claims beyond the retrieved evidence. Measured against the golden set.

- **SC-004 — Escalation Safety (False-Auto-Handle Rate)**: The
  false-auto-handle rate (system auto-handles cases where the curated
  ground truth marks the case as requiring escalation) MUST be ≤ 10% at the default escalation
  threshold setting. This is the primary safety criterion.

- **SC-005 — Escalation Precision**: Escalation precision (correct
  escalations / total predicted escalations) ≥ 0.75, ensuring the system
  does not over-escalate routine cases at the expense of operational
  efficiency.

- **SC-006 — Evaluation Pipeline Speed**: The complete evaluation pipeline
  (all headline metrics plus report generation) runs to completion in
  under 15 minutes on a single machine with no manual steps required.

- **SC-007 — Leakage Guarantee**: Zero conversation IDs appear in both
  the golden evaluation set and the retrieval corpus. This is enforced
  automatically and reported as a binary pass/fail in every evaluation run.

- **SC-008 — Reproducibility**: Running the ingestion pipeline twice
  with the same inputs produces byte-identical output files (verified
  by hash comparison).

- **SC-009 — LLM-Judge Alignment**: LLM-as-judge scores agree with
  reference ratings at a rate of ≥ 70% (within one point on a
  1–5 scale), or Cohen's kappa ≥ 0.40. Alignment must be measured and
  reported in the evaluation report.

- **SC-010 — Baseline Improvement**: The system outperforms the random
  baseline on all primary metrics (intent F1, MRR@5, escalation F1).
  Baseline comparison is mandatory in all evaluation reports.

- **SC-011 — Golden Set Coverage**: The golden evaluation set contains
  150–250 examples with at least 10 examples per defined intent class,
  at least 20 examples labelled as requiring escalation, and zero
  leakage from the retrieval corpus.

- **SC-012 — Decision Log Completeness**: The decision log contains at
  least 10 entries covering the decisions enumerated in FR-L-003.

---

## Assumptions

- **A-01 — Brand Selection**: AppleSupport is the selected focus brand,
  chosen on the basis of verified dataset statistics: 74,571 clean
  reconstructable conversations, 87.5% English/ASCII rate (highest among
  top-tier candidates), 75.4% documentation link citation rate (highest,
  enabling grounded retrieval), and a natural 52.6% DM-escalation
  boundary that provides a realistic, measurable escalation signal. This
  decision is documented in `docs/DATASET_PROFILE.md` Section 5.

- **A-02 — Dataset Immutability**: `data/raw/twcs.csv` will not be
  modified at any point. All derived data is written to `data/processed/`.

- **A-03 — Conversation Reconstruction Algorithm**: A conversation is
  defined as the BFS-reachable subtree from a clean customer root, with
  brand attribution to the first direct brand child of the root. This
  yields 74,571 AppleSupport conversations. Conversations with broken
  backward references (3,862 total in the full corpus) are excluded from
  the golden evaluation set.

- **A-04 — Temporal Split**: The leakage-free split uses a temporal
  boundary: conversations from the earlier portion of the dataset form
  the retrieval corpus; the later portion forms the candidate pool for
  golden set selection. The exact cutoff is deferred to the planning
  phase and must ensure adequate volume in both partitions.

- **A-05 — 2-Tier Benchmark Architecture**:
  1. *Curated Development Benchmark*: Authentic post-split conversations curated with deterministic rule-assisted ground truth labels according to the documented taxonomy and safety triggers (`annotator_id: "rule_assisted_curated"` in `data/processed/golden_set.jsonl`). Used for developer iteration and regression testing.
  2. *Independent Human Test Benchmark*: Authentic post-split conversations deterministically sampled (seed 42) into an unpopulated fixture (`data/processed/human_benchmark_unannotated.jsonl`). Ground truth fields remain strictly unfilled until independent manual annotation. Three-way disjointness (retrieval corpus $\cap$ curated golden set $\cap$ human benchmark $= \emptyset$) and cryptographic manifest freezing are strictly enforced.

- **A-06 — Intent Taxonomy Scope**: The initial intent taxonomy targets
  5–10 classes derived from empirical clustering of AppleSupport
  conversation text. The exact taxonomy will be defined and reviewed
  before any classifier is trained. This is a planning-phase deliverable.

- **A-07 — Single Machine Execution**: The full prototype — ingestion,
  intent classification, retrieval, reply generation, and evaluation —
  must be executable on a single developer machine without distributed
  infrastructure.

- **A-08 — LLM-as-Judge Availability**: Reply quality evaluation uses
  an LLM-as-judge approach. If the judge service is unavailable during
  an evaluation run, that metric is skipped and the report flags the gap;
  the pipeline does not fail hard on judge unavailability alone.

- **A-09 — Out of Scope for This Specification**:
  - Real-time / streaming tweet ingestion.
  - Multi-brand support (system is AppleSupport-specific).
  - Model training or fine-tuning of LLM weights.
  - User-facing web interface or API endpoint.
  - Production deployment, autoscaling, or SLA enforcement.
  - Support for languages other than English.
  - Integration with external ticketing systems (Zendesk, Salesforce, etc.).
  - Personally identifiable information handling beyond what is already
    anonymised in the dataset (all customer IDs are anonymised integers).

---

## Open Questions

None. All critical decisions have been resolved using evidence from the
verified `docs/DATASET_PROFILE.md` and the project constitution. The
following items are deferred to the planning phase with the assumption
documented above:

- Exact temporal split boundary (A-04) — documented in `docs/ARCHITECTURE.md`.
- Specific intent taxonomy classes (A-06) — determined before classifier
  training begins.
- Specific retrieval strategy (BM25 vs. dense vs. hybrid) — documented
  in `docs/ARCHITECTURE.md` with reference to the simplicity principle (VIII).
- Specific LLM / model choices — deferred; spec is technology-agnostic.

---
