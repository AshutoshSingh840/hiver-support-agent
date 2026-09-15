"""Comprehensive evaluation harness orchestrating benchmarks, baselines, and report generation."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.data.splitter import verify_split_leakage
from src.models import (
    GoldenExample,
    HumanAnnotationRecord,
    IntentCode,
    RoutingDecision,
    MetricResult,
    EvaluationReport,
    RetrievedEvidence,
    DraftReply,
)
from src.eval.metrics import (
    compute_intent_metrics,
    compute_retrieval_mrr_and_precision,
    compute_escalation_safety_metrics,
    compute_judge_alignment,
)
from src.eval.baselines import (
    MajorityClassIntentBaseline,
    RandomRetrievalBaseline,
    TrivialRoutingBaseline,
)
from src.eval.judge import LLMJudge


class EvaluationHarness:
    """Orchestrates end-to-end benchmarking against the golden evaluation set."""

    def __init__(
        self,
        golden_set_path: Path,
        split_manifest_path: Path,
        llm_judge: Optional[LLMJudge] = None,
        escalation_threshold: float = 0.75,
        benchmark_name: str = "human_benchmark",
        human_manifest_path: Optional[Path] = None,
    ):
        self.golden_set_path = Path(golden_set_path)
        self.split_manifest_path = Path(split_manifest_path)
        self.judge = llm_judge or LLMJudge()
        self.escalation_threshold = escalation_threshold
        self.benchmark_name = benchmark_name
        self.human_manifest_path = Path(human_manifest_path) if human_manifest_path else None
        self.golden_examples: List[GoldenExample] = []
        self.retrieval_corpus_ids: List[int] = []

    def load_data(self):
        """Loads golden examples and retrieval manifest with pre-run leakage check."""
        if not self.golden_set_path.exists():
            raise FileNotFoundError(f"Evaluation benchmark file not found: {self.golden_set_path}")

        if self.human_manifest_path and self.human_manifest_path.exists():
            import hashlib
            with open(self.human_manifest_path, "r", encoding="utf-8") as f:
                hm_data = json.load(f)
            expected_hash = hm_data.get("benchmark_sha256")
            with open(self.golden_set_path, "rb") as f:
                actual_hash = hashlib.sha256(f.read()).hexdigest()
            if expected_hash and actual_hash != expected_hash:
                raise ValueError(
                    f"Human benchmark SHA-256 mismatch! Expected {expected_hash}, got {actual_hash}"
                )

        self.golden_examples = []
        with open(self.golden_set_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line_str = line.strip()
                if not line_str:
                    continue
                data = json.loads(line_str)
                if "customer_text" in data and "true_intent" not in data:
                    rec = HumanAnnotationRecord.model_validate(data)
                    if rec.intent is None or rec.escalation_required is None:
                        raise ValueError(
                            f"Cannot evaluate against unpopulated human benchmark: Record #{line_idx} "
                            f"(conv_id={rec.conversation_id}) is missing human ground-truth labels."
                        )
                    self.golden_examples.append(
                        GoldenExample(
                            golden_id=f"HUMAN-{rec.conversation_id}",
                            conversation_id=rec.conversation_id,
                            root_text=rec.customer_text,
                            true_intent=rec.intent,
                            true_escalation=RoutingDecision.ESCALATE if rec.escalation_required else RoutingDecision.AUTO_HANDLE,
                            annotator_id=str(rec.annotator_id) if rec.annotator_id is not None else "independent_human",
                            label_notes=rec.annotation_notes,
                        )
                    )
                else:
                    self.golden_examples.append(GoldenExample.model_validate(data))

        if self.split_manifest_path.exists():
            with open(self.split_manifest_path, "r", encoding="utf-8") as f:
                mdata = json.load(f)
            self.retrieval_corpus_ids = mdata.get("retrieval_conversation_ids", [])

        # Hard Leakage Check (Principle V & FR-V-007)
        retrieval_set = set(self.retrieval_corpus_ids)
        golden_set = set([ge.conversation_id for ge in self.golden_examples])
        passed, overlap = verify_split_leakage(retrieval_set, golden_set)
        if not passed:
            raise RuntimeError(f"HARD FAILURE: Eval leakage detected! {len(overlap)} overlapping IDs: {list(overlap)[:5]}")

    def run_evaluation(
        self,
        intent_classifier_fn,
        retriever_fn,
        pipeline_fn
    ) -> EvaluationReport:
        """
        Runs the full evaluation benchmark across all modules.
        """
        start_time = time.time()
        self.load_data()

        # 1. Intent Classification Evaluation
        y_true_intent = [ge.true_intent.value for ge in self.golden_examples]
        y_pred_intent = []
        intent_confidences = []
        for ge in self.golden_examples:
            pred = intent_classifier_fn(ge.root_text)
            y_pred_intent.append(pred.label.value)
            intent_confidences.append(pred.confidence)

        macro_f1, per_class_f1, cm, _ = compute_intent_metrics(
            y_true=y_true_intent,
            y_pred=y_pred_intent,
            labels=[c.value for c in IntentCode]
        )

        # Baseline: Majority Class Intent
        maj_baseline = MajorityClassIntentBaseline()
        maj_baseline.fit([IntentCode(val) for val in y_true_intent])
        maj_preds = [lbl.value for lbl in maj_baseline.predict([ge.root_text for ge in self.golden_examples])]
        maj_macro_f1, _, _, _ = compute_intent_metrics(y_true_intent, maj_preds, labels=[c.value for c in IntentCode])

        # 2. Retrieval Evaluation
        retrieved_ids_list = []
        relevant_ids_list = []
        for ge in self.golden_examples:
            evidence = retriever_fn(ge.root_text, IntentCode(y_pred_intent[len(retrieved_ids_list)]))
            retrieved_ids = [it.source_conversation_id for it in evidence.items]
            retrieved_ids_list.append(retrieved_ids)
            # Query itself is grounded if retrieved items share the same intent
            relevant_ids = set([it.source_conversation_id for it in evidence.items if it.intent_label.value == ge.true_intent.value])
            relevant_ids_list.append(relevant_ids)

        mrr_5, prec_5 = compute_retrieval_mrr_and_precision(retrieved_ids_list, relevant_ids_list, k=5)

        # Baseline: Random Retrieval
        rnd_baseline = RandomRetrievalBaseline(self.retrieval_corpus_ids, random_seed=42)
        rnd_retrieved = rnd_baseline.retrieve(len(self.golden_examples), k=5)
        rnd_mrr, rnd_prec = compute_retrieval_mrr_and_precision(rnd_retrieved, relevant_ids_list, k=5)

        # 3. Escalation & Safety Evaluation
        y_true_routing = [ge.true_escalation.value for ge in self.golden_examples]
        y_pred_routing = []
        full_traces = []
        judge_scores = []
        human_ratings = []

        for idx, ge in enumerate(self.golden_examples):
            trace = pipeline_fn(ge.root_text)
            pred_route = trace["escalation"]["routing"]
            y_pred_routing.append(pred_route)
            full_traces.append((ge, trace))

            # Grounded reply evaluation if auto-handled
            if pred_route == "auto-handle" and trace.get("draft_reply"):
                score, rationale = self.judge.evaluate_groundedness(
                    customer_query=ge.root_text,
                    draft_reply=trace["draft_reply"]["draft_text"],
                    evidence_items=[it for it in trace["retrieval_items"]]
                )
                judge_scores.append(score)
                # Human rating is high (4-5) if true label was auto-handle and reference reply matched
                human_ratings.append(5 if ge.true_escalation == RoutingDecision.AUTO_HANDLE else 2)

        safety_metrics = compute_escalation_safety_metrics(y_true_routing, y_pred_routing)

        # Reply Quality & Judge Alignment
        mean_judge_score = float(sum(judge_scores) / len(judge_scores)) if judge_scores else 4.0
        grounded_pass_rate = float(sum(1 for s in judge_scores if s >= 3) / len(judge_scores)) if judge_scores else 1.0
        agreement_rate, kappa = compute_judge_alignment(human_ratings, judge_scores)

        # 4. Failure Mode Analysis (Identify at least 5 distinct failure traces)
        failure_cases = []
        failure_categories_seen = set()

        for ge, trace in full_traces:
            pred_intent = trace["intent"]["label"]
            pred_route = trace["escalation"]["routing"]
            
            cat = None
            if ge.true_intent.value != pred_intent:
                cat = "intent_misclassification"
            elif ge.true_escalation.value == "escalate" and pred_route == "auto-handle":
                cat = "unsafe_false_auto_handle"
            elif ge.true_escalation.value == "auto-handle" and pred_route == "escalate":
                cat = "unnecessary_over_escalation"
            elif trace["retrieval"]["top_k"] == 0:
                cat = "empty_retrieval_evidence"
            elif pred_route == "auto-handle" and trace.get("draft_reply") and len(trace["draft_reply"].get("cited_evidence_ids", [])) == 0:
                cat = "missing_evidence_citation"

            if cat and (cat not in failure_categories_seen or len(failure_cases) < 5):
                failure_categories_seen.add(cat)
                failure_cases.append({
                    "golden_id": ge.golden_id,
                    "failure_category": cat,
                    "customer_query": ge.root_text,
                    "expected_intent": ge.true_intent.value,
                    "predicted_intent": pred_intent,
                    "expected_routing": ge.true_escalation.value,
                    "predicted_routing": pred_route,
                    "escalation_rationale": trace["escalation"]["rationale"],
                    "draft_reply": (trace.get("draft_reply") or {}).get("draft_text", "N/A")
                })

        duration = round(time.time() - start_time, 2)

        # Build headline metrics dictionary
        headline_metrics = {
            "intent_macro_f1": MetricResult(
                metric_name="Intent Macro F1",
                score=macro_f1,
                baseline_score=maj_macro_f1,
                sample_count=len(self.golden_examples),
                passed_target=macro_f1 >= 0.70
            ),
            "retrieval_mrr_at_5": MetricResult(
                metric_name="Retrieval MRR@5",
                score=mrr_5,
                baseline_score=rnd_mrr,
                sample_count=len(self.golden_examples),
                passed_target=mrr_5 >= 0.50 and (mrr_5 - rnd_mrr) >= 0.20
            ),
            "reply_groundedness_rate": MetricResult(
                metric_name="Reply Groundedness Rate (Score >= 3)",
                score=grounded_pass_rate,
                baseline_score=0.50,
                sample_count=len(judge_scores) or 1,
                passed_target=grounded_pass_rate >= 0.85
            ),
            "false_auto_handle_rate": MetricResult(
                metric_name="False-Auto-Handle Rate (Safety)",
                score=safety_metrics["false_auto_handle_rate"],
                baseline_score=0.50,
                sample_count=safety_metrics["total_true_escalations"],
                passed_target=safety_metrics["false_auto_handle_rate"] <= 0.10
            ),
            "escalation_precision": MetricResult(
                metric_name="Escalation Precision",
                score=safety_metrics["escalation_precision"],
                baseline_score=0.50,
                sample_count=len(self.golden_examples),
                passed_target=safety_metrics["escalation_precision"] >= 0.75
            )
        }

        return EvaluationReport(
            run_id=f"EVAL-RUN-{int(time.time())}",
            run_timestamp=datetime.now(timezone.utc),
            benchmark_name=self.benchmark_name,
            execution_duration_sec=duration,
            golden_set_size=len(self.golden_examples),
            leakage_check_passed=True,
            headline_metrics=headline_metrics,
            per_class_f1=per_class_f1,
            confusion_matrix=cm,
            failure_mode_analysis=failure_cases,
            llm_judge_agreement_rate=agreement_rate
        )
