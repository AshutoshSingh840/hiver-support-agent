"""Main CLI entrypoint for Hiver Support Agent."""

import json
import sys
import time
from pathlib import Path
import click
from src.config import default_config, AppConfig, EscalationConfig
from src.data.audit import compute_file_sha256
from src.data.ingestion import TweetIndex
from src.data.splitter import (
    partition_conversations,
    save_partition_jsonl,
    generate_split_manifest,
    verify_split_leakage
)
from src.intent.classifier import IntentClassifier
from src.retrieval.engine import RetrievalEngine
from src.escalation.router import EscalationRouter
from src.generator.reply_agent import ReplyAgent
from src.pipeline import SupportAgentPipeline
from src.eval.harness import EvaluationHarness
from src.eval.judge import LLMJudge
from src.data.human_benchmark import (
    sample_human_benchmark_candidates,
    save_human_annotation_template,
    load_human_annotation_records,
    validate_human_benchmark,
    freeze_human_benchmark,
)


@click.group()
@click.option("--config", type=click.Path(exists=True), help="Custom YAML config path.")
@click.option("--json", "json_output", is_flag=True, help="Emit machine-readable JSON output.")
@click.pass_context
def cli(ctx, config, json_output):
    """Hiver Support Agent: Evaluation-first AI Customer Support System."""
    ctx.ensure_object(dict)
    ctx.obj["json"] = json_output
    if config:
        ctx.obj["config"] = AppConfig.from_yaml(Path(config))
    else:
        ctx.obj["config"] = default_config


@cli.command()
@click.option("--raw-path", type=click.Path(exists=True), default="data/raw/twcs.csv", help="Path to raw twcs.csv.")
@click.option("--output-dir", type=click.Path(), default="data/processed", help="Output directory for processed files.")
@click.option("--split-date", type=str, default="2017-11-01", help="Temporal split date (YYYY-MM-DD).")
@click.option("--brand", type=str, default="AppleSupport", help="Target brand handle.")
@click.option("--max-rows", type=int, default=None, help="Limit rows for testing.")
@click.option("--json", "sub_json", is_flag=True, help="Emit machine-readable JSON output.")
@click.pass_context
def ingest(ctx, raw_path, output_dir, split_date, brand, max_rows, sub_json):
    """Ingest raw CSV, reconstruct conversation dialogue trees, and generate temporal split."""
    start_time = time.time()
    is_json = sub_json or ctx.obj.get("json", False)
    raw_p = Path(raw_path)
    out_d = Path(output_dir)
    out_d.mkdir(parents=True, exist_ok=True)

    if not is_json:
        click.echo(f"[*] Starting ingestion from {raw_p} for brand '{brand}'...")

    raw_hash = compute_file_sha256(raw_p)
    index = TweetIndex()
    total_rows = index.load_from_csv(raw_p, max_rows=max_rows)

    conversations = []
    for root_id in index.clean_customer_roots:
        conv = index.reconstruct_conversation(root_id)
        if conv and conv.brand == brand:
            conversations.append(conv)

    retrieval_corpus, eval_pool = partition_conversations(conversations, split_date_str=split_date)

    retrieval_path = out_d / "retrieval_corpus.jsonl"
    eval_pool_path = out_d / "eval_candidate_pool.jsonl"
    manifest_path = out_d / "split_manifest.json"

    save_partition_jsonl(retrieval_corpus, retrieval_path)
    save_partition_jsonl(eval_pool, eval_pool_path)

    manifest = generate_split_manifest(
        retrieval_corpus=retrieval_corpus,
        eval_candidates=eval_pool,
        raw_hash=raw_hash,
        output_path=manifest_path,
        split_date_str=split_date
    )

    elapsed = round(time.time() - start_time, 2)

    result = {
        "status": "success",
        "input_file": str(raw_p),
        "input_sha256": raw_hash,
        "total_raw_rows_scanned": total_rows,
        "reconstructed_conversations": len(conversations),
        "retrieval_corpus_count": len(retrieval_corpus),
        "eval_candidate_pool_count": len(eval_pool),
        "runtime_sec": elapsed,
        "manifest_path": str(manifest_path)
    }

    if is_json:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f"[+] Reconstructed {len(conversations)} {brand} conversations in {elapsed}s.")
        click.echo(f"    - Retrieval Corpus: {len(retrieval_corpus)} conversations (< {split_date})")
        click.echo(f"    - Eval Candidate Pool: {len(eval_pool)} conversations (>= {split_date})")
        click.echo(f"    - Manifest: {manifest_path}")


@cli.command("audit-leakage")
@click.option("--manifest", type=click.Path(exists=True), default="data/processed/split_manifest.json", help="Path to split manifest.")
@click.option("--golden-set", type=click.Path(exists=True), required=False, help="Path to curated golden set JSONL.")
@click.option("--human-benchmark", type=click.Path(exists=True), required=False, help="Path to human benchmark JSONL.")
@click.option("--json", "sub_json", is_flag=True, help="Emit machine-readable JSON output.")
@click.pass_context
def audit_leakage(ctx, manifest, golden_set, human_benchmark, sub_json):
    """Verify zero conversation ID overlap between retrieval corpus and evaluation sets."""
    is_json = sub_json or ctx.obj.get("json", False)
    with open(manifest, "r", encoding="utf-8") as f:
        mdata = json.load(f)

    retrieval_ids = set(mdata.get("retrieval_conversation_ids", []))
    
    overlaps = {}
    eval_ids = set()
    eval_label = "eval candidate pool"

    golden_ids = set()
    if golden_set:
        with open(golden_set, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    golden_ids.add(int(item["conversation_id"]))
        ret_gold_pass, ret_gold_overlap = verify_split_leakage(retrieval_ids, golden_ids)
        if not ret_gold_pass:
            overlaps["retrieval_vs_golden"] = list(ret_gold_overlap)

    human_ids = set()
    if human_benchmark:
        with open(human_benchmark, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    human_ids.add(int(item["conversation_id"]))
        ret_human_pass, ret_human_overlap = verify_split_leakage(retrieval_ids, human_ids)
        if not ret_human_pass:
            overlaps["retrieval_vs_human"] = list(ret_human_overlap)
        
        if golden_ids:
            gold_human_pass, gold_human_overlap = verify_split_leakage(golden_ids, human_ids)
            if not gold_human_pass:
                overlaps["golden_vs_human"] = list(gold_human_overlap)

    if not golden_set and not human_benchmark:
        eval_ids = set(mdata.get("eval_conversation_ids", []))
        passed, overlap = verify_split_leakage(retrieval_ids, eval_ids)
        if not passed:
            overlaps["retrieval_vs_eval_pool"] = list(overlap)

    if overlaps:
        msg = f"[FAIL] Data leakage detected across partitions: {overlaps}"
        click.echo(msg, err=True)
        sys.exit(2)
    else:
        target_desc = []
        if golden_set:
            target_desc.append(f"golden set ({len(golden_ids)})")
        if human_benchmark:
            target_desc.append(f"human benchmark ({len(human_ids)})")
        if not target_desc:
            target_desc.append(f"eval candidate pool ({len(eval_ids)})")

        desc_str = " and ".join(target_desc)
        if is_json:
            click.echo(json.dumps({"status": "PASS", "overlap_count": 0, "eval_targets": target_desc}))
        else:
            click.echo(f"[PASS] Zero conversation ID overlap between retrieval corpus ({len(retrieval_ids)}) and {desc_str}.")
        sys.exit(0)


@cli.command("sample-human-benchmark")
@click.option("--eval-pool", type=click.Path(exists=False), default="data/processed/eval_candidate_pool.jsonl", help="Path to eval candidate pool JSONL.")
@click.option("--raw-path", type=click.Path(exists=False), default="data/raw/twcs.csv", help="Fallback path to raw twcs.csv if pool not present.")
@click.option("--manifest", type=click.Path(exists=True), default="data/processed/split_manifest.json", help="Path to split manifest.")
@click.option("--golden-set", type=click.Path(exists=True), default="data/processed/golden_set.jsonl", help="Path to golden set JSONL.")
@click.option("--output", type=click.Path(), default="data/processed/human_benchmark_unannotated.jsonl", help="Destination unannotated template.")
@click.option("--count", type=int, default=200, help="Number of authentic conversations to sample.")
@click.option("--seed", type=int, default=42, help="Deterministic random seed.")
@click.option("--json", "sub_json", is_flag=True, help="Emit machine-readable JSON output.")
@click.pass_context
def sample_human_benchmark_cmd(ctx, eval_pool, raw_path, manifest, golden_set, output, count, seed, sub_json):
    """Deterministically sample authentic post-split AppleSupport conversations for human annotation."""
    is_json = sub_json or ctx.obj.get("json", False)
    from src.data.models import Conversation

    with open(manifest, "r", encoding="utf-8") as f:
        mdata = json.load(f)
    retrieval_ids = set(mdata.get("retrieval_conversation_ids", []))

    golden_ids = set()
    if Path(golden_set).exists():
        with open(golden_set, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    golden_ids.add(int(item["conversation_id"]))

    conversations = []
    pool_path = Path(eval_pool)
    if pool_path.exists():
        with open(pool_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    conversations.append(Conversation.model_validate_json(line))
    else:
        # Reconstruct from raw CSV
        from src.data.ingestion import reconstruct_conversations_for_brand
        conversations = list(reconstruct_conversations_for_brand(Path(raw_path), "AppleSupport"))

    records = sample_human_benchmark_candidates(
        conversations=conversations,
        retrieval_ids=retrieval_ids,
        golden_ids=golden_ids,
        sample_size=count,
        seed=seed,
        target_brand="AppleSupport"
    )

    out_p = Path(output)
    save_human_annotation_template(records, out_p)

    res = {
        "status": "success",
        "output_path": str(out_p),
        "sampled_count": len(records),
        "seed": seed,
        "ground_truth_fields_populated": False,
        "leakage_retrieval_overlap": 0,
        "leakage_golden_overlap": 0
    }

    if is_json:
        click.echo(json.dumps(res, indent=2))
    else:
        click.echo(f"[+] Successfully sampled {len(records)} authentic conversations for human annotation.")
        click.echo(f"    - Output fixture: {out_p}")
        click.echo(f"    - Seed: {seed} (deterministic)")
        click.echo(f"    - Ground-truth fields: unpopulated (null) awaiting human review.")
        click.echo(f"    - Zero overlap with retrieval corpus and curated golden set enforced.")


@cli.command("validate-human-benchmark")
@click.option("--benchmark-file", type=click.Path(exists=True), default="data/processed/human_benchmark.jsonl", help="Path to human benchmark JSONL.")
@click.option("--manifest", type=click.Path(exists=True), default="data/processed/split_manifest.json", help="Path to split manifest.")
@click.option("--golden-set", type=click.Path(exists=True), default="data/processed/golden_set.jsonl", help="Path to curated golden set.")
@click.option("--count", type=int, default=200, help="Expected record count.")
@click.option("--json", "sub_json", is_flag=True, help="Emit machine-readable JSON output.")
@click.pass_context
def validate_human_benchmark_cmd(ctx, benchmark_file, manifest, golden_set, count, sub_json):
    """Validate that human benchmark dataset is complete, uncorrupted, and leak-free."""
    is_json = sub_json or ctx.obj.get("json", False)
    with open(manifest, "r", encoding="utf-8") as f:
        mdata = json.load(f)
    retrieval_ids = set(mdata.get("retrieval_conversation_ids", []))

    golden_ids = set()
    with open(golden_set, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                golden_ids.add(int(item["conversation_id"]))

    records = load_human_annotation_records(Path(benchmark_file))
    passed, errors = validate_human_benchmark(records, retrieval_ids, golden_ids, expected_count=count)

    res = {
        "status": "PASS" if passed else "FAIL",
        "benchmark_file": str(benchmark_file),
        "record_count": len(records),
        "error_count": len(errors),
        "errors": errors
    }

    if is_json:
        click.echo(json.dumps(res, indent=2))
    else:
        if passed:
            click.echo(f"[PASS] Human benchmark validation passed for {benchmark_file} ({len(records)} records).")
        else:
            click.echo(f"[FAIL] Human benchmark validation failed with {len(errors)} error(s):", err=True)
            for err in errors[:10]:
                click.echo(f"  - {err}", err=True)
    if not passed:
        sys.exit(1)


@cli.command("freeze-human-benchmark")
@click.option("--benchmark-file", type=click.Path(exists=True), default="data/processed/human_benchmark.jsonl", help="Path to completed human benchmark JSONL.")
@click.option("--raw-path", type=click.Path(exists=False), default="data/raw/twcs.csv", help="Path to raw source dataset.")
@click.option("--manifest", type=click.Path(exists=True), default="data/processed/split_manifest.json", help="Path to split manifest.")
@click.option("--golden-set", type=click.Path(exists=True), default="data/processed/golden_set.jsonl", help="Path to curated golden set.")
@click.option("--output-manifest", type=click.Path(), default="data/processed/human_benchmark_manifest.json", help="Destination manifest path.")
@click.option("--version", type=str, default="v1.0-human", help="Annotation version string.")
@click.option("--json", "sub_json", is_flag=True, help="Emit machine-readable JSON output.")
@click.pass_context
def freeze_human_benchmark_cmd(ctx, benchmark_file, raw_path, manifest, golden_set, output_manifest, version, sub_json):
    """Validate and cryptographically freeze completed human evaluation benchmark."""
    is_json = sub_json or ctx.obj.get("json", False)
    with open(manifest, "r", encoding="utf-8") as f:
        mdata = json.load(f)
    retrieval_ids = set(mdata.get("retrieval_conversation_ids", []))

    golden_ids = set()
    with open(golden_set, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                golden_ids.add(int(item["conversation_id"]))

    try:
        bm_manifest = freeze_human_benchmark(
            benchmark_path=Path(benchmark_file),
            raw_twcs_path=Path(raw_path),
            retrieval_ids=retrieval_ids,
            golden_ids=golden_ids,
            output_manifest_path=Path(output_manifest),
            annotation_version=version
        )
    except Exception as e:
        click.echo(f"[ERROR] Failed to freeze human benchmark: {e}", err=True)
        sys.exit(1)

    if is_json:
        click.echo(bm_manifest.model_dump_json(indent=2))
    else:
        click.echo(f"[+] Successfully froze human benchmark ({bm_manifest.record_count} records).")
        click.echo(f"    - Manifest: {output_manifest}")
        click.echo(f"    - Benchmark SHA-256: {bm_manifest.benchmark_sha256}")
        click.echo(f"    - Version: {bm_manifest.annotation_version}")
        click.echo(f"    - Annotators: {', '.join(bm_manifest.annotator_ids)}")



@cli.command()
@click.option("--text", type=str, required=True, help="Customer inquiry text.")
@click.option("--conversation-id", type=int, default=999001, help="Simulated conversation ID.")
@click.option("--retrieval-corpus", type=click.Path(exists=False), default="data/processed/retrieval_corpus.jsonl", help="Path to retrieval corpus.")
@click.option("--json", "sub_json", is_flag=True, help="Emit machine-readable JSON output.")
@click.pass_context
def handle(ctx, text, conversation_id, retrieval_corpus, sub_json):
    """Process a single customer ticket through the entire agent pipeline."""
    is_json = sub_json or ctx.obj.get("json", False)

    retriever = RetrievalEngine()
    rc_path = Path(retrieval_corpus)
    if rc_path.exists():
        from src.data.models import Conversation
        convs = []
        with open(rc_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    convs.append(Conversation.model_validate_json(line))
        retriever.build_from_conversations(convs)

    pipeline = SupportAgentPipeline(retrieval_engine=retriever)
    result = pipeline.process_ticket(conversation_id=conversation_id, text=text)

    if is_json:
        # Filter non-serializable objects
        clean_res = {k: v for k, v in result.items() if k != "retrieval_items"}
        click.echo(json.dumps(clean_res, indent=2))
    else:
        click.echo(f"\n=======================================================")
        click.echo(f"  HIVER SUPPORT AGENT DECISION TRACE (ID #{conversation_id})")
        click.echo(f"=======================================================")
        click.echo(f"Customer Inquiry: \"{text}\"")
        click.echo(f"\n[1] Intent Classification:")
        click.echo(f"    - Predicted Label : {result['intent']['label']}")
        click.echo(f"    - Confidence Score: {result['intent']['confidence']:.2f}")
        click.echo(f"    - Excerpt         : \"{result['intent']['supporting_excerpt']}\"")
        
        click.echo(f"\n[2] Evidence Retrieval:")
        click.echo(f"    - Retrieved Items : {result['retrieval']['top_k']} precedents (Top Score: {result['retrieval']['top_score']:.2f})")
        for ev in result["retrieval"]["evidence"][:2]:
            click.echo(f"      * [#{ev['source_conversation_id']}] (Score: {ev['relevance_score']:.2f}): {ev['resolution_text'][:70]}...")

        click.echo(f"\n[3] Escalation Routing:")
        click.echo(f"    - Decision  : {result['escalation']['routing'].upper()}")
        click.echo(f"    - Risk Score: {result['escalation'].get('risk_score', 0.0):.2f} (Threshold: {result['escalation'].get('routing_threshold', 0.45):.2f})")
        click.echo(f"    - Rationale : {result['escalation']['rationale']}")

        if result.get("draft_reply"):
            click.echo(f"\n[4] Grounded Draft Reply:")
            click.echo(f"    \"{result['draft_reply']['draft_text']}\"")
            click.echo(f"    Cited Precedent IDs: {result['draft_reply']['cited_evidence_ids']}")
        else:
            click.echo(f"\n[4] Draft Reply: [Suppressed due to human escalation]")

        click.echo(f"\nLatency: {result['latency_ms']} ms")
        click.echo(f"=======================================================\n")


@cli.command()
@click.option("--benchmark", "benchmark_type", type=click.Choice(["human", "golden"], case_sensitive=False), default="human", help="Benchmark selection: 'human' (Tier-2 human-validated) or 'golden' (Tier-1 dev golden set).")
@click.option("--benchmark-file", type=click.Path(exists=False), default=None, help="Custom path to benchmark JSONL (overrides preset).")
@click.option("--golden-set", type=click.Path(exists=False), default=None, help="Path to golden set JSONL (for backward compatibility).")
@click.option("--human-manifest", type=click.Path(exists=False), default="data/processed/human_benchmark_manifest.json", help="Path to human benchmark manifest.")
@click.option("--manifest", type=click.Path(exists=True), default="data/processed/split_manifest.json", help="Path to split manifest.")
@click.option("--retrieval-corpus", type=click.Path(exists=True), default="data/processed/retrieval_corpus.jsonl", help="Path to retrieval corpus.")
@click.option("--output-report", type=click.Path(), default="reports/evaluation_report.json", help="Destination report path.")
@click.option("--escalation-threshold", type=float, default=0.75, help="Auto-handle confidence threshold.")
@click.option("--json", "sub_json", is_flag=True, help="Emit machine-readable JSON output.")
@click.pass_context
def evaluate(ctx, benchmark_type, benchmark_file, golden_set, human_manifest, manifest, retrieval_corpus, output_report, escalation_threshold, sub_json):
    """Run full automated evaluation pipeline against evaluation benchmark."""
    is_json = sub_json or ctx.obj.get("json", False)

    # Determine benchmark selection
    if benchmark_file:
        target_benchmark_path = Path(benchmark_file)
        b_name = f"custom_benchmark ({target_benchmark_path.name})"
        h_manifest = Path(human_manifest) if human_manifest and Path(human_manifest).exists() else None
    elif golden_set:
        target_benchmark_path = Path(golden_set)
        b_name = "golden_set_dev"
        h_manifest = None
    elif benchmark_type.lower() == "golden":
        target_benchmark_path = Path("data/processed/golden_set.jsonl")
        b_name = "golden_set_dev"
        h_manifest = None
    else:
        target_benchmark_path = Path("data/processed/human_benchmark_unannotated.jsonl")
        b_name = "human_benchmark_v1.0"
        h_manifest = Path(human_manifest) if human_manifest and Path(human_manifest).exists() else None

    if not is_json:
        click.echo(f"[*] Evaluation Benchmark: {b_name} ({target_benchmark_path})")
        click.echo("[*] Loading retrieval corpus and building index...")

    from src.data.models import Conversation
    convs = []
    with open(retrieval_corpus, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                convs.append(Conversation.model_validate_json(line))

    intent_clf = IntentClassifier()
    retriever = RetrievalEngine()
    retriever.build_from_conversations(convs, intent_classifier_fn=intent_clf.predict)

    router = EscalationRouter(config=EscalationConfig(intent_confidence_threshold=escalation_threshold))
    reply_agent = ReplyAgent()
    pipeline = SupportAgentPipeline(
        retrieval_engine=retriever,
        intent_classifier=intent_clf,
        escalation_router=router,
        reply_agent=reply_agent
    )

    harness = EvaluationHarness(
        golden_set_path=target_benchmark_path,
        split_manifest_path=Path(manifest),
        escalation_threshold=escalation_threshold,
        benchmark_name=b_name,
        human_manifest_path=h_manifest
    )

    if not is_json:
        click.echo(f"[*] Running benchmark across {target_benchmark_path} evaluation examples...")

    report = harness.run_evaluation(
        intent_classifier_fn=lambda txt: intent_clf.predict(0, txt),
        retriever_fn=lambda txt, intent: retriever.retrieve(0, txt, intent, top_k=5),
        pipeline_fn=lambda txt: pipeline.process_ticket(0, txt)
    )

    out_p = Path(output_report)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))

    if is_json:
        click.echo(report.model_dump_json(indent=2))
    else:
        click.echo("\n" + "=" * 65)
        click.echo("             HIVER SUPPORT AGENT EVALUATION REPORT")
        click.echo("=" * 65)
        click.echo(f"Run ID: {report.run_id} | Benchmark: {report.benchmark_name} | Runtime: {report.execution_duration_sec}s | Samples: {report.golden_set_size}")
        click.echo(f"Pre-Run Leakage Audit: {'[PASSED - 0 Overlap]' if report.leakage_check_passed else '[FAILED]'}\n")

        click.echo(f"{'Metric':<38} | {'Score':<8} | {'Baseline':<8} | {'Target':<10}")
        click.echo("-" * 65)
        for key, m in report.headline_metrics.items():
            status = "PASS" if m.passed_target else "FAIL"
            click.echo(f"{m.metric_name:<38} | {m.score:<8.3f} | {m.baseline_score:<8.3f} | {status:<10}")

        # Baselines Comparison Table
        maj_f1 = report.headline_metrics["intent_macro_f1"].baseline_score
        simple_f1 = report.simple_baseline_macro_f1 or 0.0
        simple_f1_std = report.simple_baseline_macro_f1_std or 0.0
        agent_f1 = report.headline_metrics["intent_macro_f1"].score
        click.echo("\n" + "-" * 65)
        click.echo("Intent Classification Baselines Comparison:")
        click.echo(f"  [1] Trivial Baseline (Majority Class)    : Macro F1 = {maj_f1:.4f} (eval: all 200 golden examples)")
        click.echo(f"  [2] Simple ML Baseline (TF-IDF + LogReg) : Macro F1 = {simple_f1:.4f} ± {simple_f1_std:.4f} (5-fold CV on 200 golden examples)")
        click.echo(f"  [3] Hiver Support Agent (Hybrid Model)   : Macro F1 = {agent_f1:.4f} (eval: all 200 golden examples)")
        click.echo(f"  NOTE: Baseline [2] uses 5-fold CV (160 train/40 test per fold); Agent [3] is scored on full 200 examples.")
        click.echo(f"  These are not a perfectly apples-to-apples comparison. See EVALUATION_REPORT.md §1 for details.")

        click.echo("\nPer-Class Intent Macro F1:")
        for cls_name, f1_val in report.per_class_f1.items():
            click.echo(f"  - {cls_name:<20}: {f1_val:.3f}")

        if report.routing_threshold_sweep:
            click.echo("\nRouting Risk Threshold Sweep (Safety vs Coverage Trade-off):")
            click.echo(f"  {'Tau':>5} | {'False-Auto':>10} | {'Precision':>10} | {'Recall':>10} | {'Coverage':>10} | {'Esc F1':>8}")
            click.echo("  " + "-" * 60)
            for s in report.routing_threshold_sweep:
                click.echo(f"  {s['threshold']:5.2f} | {s['false_auto_handle_rate']:10.3f} | {s['escalation_precision']:10.3f} | {s['escalation_recall']:10.3f} | {s['auto_handle_coverage']:10.3f} | {s['escalation_f1']:8.3f}")

        click.echo(f"\nDocumented Failure Mode Analyses: {len(report.failure_mode_analysis)} distinct cases captured.")
        for idx, fc in enumerate(report.failure_mode_analysis[:3], start=1):
            query_snippet = fc['customer_query'][:60].encode('ascii', 'replace').decode('ascii')
            click.echo(f"  [{idx}] Category: {fc['failure_category']} ({fc['golden_id']})")
            click.echo(f"      Query   : \"{query_snippet}...\"")
            click.echo(f"      Expected: Intent={fc['expected_intent']}, Routing={fc['expected_routing']}")
            click.echo(f"      Actual  : Intent={fc['predicted_intent']}, Routing={fc['predicted_routing']}")

        click.echo(f"\nFull report saved to: {out_p}")
        click.echo("=" * 65 + "\n")


if __name__ == "__main__":
    cli()
