"""Independent evidence gate for GDM52."""
from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from experiments.followup49.collect import derived, metrics_for
from experiments.followup50.collect import holdout_detail
from experiments.followup51 import collect as g51collect
from .run import ARMS, FORMAT

_BASE_VERIFY = g51collect.verify_summary


def load_json(path):
    return json.loads(Path(path).read_text())


def verify_summary(path):
    # Reuse the canonical GDM51 evidence gate for checkpoints, changed weights,
    # GraphQL execution, Rover composition, timing and prediction recounts.
    g51collect.ARMS = ARMS
    g51collect.FORMAT = FORMAT
    summary, detail = _BASE_VERIFY(path)
    errors = []
    cfg = summary.get("config", {})
    training = summary.get("training", {})
    calibration = load_json(path.parent / "calibration.json").get("selected", {})
    arm = ARMS.get(cfg.get("arm"), {})
    strategy = arm.get("calibration_strategy")

    if cfg.get("calibration_strategy") != strategy:
        errors.append("config calibration strategy mismatch")
    if training.get("gdm52_family") != "status-specific-calibration":
        errors.append("missing GDM52 family evidence")
    if training.get("gdm52_calibration_strategy") != strategy:
        errors.append("training calibration strategy mismatch")
    if calibration.get("threshold_selection") != strategy:
        errors.append("selected threshold strategy mismatch")
    if strategy and strategy.startswith("sequential-"):
        if not isinstance(calibration.get("none_gate_validation_metrics"), dict):
            errors.append("missing NONE gate validation metrics")
        if not isinstance(calibration.get("ambiguity_gate_validation_metrics"), dict):
            errors.append("missing ambiguity gate validation metrics")
    if strategy == "sequential-status-specific-noninferior":
        floor = calibration.get("accepted_status_recall_floor")
        if not isinstance(floor, (int, float)):
            errors.append("missing accepted recall floor")
        if not isinstance(calibration.get("joint_baseline_validation_metrics"), dict):
            errors.append("missing joint baseline validation metrics")
        reported = training.get("selected_calibration_validation_metrics", {})
        if isinstance(floor, (int, float)) and reported.get("accepted_status_recall", -1) + 1e-12 < floor:
            errors.append("accepted recall floor violated")

    encoder = summary.get("encoder", {})
    head = encoder.get("head_training_reproducibility", {})
    if cfg.get("backbone") != "hash":
        if head.get("gdm52_execution_scope") != "capability_and_ambiguity_heads":
            errors.append("missing GDM52 execution scope")
        if head.get("canonical_gdm50_numerical_source_commit") != "62d2720800f5d98e1521e0200766a80e998ebe9b":
            errors.append("wrong canonical GDM50 numerical source")
        if head.get("canonical_gdm51_architecture_source_commit") != "f52baf82222947ca00139437e90fe9b5428bd9ae":
            errors.append("wrong canonical GDM51 architecture source")
        if head.get("gdm52_change_scope") != "validation-only status-specific threshold calibration":
            errors.append("wrong GDM52 change scope")

    if "Portfolio/Ledger" not in summary.get("calibration_scope", ""):
        errors.append("wrong calibration domain scope")
    if "Shipment/Invoice" not in summary.get("secondary_holdout_scope", ""):
        errors.append("wrong holdout domain scope")
    if "GDM46-GDM51" not in summary.get("regression_scope", ""):
        errors.append("regression scope does not include canonical GDM51 holdout")

    if errors:
        raise AssertionError(path.as_posix() + ": " + "; ".join(sorted(set(errors))))
    return summary, detail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--seeds", default="5201,5202")
    parser.add_argument("--tasks", default="operation,schema")
    args = parser.parse_args()
    root = Path(args.root)
    seeds = [int(value) for value in args.seeds.split(",") if value]
    tasks = [value for value in args.tasks.split(",") if value]
    expected = {
        (task, arm, "distilbert", seed)
        for task in tasks
        for arm in ARMS
        for seed in seeds
    }
    summaries = {}
    details = {}
    failures = []

    for worker in root.rglob("worker.json"):
        document = load_json(worker)
        if not document.get("complete") or document.get("errors"):
            failures.append({"worker": str(worker), "errors": document.get("errors")})

    for path in root.rglob("summary.json"):
        try:
            summary, detail = verify_summary(path)
            cfg = summary["config"]
            key = (cfg["task"], cfg["arm"], cfg["backbone"], int(summary["seed"]))
            if key in summaries:
                raise AssertionError("duplicate summary " + repr(key))
            summaries[key] = summary
            details[key] = detail
        except Exception as exc:
            failures.append({"summary": str(path), "error": str(exc)})

    missing = sorted(expected - set(summaries))
    unexpected = sorted(set(summaries) - expected)
    dataset_hashes = sorted({summary["dataset_sha256"] for summary in summaries.values()})
    holdout_hashes = {
        task: sorted(
            {
                summary["secondary_holdout_sha256"]
                for key, summary in summaries.items()
                if key[0] == task
            }
        )
        for task in tasks
    }
    if len(dataset_hashes) != 1:
        failures.append({"error": "dataset hashes differ", "hashes": dataset_hashes})
    if any(len(values) != 1 for values in holdout_hashes.values()):
        failures.append({"error": "holdout hashes differ", "hashes": holdout_hashes})

    groups = {}
    for key, summary in summaries.items():
        task, arm, backbone, seed = key
        group = groups.setdefault(
            (task, arm, backbone),
            {
                "reg": [],
                "hold": [],
                "p50": [],
                "p95": [],
                "rss": [],
                "recall": [],
                "precision": [],
                "clause": defaultdict(Counter),
                "risk": defaultdict(Counter),
                "conf": Counter(),
            },
        )
        group["reg"].append(derived(metrics_for(summary, "regression_metrics", task)))
        group["hold"].append(derived(metrics_for(summary, "secondary_holdout_metrics", task)))
        group["p50"].append(summary["generation_latency"]["p50_ms"])
        group["p95"].append(summary["generation_latency"]["p95_ms"])
        group["rss"].append(summary["memory"].get("rss_after_evaluation_kib"))
        clause_metrics = summary["secondary_holdout_clause_metrics"]
        if clause_metrics.get("target_recall") is not None:
            group["recall"].append(clause_metrics["target_recall"])
        if clause_metrics.get("target_precision") is not None:
            group["precision"].append(clause_metrics["target_precision"])
        detail = details[key]
        for count, values in detail["answerable_by_clause_count"].items():
            group["clause"][count].update(values)
        for status, values in detail["risk_by_reference_status"].items():
            group["risk"][status].update(values)
        for confusion in detail["semantic_confusions"]:
            group["conf"][(confusion["reference"], confusion["predicted"])] += confusion["count"]

    aggregates = []
    for (task, arm, backbone), values in sorted(groups.items()):
        aggregates.append(
            {
                "task": task,
                "arm": arm,
                "backbone": backbone,
                "runs": len(values["hold"]),
                "mean_regression_accuracy": statistics.mean(item["accuracy"] for item in values["reg"]),
                "mean_secondary_holdout_accuracy": statistics.mean(item["accuracy"] for item in values["hold"]),
                "mean_answerable_holdout_accuracy": statistics.mean(item["answerable_accuracy"] for item in values["hold"]),
                "mean_risk_holdout_accuracy": statistics.mean(item["risk_accuracy"] for item in values["hold"]),
                "mean_incorrect_publication_rate": statistics.mean(item["incorrect_publication_rate"] for item in values["hold"]),
                "mean_target_recall": statistics.mean(values["recall"]) if values["recall"] else None,
                "mean_target_precision": statistics.mean(values["precision"]) if values["precision"] else None,
                "mean_p50_ms": statistics.mean(values["p50"]),
                "mean_p95_ms": statistics.mean(values["p95"]),
                "mean_rss_kib": statistics.mean(item for item in values["rss"] if item is not None),
                "answerable_by_clause_count": {
                    count: dict(counter) for count, counter in sorted(values["clause"].items())
                },
                "risk_by_reference_status": {
                    status: dict(counter) for status, counter in sorted(values["risk"].items())
                },
                "semantic_confusions": [
                    {"reference": reference, "predicted": predicted, "count": count}
                    for (reference, predicted), count in values["conf"].most_common(12)
                ],
            }
        )

    complete = not failures and not missing and not unexpected and len(summaries) == len(expected)
    report = {
        "format": "gdm52-batch-report-v1",
        "source_commit": os.environ.get("GITHUB_SHA", "unrecorded"),
        "complete": complete,
        "verified": complete,
        "expected_results": len(expected),
        "verified_results": len(summaries),
        "missing_results": missing,
        "unexpected_results": unexpected,
        "failed_results": failures,
        "dataset_hashes": dataset_hashes,
        "secondary_holdout_hashes": holdout_hashes,
        "aggregates": aggregates,
        "hypotheses": [
            "GDM51 showed that a clause-local ambiguity discriminator improves answerable recall and AMBIGUOUS detection, but joint threshold search still trades accepted, NO_MATCH and AMBIGUOUS outcomes through one objective.",
            "Calibrating NONE first on accepted-versus-NO_MATCH validation cases and ambiguity second on accepted-versus-AMBIGUOUS cases may recover open-set risk without undoing GDM51 multi-clause recall gains.",
            "A structured ambiguity head may benefit more from status-specific calibration because its GDM51 risk/precision tradeoff was already stronger than the unconstrained MLP.",
            "A validation-only accepted-recall noninferiority constraint can test whether risk-first sequential calibration moves the Pareto frontier without recreating the false-reject collapse seen before GDM51.",
        ],
        "interpretation_limits": [
            "Frozen DistilBERT plus learned feature-space capability and ambiguity heads; no transformer fine-tuning.",
            "GDM52 changes validation-only threshold selection, not the capability/ambiguity training objective or frozen-feature quantum.",
            "The ambiguity discriminator remains clause-local and request status remains deterministic NO_MATCH-first aggregation.",
            "Schema generation remains catalog projection plus deterministic SDL/Federation realization, not unconstrained schema invention.",
            "The Shipment/Invoice secondary holdout is synthetic and not enterprise OOD.",
        ],
    }
    (root / "BATCH_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
