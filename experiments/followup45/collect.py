"""Independent evidence gate for the GDM45 measured follow-up."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics

import torch

from experiments.measured.models import state_hash
from .run import ARMS, FORMAT


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def prediction_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def metric_accuracy(summary, field, task):
    metrics = summary[field]
    if task not in metrics or "accuracy" not in metrics[task]:
        raise AssertionError(f"missing {field}.{task}.accuracy")
    return float(metrics[task]["accuracy"])


def verify_summary(summary_path):
    directory = summary_path.parent
    summary = load_json(summary_path)
    errors = []
    if summary.get("format") != FORMAT:
        errors.append("wrong format")
    if summary.get("evidence_kind") != "trained-feature-model":
        errors.append("not trained-feature-model")
    config = summary.get("config", {})
    task = config.get("task")
    if config.get("model_scope") != "frozen pretrained encoder + learned feature adapter/heads":
        errors.append("model scope missing or dishonest")
    training = summary.get("training", {})
    if training.get("optimizer_updates", 0) <= 0 or training.get("selected_optimizer_updates", 0) <= 0:
        errors.append("no optimizer updates")
    if training.get("initial_state_hash") == training.get("selected_state_hash"):
        errors.append("checkpoint state did not change")
    if training.get("independent_backbone_finetuning") is not False:
        errors.append("backbone fine-tuning scope mismatch")
    if training.get("adapter_location") != "frozen-embedding feature space":
        errors.append("feature adapter location missing")
    initial, selected = directory / "initial.pt", directory / "selected.pt"
    if not initial.exists() or not selected.exists():
        errors.append("checkpoint file missing")
    else:
        if file_hash(initial) == file_hash(selected):
            errors.append("checkpoint files identical")
        try:
            initial_obj = torch.load(initial, map_location="cpu", weights_only=True)
            selected_obj = torch.load(selected, map_location="cpu", weights_only=True)
            if state_hash(initial_obj["state"]) != training.get("initial_state_hash"):
                errors.append("initial checkpoint state hash mismatch")
            if state_hash(selected_obj["state"]) != training.get("selected_state_hash"):
                errors.append("selected checkpoint state hash mismatch")
        except Exception as exc:
            errors.append("checkpoint load failure: " + type(exc).__name__)
    for name, expected in summary.get("file_hashes", {}).items():
        path = directory / name
        if not path.exists() or file_hash(path) != expected:
            errors.append("file hash mismatch: " + name)
    for split, expected_count in [("regression", summary.get("regression_examples")),
                                  ("holdout", summary.get("secondary_holdout_examples"))]:
        path = directory / f"predictions-{split}.jsonl"
        if not path.exists():
            errors.append(f"missing {split} predictions")
            continue
        rows = prediction_rows(path)
        if len(rows) != expected_count or not rows:
            errors.append(f"bad {split} prediction count")
        raw_correct = 0
        for row in rows:
            if "reference" not in row or "prediction" not in row or "judgment" not in row:
                errors.append(f"malformed {split} prediction")
                break
            judgment = row["judgment"]
            raw_correct += int(bool(judgment.get("request_correct")))
            if row["public"]["task"] == "operation" and row["reference"]["status"] == "accepted" and row["prediction"]["status"] == "accepted":
                if not isinstance(judgment.get("graphql_valid"), bool) or len(judgment.get("response_matches", [])) != 3:
                    errors.append("operation lacks real GraphQL/fixture evaluation")
                    break
            if row["public"]["task"] == "schema" and row["prediction"]["status"] == "accepted":
                composition = judgment.get("composition")
                if not isinstance(composition, dict) or not isinstance(composition.get("success"), bool):
                    errors.append("schema accepted prediction lacks Rover composition result")
                    break
        if rows:
            field = "regression_metrics" if split == "regression" else "secondary_holdout_metrics"
            try:
                reported = metric_accuracy(summary, field, task)
                reproduced = raw_correct / len(rows)
                if abs(reported - reproduced) > 1e-12:
                    errors.append(f"{split} metric does not reproduce from predictions")
            except Exception as exc:
                errors.append(f"{split} metric unavailable: {exc}")
    timings = load_json(directory / "timings.json") if (directory / "timings.json").exists() else {}
    samples = timings.get("generation_ms", [])
    if len(samples) < 10 or any((not isinstance(x, (int, float)) or x <= 0) for x in samples):
        errors.append("insufficient measured timing samples")
    encoder = summary.get("encoder", {})
    if config.get("backbone") != "hash" and (encoder.get("frozen") is not True or not encoder.get("revision")):
        errors.append("unversioned or non-frozen encoder evidence")
    if errors:
        raise AssertionError(summary_path.as_posix() + ": " + "; ".join(sorted(set(errors))))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--seeds", default="4501,4502")
    parser.add_argument("--backbones", default="distilbert,mmbert")
    parser.add_argument("--tasks", default="operation,schema")
    args = parser.parse_args()
    root = Path(args.root)
    seeds = [int(x) for x in args.seeds.split(",") if x]
    backbones = [x for x in args.backbones.split(",") if x]
    tasks = [x for x in args.tasks.split(",") if x]
    expected = {(task, arm, backbone, seed) for task in tasks for arm in ARMS for backbone in backbones for seed in seeds}
    summaries, failures = {}, []
    for worker in root.rglob("worker.json"):
        data = load_json(worker)
        if not data.get("complete") or data.get("errors"):
            failures.append({"worker": str(worker), "errors": data.get("errors")})
    for path in root.rglob("summary.json"):
        try:
            summary = verify_summary(path)
            cfg = summary["config"]
            key = (cfg["task"], cfg["arm"], cfg["backbone"], int(summary["seed"]))
            if key in summaries:
                raise AssertionError("duplicate summary key: " + repr(key))
            summaries[key] = summary
        except Exception as exc:
            failures.append({"summary": str(path), "error": str(exc)})
    missing = sorted(expected - set(summaries))
    unexpected = sorted(set(summaries) - expected)
    dataset_hashes = sorted({s["dataset_sha256"] for s in summaries.values()})
    holdout_hashes_by_task = {}
    for task in tasks:
        holdout_hashes_by_task[task] = sorted({s["secondary_holdout_sha256"] for k, s in summaries.items() if k[0] == task})
    if len(dataset_hashes) != 1:
        failures.append({"error": "dataset hashes differ", "hashes": dataset_hashes})
    if any(len(v) != 1 for v in holdout_hashes_by_task.values()):
        failures.append({"error": "secondary holdout hashes differ within task", "hashes": holdout_hashes_by_task})

    groups = {}
    for (task, arm, backbone, seed), summary in summaries.items():
        key = (task, arm, backbone)
        group = groups.setdefault(key, {"regression_accuracy": [], "holdout_accuracy": [],
                                       "p50_ms": [], "p95_ms": [], "rss_kib": []})
        group["regression_accuracy"].append(metric_accuracy(summary, "regression_metrics", task))
        group["holdout_accuracy"].append(metric_accuracy(summary, "secondary_holdout_metrics", task))
        group["p50_ms"].append(summary["generation_latency"]["p50_ms"])
        group["p95_ms"].append(summary["generation_latency"]["p95_ms"])
        rss = summary["memory"].get("rss_after_evaluation_kib")
        if rss is not None:
            group["rss_kib"].append(rss)
    aggregate_rows = []
    for (task, arm, backbone), values in sorted(groups.items()):
        aggregate_rows.append({
            "task": task, "arm": arm, "backbone": backbone,
            "runs": len(values["holdout_accuracy"]),
            "mean_regression_accuracy": statistics.mean(values["regression_accuracy"]),
            "mean_secondary_holdout_accuracy": statistics.mean(values["holdout_accuracy"]),
            "mean_p50_ms": statistics.mean(values["p50_ms"]),
            "mean_p95_ms": statistics.mean(values["p95_ms"]),
            "mean_rss_kib": statistics.mean(values["rss_kib"]) if values["rss_kib"] else None,
        })
    report = {
        "format": "gdm45-batch-report-v1",
        "source_commit": os.environ.get("GITHUB_SHA", "unrecorded"),
        "complete": not failures and not missing and not unexpected and len(summaries) == len(expected),
        "verified": not failures and not missing and not unexpected and len(summaries) == len(expected),
        "expected_results": len(expected), "verified_results": len(summaries),
        "missing_results": missing, "unexpected_results": unexpected,
        "failed_results": failures, "dataset_hashes": dataset_hashes,
        "secondary_holdout_hashes": holdout_hashes_by_task,
        "aggregates": aggregate_rows,
        "interpretation_limits": [
            "All learned models use frozen pretrained encoders and feature-space adapters/heads; no transformer backbone is fine-tuned.",
            "Schema generation remains catalog projection with deterministic SDL and Federation realization, not unconstrained schema invention.",
            "The secondary holdout is synthetic and not a human-authored out-of-distribution benchmark.",
            "Memory is current process RSS after evaluation plus a worker high-water mark; one worker reuses a backbone across arms.",
        ],
    }
    output = root / "BATCH_REPORT.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
