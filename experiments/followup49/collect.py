"""Independent evidence gate for GDM49."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics

import torch

from experiments.measured.evidence import timing
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
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def metrics_for(summary, field, task):
    metrics = summary[field]
    if task not in metrics:
        raise AssertionError(f"missing {field}.{task}")
    return metrics[task]


def derived(metrics):
    examples = int(metrics["examples"]); answerable = int(metrics["answerable"]); risks = int(metrics["risk_cases"])
    return {
        "accuracy": float(metrics["accuracy"]),
        "answerable_accuracy": int(metrics["answerable_correct"]) / answerable if answerable else None,
        "risk_accuracy": int(metrics["risk_correct"]) / risks if risks else None,
        "incorrect_publication_rate": int(metrics["incorrect_publications"]) / examples if examples else None,
        "accepted_rate": int(metrics["accepted"]) / examples if examples else None,
    }


def recompute_clause_metrics(rows):
    tp = gold = predicted = 0
    for row in rows:
        if row["reference"]["status"] != "accepted":
            continue
        g = set(row["reference"]["paths"])
        p = set(row["prediction"].get("selected", [])) if row["prediction"]["status"] == "accepted" else set()
        tp += len(g & p); gold += len(g); predicted += len(p)
    return {"target_recall": tp / gold if gold else None,
            "target_precision": tp / predicted if predicted else None,
            "targets": gold, "predicted_targets": predicted, "true_positive_targets": tp}


def verify_summary(summary_path):
    directory = summary_path.parent; summary = load_json(summary_path); errors = []
    if summary.get("format") != FORMAT:
        errors.append("wrong format")
    if summary.get("evidence_kind") != "trained-feature-model":
        errors.append("wrong evidence kind")
    cfg = summary.get("config", {}); task = cfg.get("task"); arm_name = cfg.get("arm")
    if arm_name not in ARMS:
        errors.append("unexpected arm")
    expected_arm = ARMS.get(arm_name, {})
    if cfg.get("family") != expected_arm.get("family"):
        errors.append("family mismatch")
    if cfg.get("model_scope") != "frozen pretrained encoder + learned feature-space adapter/head":
        errors.append("dishonest model scope")
    if cfg.get("backbone") not in {"distilbert", "hash"}:
        errors.append("unexpected backbone")
    if task == "schema" and cfg.get("schema_scope") != "catalog projection + deterministic SDL/Federation realization":
        errors.append("schema scope mismatch")

    training = summary.get("training", {})
    if training.get("optimizer_updates", 0) <= 0:
        errors.append("no optimizer updates")
    if training.get("selected_state_hash") == training.get("initial_state_hash"):
        errors.append("selected model did not change")
    if training.get("independent_backbone_finetuning") is not False:
        errors.append("backbone scope mismatch")
    if training.get("adapter_location") != "frozen-embedding feature space":
        errors.append("adapter scope mismatch")
    if training.get("gdm49_family") != cfg.get("family"):
        errors.append("training family mismatch")

    if cfg.get("family") == "binary-matcher":
        if training.get("matcher_optimizer_updates", 0) <= 0:
            errors.append("binary matcher lacks optimizer updates")
        if training.get("selected_matcher_hash") == training.get("initial_matcher_hash"):
            errors.append("binary matcher weights did not change")
        if training.get("risk_training_kind") != "independent binary clause-option match supervision":
            errors.append("binary risk kind mismatch")
    else:
        if training.get("capability_optimizer_updates", 0) <= 0:
            errors.append("listwise control lacks capability updates")
        if training.get("selected_capability_hash") == training.get("initial_capability_hash"):
            errors.append("listwise capability weights did not change")
        if training.get("risk_training_kind") != "validation-only deterministic clause thresholds":
            errors.append("listwise control risk kind mismatch")

    calibration_doc = load_json(directory / "calibration.json") if (directory / "calibration.json").exists() else {}
    calibration = calibration_doc.get("selected", {})
    if cfg.get("family") == "binary-matcher":
        if calibration.get("mode") != "pair-threshold":
            errors.append("binary calibration mode mismatch")
        if calibration.get("selection_split") != "validation":
            errors.append("binary calibration not validation-only")
        threshold = calibration.get("match_threshold")
        if not isinstance(threshold, (int, float)) or not 0 < float(threshold) < 1:
            errors.append("invalid binary match threshold")
        if calibration.get("calibration_scope") != cfg.get("calibration"):
            errors.append("binary calibration scope mismatch")
    else:
        if calibration.get("mode") != "clause-thresholds" or calibration.get("gate") != "clause-hybrid":
            errors.append("control calibration mismatch")
        if calibration.get("selection_split") != "validation":
            errors.append("control calibration not validation-only")

    if summary.get("calibration_only_sha256") == summary.get("secondary_holdout_sha256"):
        errors.append("calibration and holdout hashes collide")
    if "never used for optimizer/checkpoint/calibration selection" not in summary.get("secondary_holdout_scope", ""):
        errors.append("holdout scope missing selection prohibition")

    initial, selected = directory / "initial.pt", directory / "selected.pt"
    if not initial.exists() or not selected.exists():
        errors.append("checkpoint missing")
    else:
        if file_hash(initial) == file_hash(selected):
            errors.append("checkpoint files identical")
        try:
            a = torch.load(initial, map_location="cpu", weights_only=True)
            b = torch.load(selected, map_location="cpu", weights_only=True)
            if state_hash(a["state"]) != training.get("initial_state_hash"):
                errors.append("initial state hash mismatch")
            if state_hash(b["state"]) != training.get("selected_state_hash"):
                errors.append("selected state hash mismatch")
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
        correct = 0
        for row in rows:
            if not all(k in row for k in ["public", "reference", "prediction", "judgment"]):
                errors.append(f"malformed {split} row"); break
            judgment = row["judgment"]; correct += int(bool(judgment.get("request_correct")))
            ref_status = row["reference"]["status"]; pred_status = row["prediction"]["status"]
            prefix = "Return " if row["public"]["task"] == "operation" else "Expose capabilities for "
            if not row["public"]["request"].startswith(prefix):
                errors.append("task prefix violated"); break
            if row["public"]["task"] == "operation" and ref_status == "accepted" and pred_status == "accepted":
                if not isinstance(judgment.get("graphql_valid"), bool) or len(judgment.get("response_matches", [])) != 3:
                    errors.append("operation lacks real GraphQL execution evidence"); break
            if row["public"]["task"] == "schema" and pred_status == "accepted":
                comp = judgment.get("composition")
                if not isinstance(comp, dict) or not isinstance(comp.get("success"), bool):
                    errors.append("schema accepted output lacks Rover evidence"); break
            if cfg.get("family") == "binary-matcher":
                evidence = row["prediction"].get("pair_evidence")
                if not isinstance(evidence, list) or not evidence:
                    errors.append("binary prediction lacks pair evidence"); break
                if any("matched" not in clause or "top" not in clause for clause in evidence):
                    errors.append("malformed pair evidence"); break
            elif "gate_evidence" not in row["prediction"]:
                errors.append("listwise control lacks clause gate evidence"); break
        if rows:
            field = "regression_metrics" if split == "regression" else "secondary_holdout_metrics"
            try:
                reported = float(metrics_for(summary, field, task)["accuracy"])
                if abs(reported - correct / len(rows)) > 1e-12:
                    errors.append(f"{split} accuracy not reproducible")
            except Exception as exc:
                errors.append(f"{split} metrics unavailable: {exc}")
            clause_field = "regression_clause_metrics" if split == "regression" else "secondary_holdout_clause_metrics"
            if recompute_clause_metrics(rows) != summary.get(clause_field):
                errors.append(f"{split} clause metrics not reproducible")

    timing_data = load_json(directory / "timings.json") if (directory / "timings.json").exists() else {}
    samples = timing_data.get("generation_ms", [])
    if len(samples) < 10 or any(not isinstance(x, (int, float)) or x <= 0 for x in samples):
        errors.append("insufficient timing samples")
    elif timing(samples) != summary.get("generation_latency"):
        errors.append("timing summary not reproducible")
    enc = summary.get("encoder", {})
    if cfg.get("backbone") != "hash" and (enc.get("frozen") is not True or not enc.get("revision")):
        errors.append("encoder not frozen/versioned")
    if errors:
        raise AssertionError(summary_path.as_posix() + ": " + "; ".join(sorted(set(errors))))
    return summary


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("root")
    parser.add_argument("--seeds", default="4901,4902"); parser.add_argument("--tasks", default="operation,schema")
    args = parser.parse_args(); root = Path(args.root)
    seeds = [int(x) for x in args.seeds.split(",") if x]; tasks = [x for x in args.tasks.split(",") if x]
    expected = {(task, arm, "distilbert", seed) for task in tasks for arm in ARMS for seed in seeds}
    summaries, failures = {}, []
    for worker in root.rglob("worker.json"):
        data = load_json(worker)
        if not data.get("complete") or data.get("errors"):
            failures.append({"worker": str(worker), "errors": data.get("errors")})
    for path in root.rglob("summary.json"):
        try:
            s = verify_summary(path); cfg = s["config"]
            key = (cfg["task"], cfg["arm"], cfg["backbone"], int(s["seed"]))
            if key in summaries:
                raise AssertionError("duplicate summary key " + repr(key))
            summaries[key] = s
        except Exception as exc:
            failures.append({"summary": str(path), "error": str(exc)})
    missing = sorted(expected - set(summaries)); unexpected = sorted(set(summaries) - expected)
    dataset_hashes = sorted({s["dataset_sha256"] for s in summaries.values()})
    holdout_hashes = {task: sorted({s["secondary_holdout_sha256"] for k, s in summaries.items() if k[0] == task}) for task in tasks}
    if len(dataset_hashes) != 1:
        failures.append({"error": "dataset hashes differ", "hashes": dataset_hashes})
    if any(len(v) != 1 for v in holdout_hashes.values()):
        failures.append({"error": "holdout hashes differ", "hashes": holdout_hashes})

    groups = {}
    for (task, arm, backbone, seed), s in summaries.items():
        key = (task, arm, backbone)
        g = groups.setdefault(key, {"reg": [], "hold": [], "p50": [], "p95": [], "rss": [], "recall": [], "precision": []})
        g["reg"].append(derived(metrics_for(s, "regression_metrics", task)))
        g["hold"].append(derived(metrics_for(s, "secondary_holdout_metrics", task)))
        g["p50"].append(s["generation_latency"]["p50_ms"]); g["p95"].append(s["generation_latency"]["p95_ms"])
        rss = s["memory"].get("rss_after_evaluation_kib")
        if rss is not None:
            g["rss"].append(rss)
        cm = s["secondary_holdout_clause_metrics"]
        if cm.get("target_recall") is not None:
            g["recall"].append(cm["target_recall"])
        if cm.get("target_precision") is not None:
            g["precision"].append(cm["target_precision"])
    aggregates = []
    for (task, arm, backbone), v in sorted(groups.items()):
        aggregates.append({
            "task": task, "arm": arm, "backbone": backbone, "runs": len(v["hold"]),
            "mean_regression_accuracy": statistics.mean(x["accuracy"] for x in v["reg"]),
            "mean_secondary_holdout_accuracy": statistics.mean(x["accuracy"] for x in v["hold"]),
            "mean_answerable_holdout_accuracy": statistics.mean(x["answerable_accuracy"] for x in v["hold"]),
            "mean_risk_holdout_accuracy": statistics.mean(x["risk_accuracy"] for x in v["hold"]),
            "mean_incorrect_publication_rate": statistics.mean(x["incorrect_publication_rate"] for x in v["hold"]),
            "mean_target_recall": statistics.mean(v["recall"]) if v["recall"] else None,
            "mean_target_precision": statistics.mean(v["precision"]) if v["precision"] else None,
            "mean_p50_ms": statistics.mean(v["p50"]), "mean_p95_ms": statistics.mean(v["p95"]),
            "mean_rss_kib": statistics.mean(v["rss"]) if v["rss"] else None,
        })
    report = {
        "format": "gdm49-batch-report-v1", "source_commit": os.environ.get("GITHUB_SHA", "unrecorded"),
        "complete": not failures and not missing and not unexpected and len(summaries) == len(expected),
        "verified": not failures and not missing and not unexpected and len(summaries) == len(expected),
        "expected_results": len(expected), "verified_results": len(summaries),
        "missing_results": missing, "unexpected_results": unexpected, "failed_results": failures,
        "dataset_hashes": dataset_hashes, "secondary_holdout_hashes": holdout_hashes, "aggregates": aggregates,
        "hypotheses": [
            "Listwise softmax is a poor open-set confidence signal because it must rank some catalog option first even when no option matches.",
            "Independent binary clause-option matching can represent zero, one, or multiple supported capabilities directly and should improve NO_MATCH/AMBIGUOUS behavior.",
            "A larger disjoint calibration-only set may stabilize the absolute match threshold relative to the small public validation split.",
            "Hard-negative margin or focal supervision may reduce persistent author-ID/name and adjacent semantic confusions without reintroducing a request-global status model.",
        ],
        "interpretation_limits": [
            "Frozen DistilBERT with learned feature-space pair matcher; no transformer fine-tuning.",
            "Binary matcher status is deterministic from validation-calibrated absolute match thresholds, not a trained request-global status head.",
            "Schema generation remains catalog projection plus deterministic SDL/Federation realization, not unconstrained schema invention.",
            "The GDM49 Shipment/Policy secondary holdout is synthetic and is not enterprise OOD.",
        ],
    }
    (root / "BATCH_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
