"""GDM52: status-specific calibration over canonical GDM51 ambiguity heads.

Canonical GDM51 verified that a separate clause-local ambiguity discriminator can
recover answerable multi-clause recall and materially improve AMBIGUOUS detection,
but the single joint threshold search still trades NO_MATCH rejection, ambiguity
rejection and accepted recall against one shared selective-H-mean objective. GDM52
keeps the frozen DistilBERT encoder, explicit-NONE capability model, ambiguity-head
architectures and numerical execution contract fixed. It asks whether calibrating the
upstream NONE gate and downstream ambiguity gate on their own status-specific
validation slices improves the risk/recall Pareto frontier.

No transformer fine-tuning occurs. Learned modules operate over frozen feature-space
representations. Schema generation remains catalog projection plus deterministic
SDL/Federation realization. All compilation, tests, training, validation and
benchmarking belong in GitHub Actions.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import resource
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from experiments.measured.evidence import Compositor, aggregate, file_hash, timing, write_json
from experiments.followup46 import run as g46
from experiments.followup46.run import corrected_dataset, current_rss_kib, validate_emission
from experiments.followup47 import run as g47
from experiments.followup48 import run as g48
from experiments.followup49 import run as g49
from experiments.followup50 import run as g50
from experiments.followup51 import run as g51

FORMAT = "gdm52-measured-v1"
SEEDS_DEFAULT = "5201,5202"

ARMS = {
    "joint-mlp-control": {
        "family": "explicit-none-plus-ambiguity",
        "head": "mlp",
        "structured": False,
        "ambiguous_weight": 1.0,
        "hard_negative": False,
        "calibration_strategy": "joint-selective-hmean",
    },
    "joint-structured-control": {
        "family": "explicit-none-plus-ambiguity",
        "head": "mlp",
        "structured": True,
        "ambiguous_weight": 1.0,
        "hard_negative": False,
        "calibration_strategy": "joint-selective-hmean",
    },
    "sequential-mlp": {
        "family": "explicit-none-plus-ambiguity",
        "head": "mlp",
        "structured": False,
        "ambiguous_weight": 1.0,
        "hard_negative": False,
        "calibration_strategy": "sequential-status-specific",
    },
    "sequential-structured": {
        "family": "explicit-none-plus-ambiguity",
        "head": "mlp",
        "structured": True,
        "ambiguous_weight": 1.0,
        "hard_negative": False,
        "calibration_strategy": "sequential-status-specific",
    },
    "sequential-structured-noninferior": {
        "family": "explicit-none-plus-ambiguity",
        "head": "mlp",
        "structured": True,
        "ambiguous_weight": 1.0,
        "hard_negative": False,
        "calibration_strategy": "sequential-status-specific-noninferior",
    },
}

CALIBRATION_LANGUAGE = [
    "primary interface heading for this entity",
    "timestamp when the entity was first committed to durable storage",
    "timestamp of the entity's most recent persisted rewrite",
    "whole-number score stored on each review",
    "display names of people who created the reviews",
    "display names of people who moderated the reviews",
    "registered supplier organization name",
    "persistent identifiers of people who created the reviews",
]
CALIBRATION_UNSUPPORTED = [
    "physical storage cage assigned to the entity",
    "explanation attached to a compliance waiver",
    "freight classification code for the entity",
    "flag indicating the entity is under legal hold",
]
CALIBRATION_AMBIGUOUS = [
    "review person's identity without specifying creator versus moderator",
    "persistence timestamp without specifying initial versus latest write",
]
HOLDOUT_LANGUAGE = [
    "human-readable heading customers see for this object",
    "instant this object was originally committed to storage",
    "instant this object was last rewritten in storage",
    "integer rating recorded for every review",
    "names of people who wrote the reviews",
    "names of people who moderated the reviews",
    "official supplier business name for this object",
    "stable IDs of people who wrote the reviews",
]
HOLDOUT_UNSUPPORTED = [
    "physical pallet slot where this object is stored",
    "reason an exception was approved for this object",
    "import tariff bucket assigned to this object",
    "flag indicating this object is under litigation preservation",
]
HOLDOUT_AMBIGUOUS = [
    "review person's identity without specifying writer or moderator",
    "storage event time without specifying first commit or latest rewrite",
]

Encoder = g51.Encoder


def calibration_set(task):
    return g50._dataset(
        task,
        [("Portfolio", "portfolio"), ("Ledger", "ledger")],
        CALIBRATION_LANGUAGE,
        CALIBRATION_UNSUPPORTED,
        CALIBRATION_AMBIGUOUS,
        "gdm52-calibration",
    )


def fresh_holdout(task):
    return g50._dataset(
        task,
        [("Shipment", "shipment"), ("Invoice", "invoice")],
        HOLDOUT_LANGUAGE,
        HOLDOUT_UNSUPPORTED,
        HOLDOUT_AMBIGUOUS,
        "gdm52-holdout",
    )


def _hmean(a: float, b: float) -> float:
    return 2.0 * a * b / (a + b) if a + b else 0.0


def _none_gate_metrics(records, threshold):
    """Score NONE as NO_MATCH versus all non-NO_MATCH statuses.

    AMBIGUOUS requests must survive the upstream NONE gate or the downstream
    ambiguity discriminator never gets a chance to classify them. The original
    GDM52 preflight exposed that hierarchy error before measured training began.
    """
    accepted_total = accepted_correct = 0
    ambiguous_total = ambiguous_preserved = 0
    non_nomatch_total = non_nomatch_correct = 0
    nomatch_total = nomatch_correct = false_passes = 0
    for row in records:
        ref = row["reference"]
        predicted_nomatch = any(e["none_margin"] >= threshold for e in row["evidence"])
        if ref["status"] == "NO_MATCH":
            nomatch_total += 1
            nomatch_correct += int(predicted_nomatch)
            false_passes += int(not predicted_nomatch)
        else:
            non_nomatch_total += 1
            non_nomatch_correct += int(not predicted_nomatch)
            if ref["status"] == "accepted":
                accepted_total += 1
                accepted_correct += int(not predicted_nomatch)
            elif ref["status"] == "AMBIGUOUS":
                ambiguous_total += 1
                ambiguous_preserved += int(not predicted_nomatch)
    accepted_recall = accepted_correct / accepted_total if accepted_total else 0.0
    ambiguous_recall = ambiguous_preserved / ambiguous_total if ambiguous_total else 0.0
    pass_recall = non_nomatch_correct / non_nomatch_total if non_nomatch_total else 0.0
    nomatch_recall = nomatch_correct / nomatch_total if nomatch_total else 0.0
    total = non_nomatch_total + nomatch_total
    return {
        "accepted_status_recall": accepted_recall,
        "ambiguity_preservation_recall": ambiguous_recall,
        "non_nomatch_recall": pass_recall,
        "nomatch_recall": nomatch_recall,
        "binary_hmean": _hmean(pass_recall, nomatch_recall),
        "incorrect_publication_rate": false_passes / total if total else 0.0,
        "accepted_examples": accepted_total,
        "ambiguous_examples": ambiguous_total,
        "non_nomatch_examples": non_nomatch_total,
        "nomatch_examples": nomatch_total,
    }


def _ambiguity_gate_metrics(records, none_threshold, ambiguity_threshold):
    """Score ambiguity only for requests that reach the downstream gate."""
    accepted_total = accepted_correct = 0
    ambiguity_total = ambiguity_correct = incorrect = 0
    upstream_blocked_accepted = upstream_blocked_ambiguous = 0
    for row in records:
        ref = row["reference"]
        if ref["status"] not in {"accepted", "AMBIGUOUS"}:
            continue
        blocked = any(e["none_margin"] >= none_threshold for e in row["evidence"])
        if blocked:
            if ref["status"] == "accepted":
                upstream_blocked_accepted += 1
            else:
                upstream_blocked_ambiguous += 1
            continue
        predicted_ambiguous = any(
            e["ambiguity_probability"] >= ambiguity_threshold for e in row["evidence"]
        )
        if ref["status"] == "accepted":
            accepted_total += 1
            accepted_correct += int(not predicted_ambiguous)
        else:
            ambiguity_total += 1
            ambiguity_correct += int(predicted_ambiguous)
            incorrect += int(not predicted_ambiguous)
    accepted_recall = accepted_correct / accepted_total if accepted_total else 0.0
    ambiguity_recall = ambiguity_correct / ambiguity_total if ambiguity_total else 0.0
    total = accepted_total + ambiguity_total
    return {
        "accepted_status_recall": accepted_recall,
        "ambiguity_recall": ambiguity_recall,
        "binary_hmean": _hmean(accepted_recall, ambiguity_recall),
        "incorrect_publication_rate": incorrect / total if total else 0.0,
        "accepted_examples": accepted_total,
        "ambiguity_examples": ambiguity_total,
        "upstream_blocked_accepted": upstream_blocked_accepted,
        "upstream_blocked_ambiguous": upstream_blocked_ambiguous,
    }


def _threshold_values(records, field):
    vals = [e[field] for row in records for e in row["evidence"]]
    if not vals:
        raise AssertionError("empty calibration evidence for " + field)
    return g51._threshold_candidates(vals)


def _select_none_threshold(records, accepted_floor=None, risk_first=False):
    best = None
    for threshold in _threshold_values(records, "none_margin"):
        metrics = _none_gate_metrics(records, threshold)
        if accepted_floor is not None and metrics["accepted_status_recall"] + 1e-12 < accepted_floor:
            continue
        if risk_first:
            key = (
                metrics["nomatch_recall"],
                metrics["binary_hmean"],
                metrics["non_nomatch_recall"],
                metrics["ambiguity_preservation_recall"],
                metrics["accepted_status_recall"],
                -metrics["incorrect_publication_rate"],
            )
        else:
            key = (
                metrics["binary_hmean"],
                metrics["nomatch_recall"],
                metrics["non_nomatch_recall"],
                metrics["ambiguity_preservation_recall"],
                metrics["accepted_status_recall"],
                -metrics["incorrect_publication_rate"],
            )
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)
    if best is None:
        raise RuntimeError("no NONE threshold satisfies the accepted-recall constraint")
    return best[1], best[2]


def _select_ambiguity_threshold(records, none_threshold, accepted_floor=None, risk_first=False):
    best = None
    for threshold in _threshold_values(records, "ambiguity_probability"):
        metrics = _ambiguity_gate_metrics(records, none_threshold, threshold)
        final_metrics = g51._cal_metrics(records, none_threshold, threshold, True)
        if accepted_floor is not None and final_metrics["accepted_status_recall"] + 1e-12 < accepted_floor:
            continue
        if risk_first:
            key = (
                metrics["ambiguity_recall"],
                metrics["binary_hmean"],
                final_metrics["accepted_status_recall"],
                -final_metrics["incorrect_publication_rate"],
            )
        else:
            key = (
                metrics["binary_hmean"],
                metrics["ambiguity_recall"],
                final_metrics["accepted_status_recall"],
                -final_metrics["incorrect_publication_rate"],
            )
        if best is None or key > best[0]:
            evidence = dict(metrics)
            evidence["global_accepted_status_recall"] = final_metrics["accepted_status_recall"]
            evidence["global_exact_accuracy"] = final_metrics["exact_accuracy"]
            best = (key, float(threshold), evidence)
    if best is None:
        raise RuntimeError("no ambiguity threshold satisfies the accepted-recall constraint")
    return best[1], best[2]


def calibrate_records(records, strategy):
    """Validation-only hierarchical status calibration over clause evidence."""
    if strategy not in {
        "sequential-status-specific",
        "sequential-status-specific-noninferior",
    }:
        raise ValueError(strategy)

    baseline_metrics = None
    floor = None
    risk_first = strategy.endswith("noninferior")
    if risk_first:
        nvals = _threshold_values(records, "none_margin")
        avals = _threshold_values(records, "ambiguity_probability")
        best = None
        for nt in nvals:
            for at in avals:
                metrics = g51._cal_metrics(records, nt, at, True)
                key = (
                    metrics["selective_hmean"],
                    metrics["exact_accuracy"],
                    -metrics["incorrect_publication_rate"],
                    metrics["accepted_status_recall"],
                )
                if best is None or key > best[0]:
                    best = (key, float(nt), float(at), metrics)
        if best is None:
            raise AssertionError("joint baseline calibration produced no candidate")
        baseline_metrics = best[3]
        floor = float(baseline_metrics["accepted_status_recall"])

    nt, none_metrics = _select_none_threshold(records, accepted_floor=floor, risk_first=risk_first)
    at, ambiguity_metrics = _select_ambiguity_threshold(
        records,
        nt,
        accepted_floor=floor,
        risk_first=risk_first,
    )
    final_metrics = g51._cal_metrics(records, nt, at, True)
    if floor is not None and final_metrics["accepted_status_recall"] + 1e-12 < floor:
        raise AssertionError("noninferiority accepted-recall floor was not preserved")

    calibration = {
        "mode": "explicit-none-plus-ambiguity-discriminator",
        "none_margin_threshold": nt,
        "ambiguity_probability_threshold": at,
        "selection_split": "validation",
        "calibration_scope": "expanded-validation-only",
        "threshold_selection": strategy,
        "hierarchy_contract": "NONE gate calibrated as NO_MATCH vs non-NO_MATCH (accepted + AMBIGUOUS); ambiguity gate calibrated only on requests surviving NONE",
        "none_gate_validation_metrics": none_metrics,
        "ambiguity_gate_validation_metrics": ambiguity_metrics,
    }
    if baseline_metrics is not None:
        calibration["accepted_status_recall_floor"] = floor
        calibration["accepted_status_recall_floor_source"] = "same-record GDM51 joint calibration"
        calibration["joint_baseline_validation_metrics"] = baseline_metrics
    return calibration, final_metrics


def calibrate_model(bundle, encoder, arm, rows, refs):
    strategy = arm["calibration_strategy"]
    if strategy == "joint-selective-hmean":
        calibration, metrics = g51.calibrate_model(bundle, encoder, arm, rows, refs)
        calibration = dict(calibration)
        calibration["threshold_selection"] = strategy
        return calibration, metrics
    records = [
        {
            "reference": refs[row["id"]],
            "evidence": g51.evidence_for_request(bundle, encoder, row, arm),
        }
        for row in rows
    ]
    return calibrate_records(records, strategy)


def run_one(arm_name, task, seed, encoder, public, refs, data_report, compositor, root, epochs):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    arm = ARMS[arm_name]
    directory = root / f"{task}-{arm_name}-{encoder.name}-seed{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    config = {
        "id": directory.name,
        "arm": arm_name,
        "task": task,
        "backbone": encoder.name,
        **arm,
        "model_scope": "frozen pretrained encoder + learned feature-space adapter/head",
        "schema_scope": "catalog projection + deterministic SDL/Federation realization" if task == "schema" else None,
        "research_question": "can hierarchical status-specific validation calibration preserve GDM51 answerable recall while recovering NO_MATCH and AMBIGUOUS risk accuracy",
    }
    write_json(directory / "config.json", config)
    rss_before = current_rss_kib()
    started = time.perf_counter_ns()
    cal_rows, cal_refs = calibration_set(task)

    bundle = g51.Bundle(encoder.dim * 5 + 4, g51._ambiguity_dim(arm["structured"]), arm["head"])
    receipt = g51.train_model(bundle, encoder, arm, task, public, refs, seed, epochs, directory)
    val = [row for row in public["validation"] if row["task"] == task]
    vrefs = {row["id"]: refs["validation"][row["id"]] for row in val}
    rows = val + cal_rows
    merged = {**vrefs, **cal_refs}
    calibration, calibration_metrics = calibrate_model(bundle, encoder, arm, rows, merged)
    write_json(
        directory / "calibration.json",
        {
            "selected": calibration,
            "validation_metrics": calibration_metrics,
            "status_optimizer_updates": 0,
        },
    )
    receipt.update(
        {
            "gdm51_family": arm["family"],
            "gdm52_family": "status-specific-calibration",
            "gdm52_calibration_strategy": arm["calibration_strategy"],
            "selected_calibration": calibration,
            "selected_calibration_validation_metrics": calibration_metrics,
            "calibration_optimizer_updates": 0,
        }
    )
    training_doc = json.loads((directory / "training.json").read_text())
    training_doc.update(receipt)
    write_json(directory / "training.json", training_doc)
    infer = lambda item, online=False: g51.infer_model(
        bundle, encoder, item, arm, calibration, online=online
    )

    training_ms = (time.perf_counter_ns() - started) / 1e6
    base = [row for row in public["test"] if row["task"] == task]
    base_refs = {row["id"]: refs["test"][row["id"]] for row in base}
    regression_rows = list(base)
    regression_refs = dict(base_refs)
    for mod in (g46, g47, g48, g49, g50, g51):
        old_rows, old_refs = mod.fresh_holdout(task)
        regression_rows += old_rows
        regression_refs.update(old_refs)
    hold_rows, hold_refs = fresh_holdout(task)

    regression = g49.evaluate(
        regression_rows,
        regression_refs,
        lambda item: infer(item, False),
        compositor,
        directory / "predictions-regression.jsonl",
    )
    holdout = g49.evaluate(
        hold_rows,
        hold_refs,
        lambda item: infer(item, False),
        compositor,
        directory / "predictions-holdout.jsonl",
    )

    bench = hold_rows[:12]
    for row in bench[:2]:
        validate_emission(row, infer(row, True))
    samples = []
    before_calls = encoder.calls
    for _ in range(2):
        for row in bench:
            started_ns = time.perf_counter_ns()
            pred = infer(row, True)
            validate_emission(row, pred)
            samples.append((time.perf_counter_ns() - started_ns) / 1e6)
    calls = encoder.calls - before_calls
    write_json(
        directory / "timings.json",
        {
            "generation_ms": samples,
            "warmup_requests": min(2, len(bench)),
            "encoder_calls_measured": calls,
            "scope": "single-request fresh clause query encoding + cached catalog/NONE vectors + listwise capability + ambiguity discrimination + generation/validation; excludes Rover/backend",
        },
    )

    gc.collect()
    rss_after = current_rss_kib()
    files = [
        "config.json",
        "training.json",
        "calibration.json",
        "initial.pt",
        "selected.pt",
        "predictions-regression.jsonl",
        "predictions-holdout.jsonl",
        "timings.json",
    ]
    summary = {
        "format": FORMAT,
        "evidence_kind": "trained-feature-model",
        "config": config,
        "source_commit": os.environ.get("GITHUB_SHA", "unrecorded"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "seed": seed,
        "encoder": encoder.meta,
        "training": receipt,
        "training_and_selection_ms": training_ms,
        "dataset_sha256": data_report["dataset_sha256"],
        "dataset_scope": data_report["scope"],
        "calibration_only_sha256": g50.sha({"rows": cal_rows, "references": cal_refs}),
        "calibration_scope": "disjoint Portfolio/Ledger calibration-only cases; never used for optimizer updates or checkpoint selection",
        "secondary_holdout_sha256": g50.sha({"rows": hold_rows, "references": hold_refs}),
        "secondary_holdout_scope": "new GDM52 Shipment/Invoice synthetic holdout; never used for optimizer/checkpoint/calibration selection",
        "regression_scope": "corrected public test plus inspected GDM46-GDM51 holdouts",
        "regression_examples": len(regression),
        "secondary_holdout_examples": len(holdout),
        "regression_metrics": aggregate(regression),
        "secondary_holdout_metrics": aggregate(holdout),
        "regression_clause_metrics": g49.clause_metrics(regression),
        "secondary_holdout_clause_metrics": g49.clause_metrics(holdout),
        "generation_latency": timing(samples),
        "latency_scope": "fresh request-clause encoding + learned feature-space scoring + deterministic generation/validation; catalog/NONE embeddings cached",
        "memory": {
            "rss_before_training_kib": rss_before,
            "rss_after_evaluation_kib": rss_after,
            "peak_worker_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "scope": "current process RSS plus worker high-water; worker reuses frozen backbone across arms",
        },
        "file_hashes": {name: file_hash(directory / name) for name in files},
    }
    write_json(directory / "summary.json", summary)
    print(
        json.dumps(
            {
                "event": "completed_arm",
                "config": config["id"],
                "holdout": summary["secondary_holdout_metrics"][task],
                "clauses": summary["secondary_holdout_clause_metrics"],
                "p50_ms": summary["generation_latency"]["p50_ms"],
            }
        ),
        flush=True,
    )
    return directory.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="distilbert", choices=["distilbert", "hash"])
    parser.add_argument("--task", required=True, choices=["operation", "schema"])
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--seeds", default=SEEDS_DEFAULT)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--output", default="artifacts/results")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise RuntimeError("Project execution belongs in GitHub Actions")

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    public, refs, report = corrected_dataset(root / "dataset")
    arms = [name for name in args.arms.split(",") if name]
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise ValueError(sorted(unknown))
    seeds = [int(seed) for seed in args.seeds.split(",") if seed]
    if args.smoke:
        arms = ["sequential-structured"]
        seeds = [5201]
        args.epochs = 2

    write_json(
        root / "plan.json",
        {
            "backbone": args.backbone,
            "task": args.task,
            "arms": arms,
            "seeds": seeds,
            "epochs": args.epochs,
        },
    )
    encoder = Encoder(args.backbone)
    compositor = Compositor(root / "composition")
    control = next(
        row
        for row in public["train"]
        if row["task"] == "schema" and refs["train"][row["id"]]["status"] == "accepted"
    )
    paths = [option["id"] for option in control["catalog"]["options"]]
    if not compositor.compose(control, paths)["success"]:
        raise RuntimeError("Known-valid Federation composition control failed")

    completed = []
    errors = []
    for arm in arms:
        for seed in seeds:
            try:
                completed.append(
                    run_one(
                        arm,
                        args.task,
                        seed,
                        encoder,
                        public,
                        refs,
                        report,
                        compositor,
                        root,
                        args.epochs,
                    )
                )
            except Exception:
                err = {
                    "arm": arm,
                    "task": args.task,
                    "seed": seed,
                    "traceback": traceback.format_exc(),
                }
                errors.append(err)
                print(json.dumps({"event": "failed_arm", **err}), flush=True)
    write_json(
        root / "worker.json",
        {
            "complete": not errors,
            "completed": completed,
            "errors": errors,
            "expected": len(arms) * len(seeds),
            "encoder": encoder.meta,
        },
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
