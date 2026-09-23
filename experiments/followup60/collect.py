"""Independent evidence gate for GDM60 confidence-scaled ranked ambiguity evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.followup51 import collect as g51collect
from experiments.followup52 import collect as g52collect
from experiments.followup55.run import _ambiguity_family_metrics
from .run import (
    ARMS,
    CANONICAL_GDM59_AUDIT_RUN,
    CANONICAL_GDM59_SOURCE,
    FORMAT,
    TOPK,
    ambiguity_feature_dim,
    _representation_description,
)

EXPECTED_FAMILIES = {"role", "lifecycle-time", "representation", "object-vs-supplier"}
CHANGE_SCOPE = (
    "ambiguity-head rank-preserving request-relative confidence weighting only; fixed top5 breadth, capacity, canonical GDM56 family-balanced curriculum, canonical numerical path and GDM54 5pp rescue arbitration"
)


def load_json(path):
    return json.loads(Path(path).read_text())


def verify_summary(path):
    old_arms = g51collect.ARMS
    old_format = g51collect.FORMAT
    g51collect.ARMS = ARMS
    g51collect.FORMAT = FORMAT
    try:
        summary, detail = g51collect.verify_summary(path)
    finally:
        g51collect.ARMS = old_arms
        g51collect.FORMAT = old_format

    errors = []
    cfg = summary.get("config", {})
    training = summary.get("training", {})
    calibration = load_json(path.parent / "calibration.json").get("selected", {})
    arm = ARMS.get(cfg.get("arm"), {})
    mode = arm.get("structured")

    if cfg.get("ambiguity_curriculum") != "family-balanced":
        errors.append("GDM60 curriculum must stay canonical family-balanced")
    if cfg.get("ambiguity_representation_mode") != mode or cfg.get("structured") != mode:
        errors.append("GDM60 representation mode mismatch")
    if cfg.get("ambiguity_requested_topk") != TOPK:
        errors.append("GDM60 must keep top-five candidate breadth fixed")
    if cfg.get("ambiguity_rank_preserving") is not True:
        errors.append("GDM60 must keep rank-preserving representation")
    if cfg.get("calibration_strategy") != "ambiguity-rescue" or cfg.get("status_arbitration") != "ambiguity-rescue":
        errors.append("GDM60 must keep fixed ambiguity-rescue arbitration")
    if cfg.get("nomatch_recall_budget_pp") != 5:
        errors.append("GDM60 must keep fixed 5pp NO_MATCH budget")

    if training.get("gdm60_family") != "confidence-scaled-ranked-request-ambiguity-evidence":
        errors.append("missing GDM60 family evidence")
    if training.get("gdm60_representation_mode") != mode:
        errors.append("training representation mode mismatch")
    if training.get("gdm60_requested_topk") != TOPK:
        errors.append("training top-k receipt mismatch")
    if training.get("gdm60_semantic_representation") != _representation_description(mode):
        errors.append("semantic representation receipt mismatch")
    if training.get("gdm60_rank_preserving") is not True:
        errors.append("rank-preserving receipt mismatch")
    if training.get("canonical_gdm59_source_commit") != CANONICAL_GDM59_SOURCE:
        errors.append("wrong canonical GDM59 source")
    if training.get("canonical_gdm59_audit_run") != CANONICAL_GDM59_AUDIT_RUN:
        errors.append("wrong canonical GDM59 audit receipt")
    if training.get("gdm60_change_scope") != CHANGE_SCOPE:
        errors.append("wrong GDM60 change scope")

    feature_dim = training.get("gdm60_ambiguity_feature_dim")
    if not isinstance(feature_dim, int) or feature_dim <= 11:
        errors.append("invalid GDM60 feature dimension")
    if cfg.get("ambiguity_feature_dim") != feature_dim:
        errors.append("config/training feature dimension mismatch")

    selected = path.parent / "selected.pt"
    try:
        checkpoint = torch.load(selected, map_location="cpu", weights_only=True)
        state = checkpoint["state"]
        weight = state["ambiguity.net.0.weight"]
        if int(weight.shape[1]) != int(feature_dim):
            errors.append("ambiguity checkpoint input dimension mismatch")
        expected_dim = ambiguity_feature_dim(128 if cfg.get("backbone") == "hash" else 768)
        if int(feature_dim) != int(expected_dim):
            errors.append("GDM60 matched-capacity feature dimension contract mismatch")
    except Exception as exc:
        errors.append("GDM60 checkpoint shape verification failed:" + type(exc).__name__)

    if training.get("gdm56_ambiguity_curriculum") != "family-balanced":
        errors.append("canonical GDM56 family-balanced curriculum not preserved")
    family_counts = training.get("gdm56_positive_family_counts")
    if not isinstance(family_counts, dict) or set(family_counts) != EXPECTED_FAMILIES:
        errors.append("missing fixed balanced curriculum family counts")
    else:
        values = [int(family_counts[k]) for k in sorted(EXPECTED_FAMILIES)]
        if min(values) <= 0 or max(values) - min(values) > 1:
            errors.append("fixed curriculum is not family balanced")
    if abs(float(training.get("gdm56_counterfactual_negative_fraction", -1))) > 1e-12:
        errors.append("counterfactual negatives leaked into fixed GDM60 curriculum")

    if calibration.get("threshold_selection") != "ambiguity-rescue":
        errors.append("selected threshold strategy mismatch")
    if calibration.get("status_arbitration") != "ambiguity-rescue":
        errors.append("selected status arbitration mismatch")
    if calibration.get("nomatch_recall_budget_pp") != 5:
        errors.append("selected NO_MATCH budget mismatch")
    if not isinstance(calibration.get("ambiguity_rescue_threshold"), (int, float)):
        errors.append("missing ambiguity rescue threshold")
    baseline = calibration.get("joint_baseline_validation_metrics")
    nomatch_floor = calibration.get("nomatch_recall_floor")
    accepted_floor = calibration.get("accepted_status_recall_floor")
    reported = training.get("selected_calibration_validation_metrics", {})
    if not isinstance(baseline, dict):
        errors.append("missing joint baseline validation metrics")
    elif isinstance(nomatch_floor, (int, float)) and isinstance(accepted_floor, (int, float)):
        expected_nomatch = max(0.0, float(baseline.get("nomatch_recall", -1)) - 0.05)
        expected_accepted = max(0.0, float(baseline.get("accepted_status_recall", -1)) - 0.05)
        if abs(float(nomatch_floor) - expected_nomatch) > 1e-12:
            errors.append("NO_MATCH recall floor mismatch")
        if abs(float(accepted_floor) - expected_accepted) > 1e-12:
            errors.append("accepted recall floor mismatch")
        if reported.get("nomatch_recall", -1) + 1e-12 < float(nomatch_floor):
            errors.append("NO_MATCH recall floor violated")
        if reported.get("accepted_status_recall", -1) + 1e-12 < float(accepted_floor):
            errors.append("accepted recall floor violated")
    else:
        errors.append("missing rescue recall floors")

    encoder = summary.get("encoder", {})
    head = encoder.get("head_training_reproducibility", {})
    if cfg.get("backbone") != "hash":
        if head.get("gdm60_execution_scope") != "capability_and_ambiguity_heads":
            errors.append("missing GDM60 execution scope")
        if head.get("canonical_gdm59_source_commit") != CANONICAL_GDM59_SOURCE:
            errors.append("wrong canonical GDM59 execution source")
        if head.get("canonical_gdm59_audit_run") != CANONICAL_GDM59_AUDIT_RUN:
            errors.append("wrong canonical GDM59 execution audit receipt")
        if head.get("gdm60_change_scope") != CHANGE_SCOPE:
            errors.append("wrong GDM60 execution change scope")

    if "Pass/Logbook" not in summary.get("calibration_scope", ""):
        errors.append("wrong calibration domain scope")
    if "Certificate/Manuscript" not in summary.get("secondary_holdout_scope", ""):
        errors.append("wrong holdout domain scope")
    if "GDM46-GDM59" not in summary.get("regression_scope", ""):
        errors.append("regression scope does not include canonical GDM59 holdout")

    reported_families = summary.get("secondary_holdout_ambiguity_family_metrics")
    recomputed_families = _ambiguity_family_metrics(path.parent / "predictions-holdout.jsonl")
    if reported_families != recomputed_families:
        errors.append("ambiguity-family metrics do not recount from predictions")
    if set(recomputed_families) != EXPECTED_FAMILIES:
        errors.append("fresh holdout does not contain exact four ambiguity families")
    if any(values.get("total", 0) <= 0 for values in recomputed_families.values()):
        errors.append("empty ambiguity-family holdout bucket")

    latency = summary.get("generation_latency", {})
    memory = summary.get("memory", {})
    if not isinstance(latency.get("p50_ms"), (int, float)) or not isinstance(latency.get("p95_ms"), (int, float)):
        errors.append("invalid generation timing")
    if not isinstance(memory.get("rss_after_evaluation_kib"), int) or memory.get("rss_after_evaluation_kib", 0) <= 0:
        errors.append("invalid memory evidence")

    if errors:
        raise AssertionError(path.as_posix() + ": " + "; ".join(sorted(set(errors))))
    return summary, detail


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("root")
    parser.add_argument("--seeds", default="6001,6002")
    parser.add_argument("--tasks", default="operation,schema")
    args, _ = parser.parse_known_args()

    old_arms = g52collect.ARMS
    old_format = g52collect.FORMAT
    old_verify = g52collect.verify_summary
    g52collect.ARMS = ARMS
    g52collect.FORMAT = FORMAT
    g52collect.verify_summary = verify_summary
    try:
        g52collect.main()
    finally:
        g52collect.ARMS = old_arms
        g52collect.FORMAT = old_format
        g52collect.verify_summary = old_verify

    root = Path(args.root)
    report_path = root / "BATCH_REPORT.json"
    report = load_json(report_path)
    report["format"] = "gdm60-batch-report-v1"
    report["hypotheses"] = [
        "Canonical GDM59 shows that rank-preserving request-relative top-five semantics improve answerable correctness and recall versus pooled evidence but retain a publication-safety tradeoff.",
        "All GDM60 arms use the same top-five candidate breadth, family-balanced curriculum, ambiguity-head dimension and trainable parameter count; only request-relative semantic factorization/confidence scaling differs.",
        "The ranked-request-control is the canonical GDM59 request-relative representation reproduced inside the fresh GDM60 holdout rather than an unlike cross-batch comparison.",
        "Probability-scaled arms test whether attenuating lower-confidence ranked candidates can improve risk/publication behavior without erasing answerable and multi-clause gains.",
    ]
    report["interpretation_limits"] = [
        "Frozen DistilBERT plus learned feature-space capability and ambiguity heads; no transformer fine-tuning.",
        "GDM60 tests simple per-rank confidence scaling/factorization, not a learned set transformer or new request-global status model.",
        "All arms keep the same 11+10*d ambiguity input dimension, parameter count and top-five candidate breadth.",
        "Schema generation remains catalog projection plus deterministic SDL/Federation realization, not unconstrained schema invention.",
        "Certificate/Manuscript is a synthetic secondary holdout, not enterprise OOD.",
    ]
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
