"""Independent evidence gate for GDM62 candidate-relational ambiguity evidence."""
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
    BLOCK_KINDS,
    CANONICAL_GDM61_AUDIT_RUN,
    CANONICAL_GDM61_SOURCE,
    FORMAT,
    TOPK,
    ambiguity_feature_dim,
    _representation_description,
)

EXPECTED_FAMILIES = {"role", "lifecycle-time", "representation", "object-vs-supplier"}
CHANGE_SCOPE = (
    "ambiguity-head candidate-relational semantic evidence only; fixed top5 breadth, rank preservation, capacity, canonical GDM56 family-balanced curriculum, canonical numerical path and GDM54 5pp rescue arbitration"
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
        errors.append("GDM62 curriculum must stay canonical family-balanced")
    if cfg.get("ambiguity_representation_mode") != mode or cfg.get("structured") != mode:
        errors.append("GDM62 representation mode mismatch")
    if cfg.get("ambiguity_requested_topk") != TOPK:
        errors.append("GDM62 must keep top-five candidate breadth fixed")
    if cfg.get("ambiguity_rank_preserving") is not True:
        errors.append("GDM62 must keep rank-preserving representation")
    if cfg.get("calibration_strategy") != "ambiguity-rescue" or cfg.get("status_arbitration") != "ambiguity-rescue":
        errors.append("GDM62 must keep fixed ambiguity-rescue arbitration")
    if cfg.get("nomatch_recall_budget_pp") != 5:
        errors.append("GDM62 must keep fixed 5pp NO_MATCH budget")

    if training.get("gdm62_family") != "ranked-candidate-relational-ambiguity-evidence":
        errors.append("missing GDM62 family evidence")
    if training.get("gdm62_representation_mode") != mode:
        errors.append("training representation mode mismatch")
    if training.get("gdm62_requested_topk") != TOPK:
        errors.append("training top-k receipt mismatch")
    if training.get("gdm62_semantic_representation") != _representation_description(mode):
        errors.append("semantic representation receipt mismatch")
    if training.get("gdm62_rank_preserving") is not True:
        errors.append("rank-preserving receipt mismatch")
    if training.get("gdm62_candidate_anchor") != "rank1-real-candidate":
        errors.append("candidate anchor receipt mismatch")
    if training.get("canonical_gdm61_source_commit") != CANONICAL_GDM61_SOURCE:
        errors.append("wrong canonical GDM61 source")
    if training.get("canonical_gdm61_audit_run") != CANONICAL_GDM61_AUDIT_RUN:
        errors.append("wrong canonical GDM61 audit receipt")
    if training.get("gdm62_change_scope") != CHANGE_SCOPE:
        errors.append("wrong GDM62 change scope")
    expected_blocks = BLOCK_KINDS.get(mode)
    if expected_blocks is None:
        errors.append("unknown GDM62 block mode")
    else:
        if training.get("gdm62_first_block") != expected_blocks[0]:
            errors.append("first semantic block receipt mismatch")
        if training.get("gdm62_second_block") != expected_blocks[1]:
            errors.append("second semantic block receipt mismatch")

    feature_dim = training.get("gdm62_ambiguity_feature_dim")
    if not isinstance(feature_dim, int) or feature_dim <= 11:
        errors.append("invalid GDM62 feature dimension")
    if cfg.get("ambiguity_feature_dim") != feature_dim:
        errors.append("config/training feature dimension mismatch")

    try:
        checkpoint = torch.load(path.parent / "selected.pt", map_location="cpu", weights_only=True)
        state = checkpoint["state"]
        weight = state["ambiguity.net.0.weight"]
        if int(weight.shape[1]) != int(feature_dim):
            errors.append("ambiguity checkpoint input dimension mismatch")
        expected_dim = ambiguity_feature_dim(128 if cfg.get("backbone") == "hash" else 768)
        if int(feature_dim) != int(expected_dim):
            errors.append("GDM62 matched-capacity feature dimension contract mismatch")
    except Exception as exc:
        errors.append("GDM62 checkpoint shape verification failed:" + type(exc).__name__)

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
        errors.append("counterfactual negatives leaked into fixed GDM62 curriculum")

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
        if head.get("gdm62_execution_scope") != "capability_and_ambiguity_heads":
            errors.append("missing GDM62 execution scope")
        if head.get("canonical_gdm61_source_commit") != CANONICAL_GDM61_SOURCE:
            errors.append("wrong canonical GDM61 execution source")
        if head.get("canonical_gdm61_audit_run") != CANONICAL_GDM61_AUDIT_RUN:
            errors.append("wrong canonical GDM61 execution audit receipt")
        if head.get("gdm62_change_scope") != CHANGE_SCOPE:
            errors.append("wrong GDM62 execution change scope")

    if "Medallion/Gazette" not in summary.get("calibration_scope", ""):
        errors.append("wrong calibration domain scope")
    if "Ticket/Compendium" not in summary.get("secondary_holdout_scope", ""):
        errors.append("wrong holdout domain scope")
    if "GDM46-GDM61" not in summary.get("regression_scope", ""):
        errors.append("regression scope does not include canonical GDM61 holdout")

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
    parser.add_argument("--seeds", default="6201,6202")
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
    report["format"] = "gdm62-batch-report-v1"
    report["hypotheses"] = [
        "Canonical GDM61 found no reliable answerable/safety frontier improvement from fixed scaling of existing request-relative product/delta blocks.",
        "All GDM62 arms keep top-five breadth, rank identity, ambiguity-head dimension/parameter count, curriculum, numerical path and arbitration fixed; only the two 5*d semantic relation blocks differ.",
        "Candidate-delta abs(c1-ci) and candidate-product c1*ci expose nonlinear relations among plausible candidates that are not simply another scalar rescaling of request-relative blocks.",
        "Matched request-product/request-delta pairings test whether candidate-relative evidence improves ambiguity recovery without sacrificing answerable capability or publication safety.",
    ]
    report["interpretation_limits"] = [
        "Frozen DistilBERT plus learned feature-space capability and ambiguity heads; no transformer fine-tuning.",
        "GDM62 tests rank-1-anchored candidate relations, not learned candidate-set attention or a request-global status model.",
        "All arms keep the same 11+10*d ambiguity input dimension, parameter count and top-five candidate breadth.",
        "Schema generation remains catalog projection plus deterministic SDL/Federation realization, not unconstrained schema invention.",
        "Ticket/Compendium is a synthetic secondary holdout, not enterprise OOD.",
    ]
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
