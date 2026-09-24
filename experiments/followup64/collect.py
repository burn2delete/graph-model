"""Independent operation-only evidence gate for GDM64 schema-coordinate factorization."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from experiments.followup51 import collect as g51collect
from experiments.followup52 import collect as g52collect
from experiments.followup55.run import _ambiguity_family_metrics
from .run import (
    ARMS,
    BASE_FEATURES,
    BLOCK_KINDS,
    CANONICAL_GDM63_AUDIT_RUN,
    CANONICAL_GDM63_PROMOTION,
    CANONICAL_GDM63_SOURCE,
    FORMAT,
    TOPK,
    ambiguity_feature_dim,
    _representation_description,
)

EXPECTED_FAMILIES = {"role", "lifecycle-time", "representation", "object-vs-supplier"}
CHANGE_SCOPE = (
    "operation-only schema-coordinate semantic factorization inside fixed top5 ranked ambiguity slots; deterministic parent/role and leaf/representation views derived only from provided GraphQL paths; fixed dimensions, ambiguity-head capacity, capability training, GDM56 family-balanced curriculum, canonical numerical path and GDM54 5pp rescue arbitration"
)
PARENT_TEXT_CONTRACT = "GraphQL parent path: + path[1:-1] joined by dot; <root> if empty"
LEAF_TEXT_CONTRACT = "GraphQL leaf field: + path[-1]"


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
    arm_name = cfg.get("arm")
    arm = ARMS.get(arm_name, {})
    mode = arm.get("structured")

    if cfg.get("task") != "operation":
        errors.append("GDM64 is operation-only")
    if cfg.get("forward_scope") != "operation-generation-only":
        errors.append("missing operation-only scope receipt")
    if cfg.get("schema_generation") != "suspended-frozen":
        errors.append("schema suspension receipt missing")
    if cfg.get("ambiguity_curriculum") != "family-balanced":
        errors.append("GDM64 curriculum must stay canonical family-balanced")
    if cfg.get("ambiguity_representation_mode") != mode or cfg.get("structured") != mode:
        errors.append("GDM64 representation mode mismatch")
    if cfg.get("ambiguity_requested_topk") != TOPK:
        errors.append("GDM64 must keep top-five breadth fixed")
    if cfg.get("ambiguity_rank_preserving") is not True:
        errors.append("GDM64 must preserve candidate rank")
    if cfg.get("ambiguity_catalog_semantic_views_cached") is not True:
        errors.append("catalog semantic views must remain cached")
    if cfg.get("calibration_strategy") != "ambiguity-rescue" or cfg.get("status_arbitration") != "ambiguity-rescue":
        errors.append("GDM64 must keep ambiguity-rescue arbitration")
    if cfg.get("nomatch_recall_budget_pp") != 5:
        errors.append("GDM64 must keep fixed 5pp NO_MATCH budget")

    if training.get("gdm64_family") != "schema-coordinate-semantic-factorization":
        errors.append("missing GDM64 family receipt")
    if training.get("gdm64_representation_mode") != mode:
        errors.append("training representation mode mismatch")
    if training.get("gdm64_requested_topk") != TOPK:
        errors.append("training top-k receipt mismatch")
    if training.get("gdm64_semantic_representation") != _representation_description(mode):
        errors.append("semantic representation receipt mismatch")
    if training.get("gdm64_rank_preserving") is not True:
        errors.append("rank-preserving receipt mismatch")
    if training.get("gdm64_catalog_semantic_views_cached") is not True:
        errors.append("cached catalog semantic-view receipt missing")
    if training.get("gdm64_parent_text_contract") != PARENT_TEXT_CONTRACT:
        errors.append("parent semantic-view contract mismatch")
    if training.get("gdm64_leaf_text_contract") != LEAF_TEXT_CONTRACT:
        errors.append("leaf semantic-view contract mismatch")
    if training.get("gdm64_change_scope") != CHANGE_SCOPE:
        errors.append("wrong GDM64 change scope")
    if training.get("canonical_gdm63_source_commit") != CANONICAL_GDM63_SOURCE:
        errors.append("wrong canonical GDM63 source")
    if training.get("canonical_gdm63_audit_run") != CANONICAL_GDM63_AUDIT_RUN:
        errors.append("wrong canonical GDM63 audit")
    if training.get("canonical_gdm63_promotion_commit") != CANONICAL_GDM63_PROMOTION:
        errors.append("wrong canonical GDM63 promotion")

    expected_blocks = BLOCK_KINDS.get(mode)
    if expected_blocks is None:
        errors.append("unknown GDM64 representation mode")
    else:
        if training.get("gdm64_first_block") != expected_blocks[0]:
            errors.append("first semantic block receipt mismatch")
        if training.get("gdm64_second_block") != expected_blocks[1]:
            errors.append("second semantic block receipt mismatch")

    feature_dim = training.get("gdm64_ambiguity_feature_dim")
    encoder_dim = 128 if cfg.get("backbone") == "hash" else 768
    if not isinstance(feature_dim, int) or feature_dim <= BASE_FEATURES:
        errors.append("invalid GDM64 ambiguity feature dimension")
    elif feature_dim != ambiguity_feature_dim(encoder_dim):
        errors.append("GDM64 fixed feature dimension contract mismatch")
    if cfg.get("ambiguity_feature_dim") != feature_dim:
        errors.append("config/training feature dimension mismatch")

    try:
        initial = torch.load(path.parent / "initial.pt", map_location="cpu", weights_only=True)["state"]
        selected = torch.load(path.parent / "selected.pt", map_location="cpu", weights_only=True)["state"]
        weight = selected["ambiguity.net.0.weight"]
        if tuple(weight.shape) != (32, int(feature_dim)):
            errors.append("ambiguity MLP shape mismatch")
        if torch.equal(initial["ambiguity.net.0.weight"], selected["ambiguity.net.0.weight"]):
            errors.append("ambiguity MLP did not demonstrably train")
        count = sum(v.numel() for k, v in selected.items() if k.startswith("ambiguity."))
        if training.get("gdm64_ambiguity_parameter_count") != count:
            errors.append("ambiguity parameter-count receipt mismatch")
    except Exception as exc:
        errors.append("GDM64 checkpoint verification failed:" + type(exc).__name__)

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
        errors.append("counterfactual negatives leaked into fixed GDM64 curriculum")

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
        if head.get("gdm64_execution_scope") != "operation_capability_and_ambiguity_heads_only":
            errors.append("missing GDM64 operation-only execution scope")
        if head.get("schema_generation_scope") != "suspended-frozen":
            errors.append("execution metadata does not freeze schema generation")
        if head.get("canonical_gdm63_source_commit") != CANONICAL_GDM63_SOURCE:
            errors.append("wrong canonical GDM63 execution source")
        if head.get("canonical_gdm63_audit_run") != CANONICAL_GDM63_AUDIT_RUN:
            errors.append("wrong canonical GDM63 execution audit")
        if head.get("canonical_gdm63_promotion_commit") != CANONICAL_GDM63_PROMOTION:
            errors.append("wrong canonical GDM63 execution promotion")
        if head.get("gdm64_change_scope") != CHANGE_SCOPE:
            errors.append("wrong GDM64 execution change scope")

    if "Registry/Logbook" not in summary.get("calibration_scope", ""):
        errors.append("wrong calibration domain scope")
    if "Folio/Casebook" not in summary.get("secondary_holdout_scope", ""):
        errors.append("wrong holdout domain scope")
    if "GDM46-GDM63" not in summary.get("regression_scope", ""):
        errors.append("regression scope does not include canonical GDM63 holdout")

    reported_families = summary.get("secondary_holdout_ambiguity_family_metrics")
    recomputed_families = _ambiguity_family_metrics(path.parent / "predictions-holdout.jsonl")
    if reported_families != recomputed_families:
        errors.append("ambiguity-family metrics do not recount from predictions")
    if set(recomputed_families) != EXPECTED_FAMILIES:
        errors.append("fresh operation holdout does not contain exact four ambiguity families")
    if any(values.get("total", 0) <= 0 for values in recomputed_families.values()):
        errors.append("empty ambiguity-family holdout bucket")

    latency = summary.get("generation_latency", {})
    memory = summary.get("memory", {})
    if not isinstance(latency.get("p50_ms"), (int, float)) or not isinstance(latency.get("p95_ms"), (int, float)):
        errors.append("invalid generation timing")
    if not isinstance(memory.get("rss_after_evaluation_kib"), int) or memory.get("rss_after_evaluation_kib", 0) <= 0:
        errors.append("invalid process memory evidence")

    if errors:
        raise AssertionError(path.as_posix() + ": " + "; ".join(sorted(set(errors))))
    return summary, detail


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("root")
    parser.add_argument("--seeds", default="6401,6402")
    parser.add_argument("--tasks", default="operation")
    args, _ = parser.parse_known_args()
    if args.tasks != "operation":
        raise SystemExit("GDM64 collector accepts operation evidence only")

    old_arms = g52collect.ARMS
    old_format = g52collect.FORMAT
    old_verify = g52collect.verify_summary
    old_argv = list(sys.argv)
    g52collect.ARMS = ARMS
    g52collect.FORMAT = FORMAT
    g52collect.verify_summary = verify_summary
    sys.argv = [sys.argv[0], args.root, "--seeds", args.seeds, "--tasks", "operation"]
    try:
        g52collect.main()
    finally:
        g52collect.ARMS = old_arms
        g52collect.FORMAT = old_format
        g52collect.verify_summary = old_verify
        sys.argv = old_argv

    root = Path(args.root)
    report_path = root / "BATCH_REPORT.json"
    report = load_json(report_path)
    report["format"] = "gdm64-operation-batch-report-v1"
    report["forward_scope"] = "operation-generation-only"
    report["schema_configs_compared"] = 0
    report["hypotheses"] = [
        "Canonical GDM63 found matched-capacity learned candidate-set scalar gates negative/inconclusive: learned cross-candidate modes did not improve answerable, target, clause, family, or semantic-confusion outcomes over a local control.",
        "GDM64 keeps TOP5 rank, dimensions, MLP capacity, capability training, curriculum, numerical path and arbitration fixed while changing only which frozen schema-grounded semantic view populates two d-dimensional blocks per rank.",
        "Parent/role path views may expose author-versus-moderator semantics that whole-candidate embeddings and candidate-set gates failed to separate.",
        "Leaf/representation views may expose name-versus-ID and lifecycle field distinctions without adding parameters or holdout-derived labels.",
    ]
    report["interpretation_limits"] = [
        "Operation generation only; schema generation is suspended/frozen and contributes zero forward configs.",
        "Frozen DistilBERT plus learned feature-space capability/ambiguity heads; no transformer fine-tuning.",
        "GDM64 path-factor features are deterministic text derived only from the provided GraphQL coordinate; this is not token-level generation or a learned schema encoder.",
        "All arms keep the same 11+10*d ambiguity input dimension, top-five breadth and ambiguity-head parameter count.",
        "Folio/Casebook is a synthetic secondary holdout, not enterprise OOD; after inspection it becomes regression-only.",
    ]
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
