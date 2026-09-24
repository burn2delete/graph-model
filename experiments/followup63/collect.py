"""Independent operation-only evidence gate for GDM63 learned candidate interaction."""
from __future__ import annotations

import argparse
import copy
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
    CANONICAL_GDM62_AUDIT_RUN,
    CANONICAL_GDM62_PROMOTION,
    CANONICAL_GDM62_SOURCE,
    FORMAT,
    INTERACTION_MODES,
    LATENT_WIDTH,
    TOPK,
    ambiguity_feature_dim,
    interaction_parameter_count,
    _interaction_description,
    _interaction_state_hash,
)

EXPECTED_FAMILIES = {"role", "lifecycle-time", "representation", "object-vs-supplier"}
CHANGE_SCOPE = (
    "operation-only small learned rank-preserving cross-candidate ambiguity interaction over fixed canonical q*ci+abs(q-ci) top5 slots; matched architecture/parameter count across all five arms; fixed capability training, GDM56 family-balanced curriculum, canonical numerical path and GDM54 5pp rescue arbitration"
)


def load_json(path):
    return json.loads(Path(path).read_text())


def _base_verify_with_family_compat(path):
    """Reuse hardened GDM51 verification while accepting GDM63's more specific family label."""
    old_arms = g51collect.ARMS
    old_format = g51collect.FORMAT
    old_load = g51collect.load_json
    compat_arms = {
        name: {**spec, "family": "explicit-none-plus-ambiguity"}
        for name, spec in ARMS.items()
    }

    def compat_load(p):
        data = old_load(p)
        if Path(p).name == "summary.json" and data.get("format") == FORMAT:
            data = copy.deepcopy(data)
            data.setdefault("config", {})["family"] = "explicit-none-plus-ambiguity"
            data.setdefault("training", {})["gdm51_family"] = "explicit-none-plus-ambiguity"
        return data

    g51collect.ARMS = compat_arms
    g51collect.FORMAT = FORMAT
    g51collect.load_json = compat_load
    try:
        _summary, detail = g51collect.verify_summary(path)
    finally:
        g51collect.ARMS = old_arms
        g51collect.FORMAT = old_format
        g51collect.load_json = old_load
    return detail


def verify_summary(path):
    detail = _base_verify_with_family_compat(path)
    summary = load_json(path)
    errors = []
    cfg = summary.get("config", {})
    training = summary.get("training", {})
    calibration = load_json(path.parent / "calibration.json").get("selected", {})
    arm_name = cfg.get("arm")
    arm = ARMS.get(arm_name)
    if arm is None:
        errors.append("unexpected GDM63 arm")
        arm = {}

    if cfg.get("task") != "operation":
        errors.append("GDM63 is operation-only")
    if cfg.get("forward_scope") != "operation-generation-only":
        errors.append("missing operation-only scope receipt")
    if cfg.get("schema_generation") != "suspended-frozen":
        errors.append("schema suspension receipt missing")
    if cfg.get("ambiguity_curriculum") != "family-balanced":
        errors.append("GDM63 curriculum must stay canonical family-balanced")
    if cfg.get("ambiguity_representation_mode") != "ranked-request-control":
        errors.append("raw representation must remain canonical request-relative")
    if cfg.get("ambiguity_interaction_mode") != arm_name:
        errors.append("interaction mode/config mismatch")
    if cfg.get("structured") != "ranked-request-control":
        errors.append("all GDM63 arms must receive identical canonical raw slots")
    if cfg.get("ambiguity_requested_topk") != TOPK:
        errors.append("GDM63 must keep top-five breadth fixed")
    if cfg.get("ambiguity_rank_preserving") is not True:
        errors.append("GDM63 must preserve candidate rank")
    if cfg.get("calibration_strategy") != "ambiguity-rescue" or cfg.get("status_arbitration") != "ambiguity-rescue":
        errors.append("GDM63 must keep ambiguity-rescue arbitration")
    if cfg.get("nomatch_recall_budget_pp") != 5:
        errors.append("GDM63 must keep fixed 5pp NO_MATCH budget")

    if training.get("gdm63_family") != "learned-rank-preserving-cross-candidate-interaction":
        errors.append("missing GDM63 learned interaction family receipt")
    if training.get("gdm63_interaction_mode") != arm_name:
        errors.append("training interaction mode mismatch")
    if training.get("gdm63_requested_topk") != TOPK:
        errors.append("training top-k receipt mismatch")
    if training.get("gdm63_raw_representation") != "canonical ranked request-product q*ci plus request-delta abs(q-ci)":
        errors.append("raw representation receipt mismatch")
    if arm_name in INTERACTION_MODES and training.get("gdm63_interaction_description") != _interaction_description(arm_name):
        errors.append("interaction description mismatch")
    if training.get("gdm63_latent_width") != LATENT_WIDTH:
        errors.append("latent-width receipt mismatch")
    if training.get("gdm63_rank_preserving") is not True:
        errors.append("rank-preserving receipt mismatch")
    if training.get("gdm63_change_scope") != CHANGE_SCOPE:
        errors.append("wrong GDM63 change scope")
    if training.get("canonical_gdm62_source_commit") != CANONICAL_GDM62_SOURCE:
        errors.append("wrong canonical GDM62 source")
    if training.get("canonical_gdm62_audit_run") != CANONICAL_GDM62_AUDIT_RUN:
        errors.append("wrong canonical GDM62 audit")
    if training.get("canonical_gdm62_promotion_commit") != CANONICAL_GDM62_PROMOTION:
        errors.append("wrong canonical GDM62 promotion receipt")

    feature_dim = training.get("gdm63_ambiguity_feature_dim")
    if not isinstance(feature_dim, int) or feature_dim <= BASE_FEATURES:
        errors.append("invalid GDM63 ambiguity feature dimension")
    if cfg.get("ambiguity_feature_dim") != feature_dim:
        errors.append("config/training feature dimension mismatch")
    encoder_dim = 128 if cfg.get("backbone") == "hash" else 768
    if feature_dim != ambiguity_feature_dim(encoder_dim):
        errors.append("GDM63 fixed raw feature dimension contract mismatch")
    expected_interaction_params = interaction_parameter_count(encoder_dim)
    if training.get("gdm63_interaction_parameter_count") != expected_interaction_params:
        errors.append("interaction parameter count mismatch")
    if not isinstance(training.get("gdm63_ambiguity_parameter_count"), int):
        errors.append("missing total ambiguity parameter count")
    changed = training.get("gdm63_changed_interaction_parameter_tensors")
    if not isinstance(changed, list) or "ambiguity.proj.weight" not in changed:
        errors.append("learned interaction projection did not demonstrably change")
    if training.get("gdm63_initial_interaction_hash") == training.get("gdm63_selected_interaction_hash"):
        errors.append("learned interaction state unchanged")

    try:
        initial = torch.load(path.parent / "initial.pt", map_location="cpu", weights_only=True)["state"]
        selected = torch.load(path.parent / "selected.pt", map_location="cpu", weights_only=True)["state"]
        if _interaction_state_hash(initial) != training.get("gdm63_initial_interaction_hash"):
            errors.append("initial interaction hash mismatch")
        if _interaction_state_hash(selected) != training.get("gdm63_selected_interaction_hash"):
            errors.append("selected interaction hash mismatch")
        weight = selected["ambiguity.net.0.weight"]
        if int(weight.shape[1]) != int(feature_dim) or int(weight.shape[0]) != 32:
            errors.append("downstream ambiguity MLP shape mismatch")
        proj = selected["ambiguity.proj.weight"]
        if tuple(proj.shape) != (LATENT_WIDTH, 2 * encoder_dim):
            errors.append("learned projection shape mismatch")
    except Exception as exc:
        errors.append("GDM63 checkpoint verification failed:" + type(exc).__name__)

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
        errors.append("counterfactual negatives leaked into fixed GDM63 curriculum")

    if calibration.get("threshold_selection") != "ambiguity-rescue":
        errors.append("selected threshold strategy mismatch")
    if calibration.get("status_arbitration") != "ambiguity-rescue":
        errors.append("selected status arbitration mismatch")
    if calibration.get("nomatch_recall_budget_pp") != 5:
        errors.append("selected NO_MATCH budget mismatch")
    if not isinstance(calibration.get("ambiguity_rescue_threshold"), (int, float)):
        errors.append("missing ambiguity rescue threshold")

    encoder = summary.get("encoder", {})
    head = encoder.get("head_training_reproducibility", {})
    if cfg.get("backbone") != "hash":
        if head.get("gdm63_execution_scope") != "operation_capability_and_ambiguity_heads_only":
            errors.append("missing GDM63 operation-only execution scope")
        if head.get("schema_generation_scope") != "suspended-frozen":
            errors.append("execution metadata does not freeze schema generation")
        if head.get("canonical_gdm62_source_commit") != CANONICAL_GDM62_SOURCE:
            errors.append("wrong canonical GDM62 execution source")
        if head.get("canonical_gdm62_audit_run") != CANONICAL_GDM62_AUDIT_RUN:
            errors.append("wrong canonical GDM62 execution audit")
        if head.get("gdm63_change_scope") != CHANGE_SCOPE:
            errors.append("wrong GDM63 execution change scope")

    if "Ledger/Index" not in summary.get("calibration_scope", ""):
        errors.append("wrong calibration domain scope")
    if "Docket/Portfolio" not in summary.get("secondary_holdout_scope", ""):
        errors.append("wrong holdout domain scope")
    if "GDM46-GDM62" not in summary.get("regression_scope", ""):
        errors.append("regression scope does not include canonical GDM62 operation holdout")

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
    parser.add_argument("--seeds", default="6301,6302")
    parser.add_argument("--tasks", default="operation")
    args, _ = parser.parse_known_args()
    if args.tasks != "operation":
        raise SystemExit("GDM63 collector accepts operation evidence only")

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
    report["format"] = "gdm63-operation-batch-report-v1"
    report["forward_scope"] = "operation-generation-only"
    report["schema_configs_compared"] = 0
    report["hypotheses"] = [
        "Canonical GDM62 found fixed rank-1 candidate relations negative/inconclusive and retained canonical q*ci+abs(q-ci) ranked request-relative slots.",
        "Every GDM63 arm receives exactly the same canonical raw top-five request-relative slots and the same small learned interaction adapter/downstream MLP parameter count.",
        "The matched-capacity local-self arm controls for merely adding learned capacity; cross-candidate arms differ only in the fixed relation used to aggregate projected rank slots into learned gates.",
        "A useful learned cross-candidate interaction should improve role/representation ambiguity or multi-clause exactness without buying publication safety through answerable/recall collapse.",
    ]
    report["interpretation_limits"] = [
        "Operation generation only; schema generation is suspended/frozen and contributes zero forward configs.",
        "Frozen DistilBERT plus learned feature-space capability/ambiguity modules; no transformer fine-tuning.",
        "GDM63 tests a small rank-preserving learned gate over canonical request-relative slots, not token-level generation, full self-attention, or a request-global trained status model.",
        "Docket/Portfolio is a synthetic secondary holdout, not enterprise OOD; after inspection it becomes regression-only.",
    ]
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
