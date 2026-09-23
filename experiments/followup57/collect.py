"""Independent evidence gate for GDM57 ambiguity-representation experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.followup51 import collect as g51collect
from experiments.followup52 import collect as g52collect
from experiments.followup55.run import _ambiguity_family_metrics
from .run import ARMS, FORMAT, ambiguity_feature_dim

EXPECTED_FAMILIES = {"role", "lifecycle-time", "representation", "object-vs-supplier"}
CHANGE_SCOPE = (
    "ambiguity-head top-two semantic input representation only; fixed canonical GDM56 family-balanced curriculum, numerical path and GDM54 5pp rescue arbitration"
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
        errors.append("GDM57 curriculum must stay canonical family-balanced")
    if cfg.get("ambiguity_representation_mode") != mode:
        errors.append("config representation mode mismatch")
    if cfg.get("structured") != mode:
        errors.append("structured representation mode mismatch")
    if cfg.get("calibration_strategy") != "ambiguity-rescue" or cfg.get("status_arbitration") != "ambiguity-rescue":
        errors.append("GDM57 must keep fixed ambiguity-rescue arbitration")
    if cfg.get("nomatch_recall_budget_pp") != 5:
        errors.append("GDM57 must keep fixed 5pp NO_MATCH budget")

    if training.get("gdm57_family") != "top-two-semantic-ambiguity-representation":
        errors.append("missing GDM57 family evidence")
    if training.get("gdm57_representation_mode") != mode:
        errors.append("training representation mode mismatch")
    if training.get("canonical_gdm56_source_commit") != "168317f6495b05e04597ae084b75be6890fe0621":
        errors.append("wrong canonical GDM56 source")
    if training.get("gdm57_change_scope") != CHANGE_SCOPE:
        errors.append("wrong GDM57 change scope")

    encoder = summary.get("encoder", {})
    encoder_dim = int(encoder.get("hidden_size") or encoder.get("dimension") or 0)
    # Current encoder metadata does not promise an explicit dimension field. Infer it
    # from the selected ambiguity checkpoint and require the recorded dimension to
    # satisfy the fixed 11 + 10*d contract for known backbones.
    feature_dim = training.get("gdm57_ambiguity_feature_dim")
    if not isinstance(feature_dim, int) or feature_dim <= 11:
        errors.append("invalid GDM57 feature dimension")
    if cfg.get("ambiguity_feature_dim") != feature_dim:
        errors.append("config/training feature dimension mismatch")

    expected_blocks = {
        "scalar-control": (False, False, False),
        "candidate-directed": (True, False, False),
        "request-conditioned": (False, True, False),
        "candidate-request": (True, True, False),
        "symmetric-request": (True, True, True),
    }.get(mode)
    actual_blocks = (
        training.get("gdm57_candidate_semantic_block"),
        training.get("gdm57_request_semantic_block"),
        training.get("gdm57_order_invariant_candidate_block"),
    )
    if expected_blocks is None or actual_blocks != expected_blocks:
        errors.append("semantic block receipt mismatch")

    selected = path.parent / "selected.pt"
    try:
        checkpoint = torch.load(selected, map_location="cpu", weights_only=True)
        state = checkpoint["state"]
        weight = state["ambiguity.net.0.weight"]
        if int(weight.shape[1]) != int(feature_dim):
            errors.append("ambiguity checkpoint input dimension mismatch")
        # DistilBERT is 768d and the hash smoke is 128d.
        expected_dim = ambiguity_feature_dim(128 if cfg.get("backbone") == "hash" else 768)
        if int(feature_dim) != int(expected_dim):
            errors.append("GDM57 matched-capacity feature dimension contract mismatch")
    except Exception as exc:
        errors.append("GDM57 checkpoint shape verification failed:" + type(exc).__name__)

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
        errors.append("counterfactual negatives leaked into fixed GDM57 curriculum")

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

    head = encoder.get("head_training_reproducibility", {})
    if cfg.get("backbone") != "hash":
        if head.get("gdm57_execution_scope") != "capability_and_ambiguity_heads":
            errors.append("missing GDM57 execution scope")
        if head.get("canonical_gdm56_source_commit") != "168317f6495b05e04597ae084b75be6890fe0621":
            errors.append("wrong canonical GDM56 execution source")
        if head.get("gdm57_change_scope") != CHANGE_SCOPE:
            errors.append("wrong GDM57 execution change scope")

    if "Credential/Snapshot" not in summary.get("calibration_scope", ""):
        errors.append("wrong calibration domain scope")
    if "Reservation/Artifact" not in summary.get("secondary_holdout_scope", ""):
        errors.append("wrong holdout domain scope")
    if "GDM46-GDM56" not in summary.get("regression_scope", ""):
        errors.append("regression scope does not include canonical GDM56 holdout")

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
    parser.add_argument("--seeds", default="5701,5702")
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
    report["format"] = "gdm57-batch-report-v1"
    report["hypotheses"] = [
        "Canonical GDM56 shows curriculum balancing can stabilize role/lifecycle ambiguity for operations but representation and object-vs-supplier ambiguity remain near zero.",
        "All GDM57 arms use the same family-balanced ambiguity curriculum and the same ambiguity-head parameter count; only semantic input blocks differ.",
        "Candidate-directed features test whether top-two candidate semantic identity is missing from the canonical eleven-feature representation.",
        "Request-conditioned features test whether frozen request/candidate interactions add information beyond scalar ranking evidence.",
        "Combined and order-invariant candidate/request representations test whether richer semantics can improve the unresolved ambiguity families without changing capability scoring or arbitration.",
    ]
    report["interpretation_limits"] = [
        "Frozen DistilBERT plus learned feature-space capability and ambiguity heads; no transformer fine-tuning.",
        "GDM57 remains top-two: it does not test broader top-k candidate evidence if the relevant competing capability is absent from the top two.",
        "All semantic feature arms increase nonzero input information but keep ambiguity-head input dimensionality and trainable parameter count matched through zero padding.",
        "Schema generation remains catalog projection plus deterministic SDL/Federation realization, not unconstrained schema invention.",
        "Reservation/Artifact is a synthetic secondary holdout, not enterprise OOD.",
    ]
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
