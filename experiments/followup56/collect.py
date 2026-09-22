"""Independent evidence gate for GDM56 relation-family balance experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.followup51 import collect as g51collect
from experiments.followup52 import collect as g52collect
from experiments.followup55.run import _ambiguity_family_metrics
from .run import ARMS, FORMAT

EXPECTED_FAMILIES = {"role", "lifecycle-time", "representation", "object-vs-supplier"}
BALANCED_CURRICULA = {
    "family-balanced",
    "family-balanced-counterfactual",
    "family-balanced-domain",
    "family-balanced-counterfactual-domain",
}


def load_json(path):
    return json.loads(Path(path).read_text())


def verify_summary(path):
    # Canonical GDM51 verification checks changed learned checkpoints, optimizer
    # evidence, real GraphQL execution, Rover composition, prediction recounts and
    # the cache-corrected timing boundary. GDM56 adds curriculum/family contracts.
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
    curriculum = arm.get("ambiguity_curriculum")

    if cfg.get("ambiguity_curriculum") != curriculum:
        errors.append("config ambiguity curriculum mismatch")
    if cfg.get("calibration_strategy") != "ambiguity-rescue":
        errors.append("GDM56 must keep fixed ambiguity-rescue calibration")
    if cfg.get("status_arbitration") != "ambiguity-rescue":
        errors.append("GDM56 must keep fixed ambiguity-rescue arbitration")
    if cfg.get("nomatch_recall_budget_pp") != 5:
        errors.append("GDM56 must keep fixed 5pp NO_MATCH budget")

    if training.get("gdm56_family") != "relation-family-balance-stability":
        errors.append("missing GDM56 family evidence")
    if training.get("gdm56_ambiguity_curriculum") != curriculum:
        errors.append("training ambiguity curriculum mismatch")
    if training.get("canonical_gdm55_source_commit") != "db13313e7fd5021922768add76aaac617ec21208":
        errors.append("wrong canonical GDM55 source")
    if training.get("gdm56_change_scope") != (
        "ambiguity-head relation-family balance/factorization only; fixed canonical GDM55 architecture, numerical path and GDM54 5pp rescue arbitration"
    ):
        errors.append("wrong GDM56 change scope")

    positive = training.get("gdm56_curriculum_positive_examples")
    negative = training.get("gdm56_curriculum_negative_examples")
    total = training.get("gdm56_curriculum_examples")
    if not all(isinstance(value, int) and value > 0 for value in (positive, negative, total)):
        errors.append("invalid GDM56 curriculum counts")
    elif positive + negative != total:
        errors.append("GDM56 positive/negative curriculum counts do not sum")
    if total != training.get("ambiguity_train_examples"):
        errors.append("curriculum example count does not match optimizer curriculum")

    sources = training.get("gdm56_curriculum_sources")
    if not isinstance(sources, dict) or not sources:
        errors.append("missing GDM56 curriculum source counts")
    family_counts = training.get("gdm56_positive_family_counts")
    if not isinstance(family_counts, dict) or set(family_counts) != EXPECTED_FAMILIES:
        errors.append("missing exact GDM56 positive relation families")
    elif curriculum in BALANCED_CURRICULA:
        values = [int(family_counts[family]) for family in sorted(EXPECTED_FAMILIES)]
        if min(values) <= 0 or max(values) - min(values) > 1:
            errors.append("balanced curriculum is not relation-family balanced")

    cf = training.get("gdm56_counterfactual_negative_fraction")
    wants_cf = curriculum in {
        "family-balanced-counterfactual",
        "family-balanced-counterfactual-domain",
    }
    if not isinstance(cf, (int, float)):
        errors.append("missing counterfactual fraction")
    elif wants_cf:
        expected = (int(negative) // 4) / int(negative) if negative else 0.0
        if abs(float(cf) - expected) > 1e-12:
            errors.append("counterfactual fraction mismatch")
    elif abs(float(cf)) > 1e-12:
        errors.append("counterfactual negatives leaked into non-counterfactual arm")

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
            errors.append("NO_MATCH recall floor does not match fixed 5pp budget")
        if abs(float(accepted_floor) - expected_accepted) > 1e-12:
            errors.append("accepted recall floor does not match fixed 5pp guardrail")
        if reported.get("nomatch_recall", -1) + 1e-12 < float(nomatch_floor):
            errors.append("NO_MATCH recall floor violated")
        if reported.get("accepted_status_recall", -1) + 1e-12 < float(accepted_floor):
            errors.append("accepted recall floor violated")
    else:
        errors.append("missing rescue recall floors")

    encoder = summary.get("encoder", {})
    head = encoder.get("head_training_reproducibility", {})
    if cfg.get("backbone") != "hash":
        if head.get("gdm56_execution_scope") != "capability_and_ambiguity_heads":
            errors.append("missing GDM56 execution scope")
        if head.get("canonical_gdm50_numerical_source_commit") != "62d2720800f5d98e1521e0200766a80e998ebe9b":
            errors.append("wrong canonical GDM50 numerical source")
        if head.get("canonical_gdm51_architecture_source_commit") != "f52baf82222947ca00139437e90fe9b5428bd9ae":
            errors.append("wrong canonical GDM51 architecture source")
        if head.get("canonical_gdm55_source_commit") != "db13313e7fd5021922768add76aaac617ec21208":
            errors.append("wrong canonical GDM55 source in execution metadata")
        if head.get("gdm56_change_scope") != (
            "ambiguity-head relation-family balance/factorization only; fixed canonical GDM55 architecture, numerical path and GDM54 5pp rescue arbitration"
        ):
            errors.append("wrong GDM56 execution change scope")

    if "Ledger/Session" not in summary.get("calibration_scope", ""):
        errors.append("wrong calibration domain scope")
    if "Agreement/Device" not in summary.get("secondary_holdout_scope", ""):
        errors.append("wrong holdout domain scope")
    if "GDM46-GDM55" not in summary.get("regression_scope", ""):
        errors.append("regression scope does not include canonical GDM55 holdout")

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
    parser.add_argument("--seeds", default="5601,5602")
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
    report["format"] = "gdm56-batch-report-v1"
    report["hypotheses"] = [
        "Canonical GDM55 showed relation-family expansion can recover additional ambiguity beyond matched optimizer exposure, but the effect is seed/task unstable.",
        "Equalizing positive exposure across role, lifecycle-time, representation and object-vs-supplier families tests whether curriculum balance improves cross-seed stability.",
        "Counterfactual negatives are isolated at a fixed 25% of the negative curriculum instead of bundled with lexical/domain changes.",
        "Training-only domain randomization is isolated while semantic relation-family counts and optimizer exposure remain fixed.",
        "Every arm uses the same canonical structured heads, explicit-NONE capability objective, numerical path and GDM54 5pp ambiguity-rescue arbitration.",
    ]
    report["interpretation_limits"] = [
        "Frozen DistilBERT plus learned feature-space capability and ambiguity heads; no transformer fine-tuning.",
        "GDM56 changes ambiguity-head relation-family sampling/factors only; capability objective, head architecture, 1e-4 feature quantum, numerical execution and status arbitration are fixed.",
        "Schema generation remains catalog projection plus deterministic SDL/Federation realization, not unconstrained schema invention.",
        "Agreement/Device is a synthetic secondary holdout, not enterprise OOD.",
        "Absolute metrics should be compared within this batch; prior batches use different fresh holdout domains.",
    ]
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
