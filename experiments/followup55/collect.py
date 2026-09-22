"""Independent evidence gate for GDM55 ambiguity-curriculum experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.followup51 import collect as g51collect
from experiments.followup52 import collect as g52collect
from .run import ARMS, FORMAT, _ambiguity_family_metrics

EXPECTED_FAMILIES = {"role", "lifecycle-time", "representation", "object-vs-supplier"}


def load_json(path):
    return json.loads(Path(path).read_text())


def verify_summary(path):
    # Canonical GDM51 verifies changed capability/ambiguity checkpoints, optimizer
    # evidence, real GraphQL execution, Rover composition, prediction recounts and
    # the cache-corrected timing boundary.
    g51collect.ARMS = ARMS
    g51collect.FORMAT = FORMAT
    summary, detail = g51collect.verify_summary(path)
    errors = []
    cfg = summary.get("config", {})
    training = summary.get("training", {})
    calibration = load_json(path.parent / "calibration.json").get("selected", {})
    arm = ARMS.get(cfg.get("arm"), {})
    curriculum = arm.get("ambiguity_curriculum")

    if cfg.get("ambiguity_curriculum") != curriculum:
        errors.append("config ambiguity curriculum mismatch")
    if cfg.get("calibration_strategy") != "ambiguity-rescue":
        errors.append("GDM55 must keep fixed ambiguity-rescue calibration")
    if cfg.get("status_arbitration") != "ambiguity-rescue":
        errors.append("GDM55 must keep fixed ambiguity-rescue arbitration")
    if cfg.get("nomatch_recall_budget_pp") != 5:
        errors.append("GDM55 must keep fixed 5pp NO_MATCH budget")

    if training.get("gdm55_family") != "ambiguity-curriculum-generalization":
        errors.append("missing GDM55 family evidence")
    if training.get("gdm55_ambiguity_curriculum") != curriculum:
        errors.append("training ambiguity curriculum mismatch")
    if training.get("canonical_gdm54_source_commit") != "031c7f6087eb356f8baeeedcb2bf25ce9fe35cb4":
        errors.append("wrong canonical GDM54 source")
    if training.get("gdm55_change_scope") != "ambiguity-head training curriculum only; fixed GDM54 5pp ambiguity rescue arbitration":
        errors.append("wrong GDM55 change scope")
    if training.get("gdm55_curriculum_examples") != training.get("ambiguity_train_examples"):
        errors.append("curriculum example count does not match optimizer curriculum")
    positive = training.get("gdm55_curriculum_positive_examples")
    negative = training.get("gdm55_curriculum_negative_examples")
    total = training.get("gdm55_curriculum_examples")
    if not all(isinstance(value, int) and value > 0 for value in (positive, negative, total)):
        errors.append("invalid GDM55 curriculum counts")
    elif positive + negative != total:
        errors.append("GDM55 positive/negative curriculum counts do not sum")
    sources = training.get("gdm55_curriculum_sources")
    if not isinstance(sources, dict) or not sources:
        errors.append("missing GDM55 curriculum source counts")
    if curriculum == "full-curriculum":
        source_text = " ".join(sorted(sources)) if isinstance(sources, dict) else ""
        for token in ("lexical", "relation", "counterfactual", "domain-randomized"):
            if token not in source_text:
                errors.append("full curriculum missing source family " + token)

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
        if head.get("gdm55_execution_scope") != "capability_and_ambiguity_heads":
            errors.append("missing GDM55 execution scope")
        if head.get("canonical_gdm50_numerical_source_commit") != "62d2720800f5d98e1521e0200766a80e998ebe9b":
            errors.append("wrong canonical GDM50 numerical source")
        if head.get("canonical_gdm51_architecture_source_commit") != "f52baf82222947ca00139437e90fe9b5428bd9ae":
            errors.append("wrong canonical GDM51 architecture source")
        if head.get("canonical_gdm54_source_commit") != "031c7f6087eb356f8baeeedcb2bf25ce9fe35cb4":
            errors.append("wrong canonical GDM54 source in execution metadata")
        if head.get("gdm55_change_scope") != "ambiguity-head training curriculum only; fixed GDM54 5pp ambiguity rescue arbitration":
            errors.append("wrong GDM55 execution change scope")

    if "Environment/Pipeline" not in summary.get("calibration_scope", ""):
        errors.append("wrong calibration domain scope")
    if "Campaign/Profile" not in summary.get("secondary_holdout_scope", ""):
        errors.append("wrong holdout domain scope")
    if "GDM46-GDM54" not in summary.get("regression_scope", ""):
        errors.append("regression scope does not include canonical GDM54 holdout")

    reported_families = summary.get("secondary_holdout_ambiguity_family_metrics")
    recomputed_families = _ambiguity_family_metrics(path.parent / "predictions-holdout.jsonl")
    if reported_families != recomputed_families:
        errors.append("ambiguity-family metrics do not recount from predictions")
    if set(recomputed_families) != EXPECTED_FAMILIES:
        errors.append("fresh holdout does not contain exact four ambiguity families")
    if any(values.get("total", 0) <= 0 for values in recomputed_families.values()):
        errors.append("empty ambiguity-family holdout bucket")

    if errors:
        raise AssertionError(path.as_posix() + ": " + "; ".join(sorted(set(errors))))
    return summary, detail


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("root")
    parser.add_argument("--seeds", default="5501,5502")
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
    report["format"] = "gdm55-batch-report-v1"
    report["hypotheses"] = [
        "Canonical GDM54 established that status arbitration alone cannot recover AMBIGUOUS requests without severe NO_MATCH loss.",
        "A volume-matched control distinguishes additional ambiguity-head optimizer exposure from genuinely broader semantic curriculum coverage.",
        "Lexical expansion tests paraphrase generalization over the same three canonical relation families; relation expansion tests a wider ambiguity relation family set.",
        "The full curriculum combines lexical and relation diversity with matched disambiguating counterfactual negatives and training-only domain randomization while keeping the expanded update budget fixed.",
        "Every arm uses the same canonical structured head, explicit-NONE capability objective and GDM54 5pp high-confidence ambiguity-rescue arbitration.",
    ]
    report["interpretation_limits"] = [
        "Frozen DistilBERT plus learned feature-space capability and ambiguity heads; no transformer fine-tuning.",
        "GDM55 changes ambiguity-head training curriculum only; capability objective, head architecture, 1e-4 feature quantum, numerical execution and status arbitration are fixed.",
        "Schema generation remains catalog projection plus deterministic SDL/Federation realization, not unconstrained schema invention.",
        "Campaign/Profile is a synthetic secondary holdout, not enterprise OOD.",
        "Absolute metrics should be compared within this batch; prior batches use different fresh holdout domains.",
    ]
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
