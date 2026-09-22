"""Independent evidence gate for GDM54."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.followup51 import collect as g51collect
from experiments.followup52 import collect as g52collect
from .run import ARMS, FORMAT


def load_json(path):
    return json.loads(Path(path).read_text())


def verify_summary(path):
    # Reuse canonical GDM51 evidence checks for changed checkpoints, executable
    # GraphQL, Rover composition, prediction recounts and timing evidence.
    g51collect.ARMS = ARMS
    g51collect.FORMAT = FORMAT
    summary, detail = g51collect.verify_summary(path)
    errors = []
    cfg = summary.get("config", {})
    training = summary.get("training", {})
    calibration = load_json(path.parent / "calibration.json").get("selected", {})
    arm = ARMS.get(cfg.get("arm"), {})
    strategy = arm.get("calibration_strategy")
    arbitration = arm.get("status_arbitration")
    budget_pp = arm.get("nomatch_recall_budget_pp")

    if cfg.get("calibration_strategy") != strategy:
        errors.append("config calibration strategy mismatch")
    if cfg.get("status_arbitration") != arbitration:
        errors.append("config status arbitration mismatch")
    if cfg.get("nomatch_recall_budget_pp") != budget_pp:
        errors.append("config NO_MATCH recall budget mismatch")
    if training.get("gdm54_family") != "ambiguity-before-none-arbitration":
        errors.append("missing GDM54 family evidence")
    if training.get("gdm54_calibration_strategy") != strategy:
        errors.append("training calibration strategy mismatch")
    if training.get("gdm54_status_arbitration") != arbitration:
        errors.append("training status arbitration mismatch")
    if training.get("gdm54_nomatch_recall_budget_pp") != budget_pp:
        errors.append("training NO_MATCH recall budget mismatch")
    if training.get("canonical_gdm53_source_commit") != "97b8ceb7c3820690545873e9ced2854ad1e26cee":
        errors.append("wrong canonical GDM53 source")
    if calibration.get("threshold_selection") != strategy:
        errors.append("selected threshold strategy mismatch")
    if calibration.get("status_arbitration") != arbitration:
        errors.append("selected status arbitration mismatch")

    reported = training.get("selected_calibration_validation_metrics", {})
    if strategy == "ambiguity-rescue":
        if calibration.get("nomatch_recall_budget_pp") != budget_pp:
            errors.append("selected NO_MATCH budget mismatch")
        if not isinstance(calibration.get("ambiguity_rescue_threshold"), (int, float)):
            errors.append("missing ambiguity rescue threshold")
        baseline = calibration.get("joint_baseline_validation_metrics")
        nomatch_floor = calibration.get("nomatch_recall_floor")
        accepted_floor = calibration.get("accepted_status_recall_floor")
        if not isinstance(baseline, dict):
            errors.append("missing joint baseline validation metrics")
        if not isinstance(nomatch_floor, (int, float)) or not isinstance(accepted_floor, (int, float)):
            errors.append("missing rescue recall floors")
        else:
            expected_nomatch = max(0.0, float(baseline.get("nomatch_recall", -1)) - float(budget_pp) / 100.0)
            expected_accepted = max(0.0, float(baseline.get("accepted_status_recall", -1)) - 0.05)
            if abs(float(nomatch_floor) - expected_nomatch) > 1e-12:
                errors.append("NO_MATCH recall floor does not match budget")
            if abs(float(accepted_floor) - expected_accepted) > 1e-12:
                errors.append("accepted recall floor does not match fixed 5pp guardrail")
            if reported.get("nomatch_recall", -1) + 1e-12 < float(nomatch_floor):
                errors.append("NO_MATCH recall floor violated")
            if reported.get("accepted_status_recall", -1) + 1e-12 < float(accepted_floor):
                errors.append("accepted recall floor violated")
        if not isinstance(calibration.get("joint_baseline_thresholds"), dict):
            errors.append("missing joint baseline thresholds")
        if "preempt NONE" not in calibration.get("rescue_contract", ""):
            errors.append("missing ambiguity rescue contract")

    encoder = summary.get("encoder", {})
    head = encoder.get("head_training_reproducibility", {})
    if cfg.get("backbone") != "hash":
        if head.get("gdm54_execution_scope") != "capability_and_ambiguity_heads":
            errors.append("missing GDM54 execution scope")
        if head.get("canonical_gdm50_numerical_source_commit") != "62d2720800f5d98e1521e0200766a80e998ebe9b":
            errors.append("wrong canonical GDM50 numerical source")
        if head.get("canonical_gdm51_architecture_source_commit") != "f52baf82222947ca00139437e90fe9b5428bd9ae":
            errors.append("wrong canonical GDM51 architecture source")
        if head.get("canonical_gdm53_source_commit") != "97b8ceb7c3820690545873e9ced2854ad1e26cee":
            errors.append("wrong canonical GDM53 source in execution metadata")
        if head.get("gdm54_change_scope") != "validation-only ambiguity-before-NONE status arbitration":
            errors.append("wrong GDM54 change scope")

    if "Workspace/Build" not in summary.get("calibration_scope", ""):
        errors.append("wrong calibration domain scope")
    if "Ticket/Account" not in summary.get("secondary_holdout_scope", ""):
        errors.append("wrong holdout domain scope")
    if "GDM46-GDM53" not in summary.get("regression_scope", ""):
        errors.append("regression scope does not include canonical GDM53 holdout")

    if errors:
        raise AssertionError(path.as_posix() + ": " + "; ".join(sorted(set(errors))))
    return summary, detail


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("root")
    parser.add_argument("--seeds", default="5401,5402")
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
    report["format"] = "gdm54-batch-report-v1"
    report["hypotheses"] = [
        "Canonical GDM53 showed that accepted-recall budgets alone do not restore cross-domain AMBIGUOUS detection: the NO_MATCH-first upstream gate frequently consumes ambiguous requests.",
        "Using the exact same learned structured heads and joint thresholds, ambiguity-first arbitration directly tests whether status ordering rather than representation is the limiting factor.",
        "A separately validation-selected high-confidence ambiguity rescue threshold can allow ambiguity evidence to preempt NONE while bounding NO_MATCH recall loss to 0/5/10 percentage points.",
        "A fixed five-point accepted-status recall guardrail prevents rescue calibration from improving ambiguity by indiscriminately rejecting answerable requests.",
    ]
    report["interpretation_limits"] = [
        "Frozen DistilBERT plus learned feature-space capability and structured ambiguity heads; no transformer fine-tuning.",
        "GDM54 changes validation-only status arbitration and rescue calibration, not learned-head objectives, frozen-feature quantum or numerical execution.",
        "Schema generation remains catalog projection plus deterministic SDL/Federation realization, not unconstrained schema invention.",
        "The Ticket/Account secondary holdout is synthetic and not enterprise OOD.",
    ]
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
