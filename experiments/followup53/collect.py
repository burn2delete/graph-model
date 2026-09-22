"""Independent evidence gate for GDM53."""
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
    # Reuse the canonical GDM51 evidence gate for checkpoints, changed learned heads,
    # executable GraphQL, Rover composition, prediction recounts and timing evidence.
    g51collect.ARMS = ARMS
    g51collect.FORMAT = FORMAT
    summary, detail = g51collect.verify_summary(path)
    errors = []
    cfg = summary.get("config", {})
    training = summary.get("training", {})
    calibration = load_json(path.parent / "calibration.json").get("selected", {})
    arm = ARMS.get(cfg.get("arm"), {})
    strategy = arm.get("calibration_strategy")
    budget_pp = arm.get("accepted_recall_budget_pp")

    if cfg.get("calibration_strategy") != strategy:
        errors.append("config calibration strategy mismatch")
    if cfg.get("accepted_recall_budget_pp") != budget_pp:
        errors.append("config recall budget mismatch")
    if training.get("gdm53_family") != "accepted-recall-budget-sensitivity":
        errors.append("missing GDM53 family evidence")
    if training.get("gdm53_calibration_strategy") != strategy:
        errors.append("training calibration strategy mismatch")
    if training.get("gdm53_accepted_recall_budget_pp") != budget_pp:
        errors.append("training recall budget mismatch")
    if training.get("canonical_gdm52_calibration_source_commit") != "f0ca439c3bb567213108b55a3bd56e449425509b":
        errors.append("wrong canonical GDM52 source")
    if calibration.get("threshold_selection") != strategy:
        errors.append("selected threshold strategy mismatch")

    if strategy in {"sequential-status-specific", "accepted-recall-budget"}:
        if not isinstance(calibration.get("none_gate_validation_metrics"), dict):
            errors.append("missing NONE gate validation metrics")
        if not isinstance(calibration.get("ambiguity_gate_validation_metrics"), dict):
            errors.append("missing ambiguity gate validation metrics")

    if strategy == "accepted-recall-budget":
        floor = calibration.get("accepted_status_recall_floor")
        baseline = calibration.get("joint_baseline_validation_metrics")
        reported = training.get("selected_calibration_validation_metrics", {})
        if calibration.get("accepted_recall_budget_pp") != budget_pp:
            errors.append("selected recall budget mismatch")
        if not isinstance(floor, (int, float)) or not isinstance(baseline, dict):
            errors.append("missing budget floor evidence")
        else:
            expected_floor = max(0.0, float(baseline.get("accepted_status_recall", -1)) - float(budget_pp) / 100.0)
            if abs(float(floor) - expected_floor) > 1e-12:
                errors.append("accepted recall floor does not match budget")
            if reported.get("accepted_status_recall", -1) + 1e-12 < float(floor):
                errors.append("accepted recall budget floor violated")
        if not isinstance(calibration.get("joint_baseline_thresholds"), dict):
            errors.append("missing joint baseline thresholds")

    encoder = summary.get("encoder", {})
    head = encoder.get("head_training_reproducibility", {})
    if cfg.get("backbone") != "hash":
        if head.get("gdm53_execution_scope") != "capability_and_ambiguity_heads":
            errors.append("missing GDM53 execution scope")
        if head.get("canonical_gdm50_numerical_source_commit") != "62d2720800f5d98e1521e0200766a80e998ebe9b":
            errors.append("wrong canonical GDM50 numerical source")
        if head.get("canonical_gdm51_architecture_source_commit") != "f52baf82222947ca00139437e90fe9b5428bd9ae":
            errors.append("wrong canonical GDM51 architecture source")
        if head.get("canonical_gdm52_calibration_source_commit") != "f0ca439c3bb567213108b55a3bd56e449425509b":
            errors.append("wrong canonical GDM52 calibration source")
        if head.get("gdm53_change_scope") != "validation-only accepted-recall budget sensitivity":
            errors.append("wrong GDM53 change scope")

    if "Repository/Deployment" not in summary.get("calibration_scope", ""):
        errors.append("wrong calibration domain scope")
    if "Order/Payment" not in summary.get("secondary_holdout_scope", ""):
        errors.append("wrong holdout domain scope")
    if "GDM46-GDM52" not in summary.get("regression_scope", ""):
        errors.append("regression scope does not include canonical GDM52 holdout")

    if errors:
        raise AssertionError(path.as_posix() + ": " + "; ".join(sorted(set(errors))))
    return summary, detail


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("root")
    parser.add_argument("--seeds", default="5301,5302")
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
    report["format"] = "gdm53-batch-report-v1"
    report["hypotheses"] = [
        "GDM52 verified that unconstrained risk-first sequential calibration lowers incorrect publication but recreates accepted-recall collapse.",
        "Exact accepted-recall noninferiority is too restrictive and collapses to the joint control on the GDM52 holdout.",
        "A validation-only 5/10/15 percentage-point accepted-recall budget sweep can reveal whether a stable intermediate risk/recall Pareto region exists without changing learned heads.",
        "The structured ambiguity head is held fixed so any measured differences are attributable to threshold selection rather than representation or training changes.",
    ]
    report["interpretation_limits"] = [
        "Frozen DistilBERT plus learned feature-space capability and structured ambiguity heads; no transformer fine-tuning.",
        "GDM53 changes validation-only threshold selection sensitivity, not learned-head objectives, frozen-feature quantum or numerical execution.",
        "Request status remains deterministic NO_MATCH-first aggregation over clause-local evidence.",
        "Schema generation remains catalog projection plus deterministic SDL/Federation realization, not unconstrained schema invention.",
        "The Order/Payment secondary holdout is synthetic and not enterprise OOD.",
    ]
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
