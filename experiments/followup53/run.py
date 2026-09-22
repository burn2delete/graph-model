"""GDM53: accepted-recall-budgeted hierarchical calibration sensitivity.

Canonical GDM52 verified that status-specific sequential calibration can improve
NO_MATCH/AMBIGUOUS risk accuracy and sharply reduce incorrect publication, but it
recreates the multi-clause false-reject problem: fully risk-first thresholds lose too
much accepted recall, while an exact noninferiority floor collapses back to the joint
GDM51-style control. GDM53 keeps the frozen DistilBERT encoder, learned capability and
ambiguity heads, training objective, 1e-4 frozen-feature boundary, numerical execution,
and deterministic NO_MATCH-first request aggregation fixed. It changes only
validation-only threshold selection and measures the Pareto sensitivity to explicit
accepted-status recall budgets of 5, 10 and 15 percentage points.

No transformer fine-tuning occurs. Schema generation remains catalog projection plus
deterministic SDL/Federation realization. All compilation, tests, training,
validation and benchmarking belong in GitHub Actions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.followup50 import run as g50
from experiments.followup51 import run as g51
from experiments.followup52 import run as g52

FORMAT = "gdm53-measured-v1"
SEEDS_DEFAULT = "5301,5302"

_BASE = {
    "family": "explicit-none-plus-ambiguity",
    "head": "mlp",
    "structured": True,
    "ambiguous_weight": 1.0,
    "hard_negative": False,
}

ARMS = {
    "joint-structured-control": {
        **_BASE,
        "calibration_strategy": "joint-selective-hmean",
        "accepted_recall_budget_pp": 0,
    },
    "sequential-structured-control": {
        **_BASE,
        "calibration_strategy": "sequential-status-specific",
        "accepted_recall_budget_pp": None,
    },
    "budget-structured-5pp": {
        **_BASE,
        "calibration_strategy": "accepted-recall-budget",
        "accepted_recall_budget_pp": 5,
    },
    "budget-structured-10pp": {
        **_BASE,
        "calibration_strategy": "accepted-recall-budget",
        "accepted_recall_budget_pp": 10,
    },
    "budget-structured-15pp": {
        **_BASE,
        "calibration_strategy": "accepted-recall-budget",
        "accepted_recall_budget_pp": 15,
    },
}

CALIBRATION_LANGUAGE = [
    "public heading displayed for this resource",
    "timestamp when this resource was first durably persisted",
    "timestamp when this resource was most recently durably rewritten",
    "whole-number score stored for every review",
    "display names of people who authored the reviews",
    "display names of people who moderated the reviews",
    "registered name of the resource supplier",
    "persistent identifiers of people who authored the reviews",
]
CALIBRATION_UNSUPPORTED = [
    "physical rack coordinate assigned to this resource",
    "reason a security exception was approved for this resource",
    "customs declaration class assigned to this resource",
    "flag indicating this resource is subject to a preservation order",
]
CALIBRATION_AMBIGUOUS = [
    "review participant identity without choosing author or moderator",
    "persistence time without choosing original write or latest rewrite",
]
HOLDOUT_LANGUAGE = [
    "customer-facing heading visible for this record",
    "instant this record was first committed to durable storage",
    "instant this record was last changed in durable storage",
    "integer score attached to each review",
    "names of people who wrote each review",
    "names of people who moderated each review",
    "official supplier organization name for this record",
    "stable identifiers of people who wrote each review",
]
HOLDOUT_UNSUPPORTED = [
    "warehouse bin coordinate assigned to this record",
    "rationale for an approved policy exception on this record",
    "international tariff class assigned to this record",
    "flag indicating this record is under evidence preservation",
]
HOLDOUT_AMBIGUOUS = [
    "review participant identity without specifying writer versus moderator",
    "storage timestamp without specifying first commit versus latest rewrite",
]


def calibration_set(task):
    return g50._dataset(
        task,
        [("Repository", "repository"), ("Deployment", "deployment")],
        CALIBRATION_LANGUAGE,
        CALIBRATION_UNSUPPORTED,
        CALIBRATION_AMBIGUOUS,
        "gdm53-calibration",
    )


def fresh_holdout(task):
    return g50._dataset(
        task,
        [("Order", "order"), ("Payment", "payment")],
        HOLDOUT_LANGUAGE,
        HOLDOUT_UNSUPPORTED,
        HOLDOUT_AMBIGUOUS,
        "gdm53-holdout",
    )


def _joint_baseline(records):
    best = None
    for nt in g52._threshold_values(records, "none_margin"):
        for at in g52._threshold_values(records, "ambiguity_probability"):
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
    return best


def calibrate_budget(records, budget_pp):
    """Risk-first hierarchy subject to a validation-only accepted-recall budget."""
    if budget_pp not in {5, 10, 15}:
        raise ValueError(budget_pp)
    baseline = _joint_baseline(records)
    baseline_metrics = baseline[3]
    budget_fraction = float(budget_pp) / 100.0
    floor = max(0.0, float(baseline_metrics["accepted_status_recall"]) - budget_fraction)

    nt, none_metrics = g52._select_none_threshold(
        records,
        accepted_floor=floor,
        risk_first=True,
    )
    at, ambiguity_metrics = g52._select_ambiguity_threshold(
        records,
        nt,
        accepted_floor=floor,
        risk_first=True,
    )
    final_metrics = g51._cal_metrics(records, nt, at, True)
    if final_metrics["accepted_status_recall"] + 1e-12 < floor:
        raise AssertionError("accepted-recall budget floor was violated")

    calibration = {
        "mode": "explicit-none-plus-ambiguity-discriminator",
        "none_margin_threshold": nt,
        "ambiguity_probability_threshold": at,
        "selection_split": "validation",
        "calibration_scope": "expanded-validation-only",
        "threshold_selection": "accepted-recall-budget",
        "hierarchy_contract": "NONE gate calibrated as NO_MATCH vs non-NO_MATCH (accepted + AMBIGUOUS); ambiguity gate calibrated only on requests surviving NONE",
        "none_gate_validation_metrics": none_metrics,
        "ambiguity_gate_validation_metrics": ambiguity_metrics,
        "accepted_recall_budget_pp": budget_pp,
        "accepted_recall_budget_fraction": budget_fraction,
        "accepted_status_recall_floor": floor,
        "accepted_status_recall_floor_source": "same-record joint-selective-hmean accepted recall minus fixed validation sensitivity budget",
        "joint_baseline_thresholds": {
            "none_margin_threshold": baseline[1],
            "ambiguity_probability_threshold": baseline[2],
        },
        "joint_baseline_validation_metrics": baseline_metrics,
    }
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
    if strategy == "sequential-status-specific":
        return g52.calibrate_records(records, strategy)
    if strategy == "accepted-recall-budget":
        return calibrate_budget(records, int(arm["accepted_recall_budget_pp"]))
    raise ValueError(strategy)


def _output_root():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", default="artifacts/results")
    args, _ = parser.parse_known_args()
    return Path(args.output)


def _postprocess(root):
    """Replace inherited GDM52 bookkeeping with explicit GDM53 scope evidence."""
    for summary_path in root.rglob("summary.json"):
        directory = summary_path.parent
        config_path = directory / "config.json"
        training_path = directory / "training.json"
        config = json.loads(config_path.read_text())
        training = json.loads(training_path.read_text())
        summary = json.loads(summary_path.read_text())
        arm = ARMS[config["arm"]]

        config["research_question"] = (
            "how much accepted-status recall budget is required for hierarchical risk-first calibration "
            "to improve NO_MATCH/AMBIGUOUS rejection without recreating multi-clause false rejects"
        )
        config["accepted_recall_budget_pp"] = arm["accepted_recall_budget_pp"]
        training["gdm53_family"] = "accepted-recall-budget-sensitivity"
        training["gdm53_calibration_strategy"] = arm["calibration_strategy"]
        training["gdm53_accepted_recall_budget_pp"] = arm["accepted_recall_budget_pp"]
        training["canonical_gdm52_calibration_source_commit"] = "f0ca439c3bb567213108b55a3bd56e449425509b"
        training["selected_calibration"] = json.loads((directory / "calibration.json").read_text())["selected"]

        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        training_path.write_text(json.dumps(training, indent=2, sort_keys=True) + "\n")

        summary["config"] = config
        summary["training"] = training
        summary["calibration_scope"] = (
            "disjoint Repository/Deployment calibration-only cases; never used for optimizer updates or checkpoint selection"
        )
        summary["secondary_holdout_scope"] = (
            "new GDM53 Order/Payment synthetic holdout; never used for optimizer/checkpoint/calibration selection"
        )
        summary["regression_scope"] = "corrected public test plus inspected GDM46-GDM52 holdouts"
        summary["file_hashes"]["config.json"] = g52.file_hash(config_path)
        summary["file_hashes"]["training.json"] = g52.file_hash(training_path)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main():
    root = _output_root()
    old = {
        "FORMAT": g52.FORMAT,
        "SEEDS_DEFAULT": g52.SEEDS_DEFAULT,
        "ARMS": g52.ARMS,
        "calibration_set": g52.calibration_set,
        "fresh_holdout": g52.fresh_holdout,
        "calibrate_model": g52.calibrate_model,
        "g51_holdout": g52.g51.fresh_holdout,
    }
    old_g52_holdout = g52.fresh_holdout
    old_g51_holdout = g52.g51.fresh_holdout

    def regression_g51_plus_g52(task):
        rows51, refs51 = old_g51_holdout(task)
        rows52, refs52 = old_g52_holdout(task)
        return rows51 + rows52, {**refs51, **refs52}

    g52.FORMAT = FORMAT
    g52.SEEDS_DEFAULT = SEEDS_DEFAULT
    g52.ARMS = ARMS
    g52.calibration_set = calibration_set
    g52.fresh_holdout = fresh_holdout
    g52.calibrate_model = calibrate_model
    g52.g51.fresh_holdout = regression_g51_plus_g52
    try:
        g52.main()
        _postprocess(root)
    finally:
        g52.FORMAT = old["FORMAT"]
        g52.SEEDS_DEFAULT = old["SEEDS_DEFAULT"]
        g52.ARMS = old["ARMS"]
        g52.calibration_set = old["calibration_set"]
        g52.fresh_holdout = old["fresh_holdout"]
        g52.calibrate_model = old["calibrate_model"]
        g52.g51.fresh_holdout = old["g51_holdout"]


if __name__ == "__main__":
    main()
