"""GDM54: validation-only ambiguity-before-NONE arbitration.

Canonical GDM53 showed that accepted-recall budgets can expose an intermediate
selective-risk region, but its fresh Order/Payment holdout also exposed a sharper
bottleneck: AMBIGUOUS requests are usually consumed by the upstream NONE gate before
the learned structured ambiguity discriminator can act. GDM54 keeps the canonical
GDM51 structured learned heads, canonical GDM50 numerical execution, 1e-4 frozen
feature boundary, training objectives and schema realization fixed. It changes only
validation-only status arbitration over the already learned clause evidence.

The bounded experiment compares canonical NO_MATCH-first arbitration, ambiguity-first
arbitration using the same joint thresholds, and high-confidence ambiguity rescue
with 0/5/10 percentage-point validation NO_MATCH-recall budgets. Rescue arms also
require accepted-status recall to remain within five percentage points of the same
validation joint baseline. No holdout labels select thresholds.

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
from experiments.followup53 import run as g53

FORMAT = "gdm54-measured-v1"
SEEDS_DEFAULT = "5401,5402"
CANONICAL_GDM53_SOURCE = "97b8ceb7c3820690545873e9ced2854ad1e26cee"

_BASE = {
    "family": "explicit-none-plus-ambiguity",
    "head": "mlp",
    "structured": True,
    "ambiguous_weight": 1.0,
    "hard_negative": False,
}

ARMS = {
    "nomatch-first-control": {
        **_BASE,
        "calibration_strategy": "joint-thresholds-fixed",
        "status_arbitration": "nomatch-first",
        "nomatch_recall_budget_pp": None,
    },
    "ambiguity-first-control": {
        **_BASE,
        "calibration_strategy": "joint-thresholds-fixed",
        "status_arbitration": "ambiguity-first",
        "nomatch_recall_budget_pp": None,
    },
    "ambiguity-rescue-0pp": {
        **_BASE,
        "calibration_strategy": "ambiguity-rescue",
        "status_arbitration": "ambiguity-rescue",
        "nomatch_recall_budget_pp": 0,
    },
    "ambiguity-rescue-5pp": {
        **_BASE,
        "calibration_strategy": "ambiguity-rescue",
        "status_arbitration": "ambiguity-rescue",
        "nomatch_recall_budget_pp": 5,
    },
    "ambiguity-rescue-10pp": {
        **_BASE,
        "calibration_strategy": "ambiguity-rescue",
        "status_arbitration": "ambiguity-rescue",
        "nomatch_recall_budget_pp": 10,
    },
}

CALIBRATION_LANGUAGE = [
    "public title displayed for this object",
    "timestamp when the object was first durably written",
    "timestamp when the object was most recently durably updated",
    "whole-number score stored for each review",
    "display names of review authors",
    "display names of review moderators",
    "registered supplier company name",
    "stable identifiers of review authors",
]
CALIBRATION_UNSUPPORTED = [
    "physical shelf coordinate assigned to this object",
    "reason a temporary governance waiver was approved",
    "cross-border tariff category assigned to this object",
    "flag indicating the object is under discovery hold",
]
CALIBRATION_AMBIGUOUS = [
    "review participant identity without choosing author or moderator",
    "durable-write timestamp without choosing original or latest write",
]
HOLDOUT_LANGUAGE = [
    "customer-visible title for this record",
    "instant this record was first persisted",
    "instant this record was most recently persisted",
    "integer score recorded on every review",
    "names of people who authored each review",
    "names of people who moderated each review",
    "official supplier organization name for this record",
    "stable IDs of people who authored each review",
]
HOLDOUT_UNSUPPORTED = [
    "warehouse aisle coordinate assigned to this record",
    "explanation for an approved control exception",
    "international customs category assigned to this record",
    "flag indicating this record is under retention freeze",
]
HOLDOUT_AMBIGUOUS = [
    "review participant identity without specifying author versus moderator",
    "persistence timestamp without specifying initial versus latest write",
]


def calibration_set(task):
    return g50._dataset(
        task,
        [("Workspace", "workspace"), ("Build", "build")],
        CALIBRATION_LANGUAGE,
        CALIBRATION_UNSUPPORTED,
        CALIBRATION_AMBIGUOUS,
        "gdm54-calibration",
    )


def fresh_holdout(task):
    return g50._dataset(
        task,
        [("Ticket", "ticket"), ("Account", "account")],
        HOLDOUT_LANGUAGE,
        HOLDOUT_UNSUPPORTED,
        HOLDOUT_AMBIGUOUS,
        "gdm54-holdout",
    )


def request_from_evidence(evidence, calibration, learned=True):
    """Apply the GDM54 arbitration contract to already learned clause evidence."""
    if not learned or "status_arbitration" not in calibration:
        return _ORIGINAL_REQUEST_FROM_EVIDENCE(evidence, calibration, learned=learned)

    nt = float(calibration["none_margin_threshold"])
    at = float(calibration["ambiguity_probability_threshold"])
    mode = calibration["status_arbitration"]
    rescue = calibration.get("ambiguity_rescue_threshold")
    if mode == "ambiguity-rescue":
        if not isinstance(rescue, (int, float)):
            raise AssertionError("ambiguity rescue threshold missing")
        rescue = float(rescue)

    statuses = []
    selected = []
    for item in evidence:
        ambiguity = float(item["ambiguity_probability"])
        none_margin = float(item["none_margin"])
        if mode == "ambiguity-first":
            if ambiguity >= at:
                status = "AMBIGUOUS"
            elif none_margin >= nt:
                status = "NO_MATCH"
            else:
                status = "accepted"
                selected.append(item["best_real_id"])
        elif mode == "ambiguity-rescue":
            if ambiguity >= rescue:
                status = "AMBIGUOUS"
            elif none_margin >= nt:
                status = "NO_MATCH"
            elif ambiguity >= at:
                status = "AMBIGUOUS"
            else:
                status = "accepted"
                selected.append(item["best_real_id"])
        elif mode == "nomatch-first":
            if none_margin >= nt:
                status = "NO_MATCH"
            elif ambiguity >= at:
                status = "AMBIGUOUS"
            else:
                status = "accepted"
                selected.append(item["best_real_id"])
        else:
            raise ValueError(mode)
        statuses.append(status)

    if any(status == "NO_MATCH" for status in statuses):
        request_status = "NO_MATCH"
        selected = []
    elif any(status == "AMBIGUOUS" for status in statuses):
        request_status = "AMBIGUOUS"
        selected = []
    else:
        request_status = "accepted"
        selected = sorted(set(selected))
    return {"status": request_status, "selected": selected, "clause_statuses": statuses}


def _metrics(records, calibration):
    accepted_total = accepted_correct = 0
    nomatch_total = nomatch_correct = 0
    ambiguous_total = ambiguous_correct = 0
    exact = incorrect_publication = 0
    for record in records:
        pred = request_from_evidence(record["evidence"], calibration, learned=True)
        ref = record["reference"]
        ok = g51._exact(pred, ref)
        exact += int(ok)
        if ref["status"] == "accepted":
            accepted_total += 1
            accepted_correct += int(pred["status"] == "accepted")
        elif ref["status"] == "NO_MATCH":
            nomatch_total += 1
            nomatch_correct += int(pred["status"] == "NO_MATCH")
        elif ref["status"] == "AMBIGUOUS":
            ambiguous_total += 1
            ambiguous_correct += int(pred["status"] == "AMBIGUOUS")
        incorrect_publication += int(pred["status"] == "accepted" and not ok)
    accepted_recall = accepted_correct / accepted_total if accepted_total else 0.0
    nomatch_recall = nomatch_correct / nomatch_total if nomatch_total else 0.0
    ambiguous_recall = ambiguous_correct / ambiguous_total if ambiguous_total else 0.0
    risk_total = nomatch_total + ambiguous_total
    risk_accuracy = (nomatch_correct + ambiguous_correct) / risk_total if risk_total else 0.0
    selective_hmean = (
        2.0 * accepted_recall * risk_accuracy / (accepted_recall + risk_accuracy)
        if accepted_recall + risk_accuracy
        else 0.0
    )
    return {
        "accepted_status_recall": accepted_recall,
        "nomatch_recall": nomatch_recall,
        "ambiguous_recall": ambiguous_recall,
        "risk_accuracy": risk_accuracy,
        "selective_hmean": selective_hmean,
        "exact_accuracy": exact / len(records) if records else 0.0,
        "incorrect_publication_rate": incorrect_publication / len(records) if records else 0.0,
        "accepted_examples": accepted_total,
        "nomatch_examples": nomatch_total,
        "ambiguous_examples": ambiguous_total,
        "examples": len(records),
    }


def _joint_thresholds(records):
    baseline = g53._joint_baseline(records)
    return float(baseline[1]), float(baseline[2]), baseline[3]


def _base_calibration(records, arbitration):
    nt, at, _ = _joint_thresholds(records)
    calibration = {
        "mode": "explicit-none-plus-ambiguity-discriminator",
        "none_margin_threshold": nt,
        "ambiguity_probability_threshold": at,
        "selection_split": "validation",
        "calibration_scope": "expanded-validation-only",
        "threshold_selection": "joint-thresholds-fixed",
        "status_arbitration": arbitration,
        "joint_threshold_source": "same-record canonical GDM51 joint-selective-hmean calibration",
    }
    return calibration, _metrics(records, calibration)


def _rescue_calibration(records, budget_pp):
    if budget_pp not in {0, 5, 10}:
        raise ValueError(budget_pp)
    nt, at, _ = _joint_thresholds(records)
    baseline_calibration = {
        "mode": "explicit-none-plus-ambiguity-discriminator",
        "none_margin_threshold": nt,
        "ambiguity_probability_threshold": at,
        "selection_split": "validation",
        "calibration_scope": "expanded-validation-only",
        "threshold_selection": "joint-thresholds-fixed",
        "status_arbitration": "nomatch-first",
    }
    baseline_metrics = _metrics(records, baseline_calibration)
    nomatch_floor = max(0.0, baseline_metrics["nomatch_recall"] - float(budget_pp) / 100.0)
    accepted_floor = max(0.0, baseline_metrics["accepted_status_recall"] - 0.05)

    probabilities = [item["ambiguity_probability"] for row in records for item in row["evidence"]]
    candidates = list(g51._threshold_candidates(probabilities))
    candidates.append(max(probabilities) + 1.0)
    candidates = sorted(set(float(value) for value in candidates if float(value) + 1e-12 >= at))
    best = None
    for rescue_threshold in candidates:
        calibration = {
            "mode": "explicit-none-plus-ambiguity-discriminator",
            "none_margin_threshold": nt,
            "ambiguity_probability_threshold": at,
            "ambiguity_rescue_threshold": rescue_threshold,
            "selection_split": "validation",
            "calibration_scope": "expanded-validation-only",
            "threshold_selection": "ambiguity-rescue",
            "status_arbitration": "ambiguity-rescue",
        }
        metrics = _metrics(records, calibration)
        if metrics["nomatch_recall"] + 1e-12 < nomatch_floor:
            continue
        if metrics["accepted_status_recall"] + 1e-12 < accepted_floor:
            continue
        key = (
            metrics["ambiguous_recall"],
            metrics["risk_accuracy"],
            metrics["exact_accuracy"],
            metrics["accepted_status_recall"],
            -metrics["incorrect_publication_rate"],
            metrics["nomatch_recall"],
        )
        if best is None or key > best[0]:
            best = (key, rescue_threshold, calibration, metrics)
    if best is None:
        raise RuntimeError("no ambiguity rescue threshold satisfies validation recall floors")

    calibration = dict(best[2])
    calibration.update(
        {
            "nomatch_recall_budget_pp": budget_pp,
            "nomatch_recall_floor": nomatch_floor,
            "accepted_status_recall_floor": accepted_floor,
            "accepted_status_recall_floor_pp": 5,
            "joint_baseline_validation_metrics": baseline_metrics,
            "joint_baseline_thresholds": {
                "none_margin_threshold": nt,
                "ambiguity_probability_threshold": at,
            },
            "rescue_contract": "ambiguity may preempt NONE only above a separately validation-selected high-confidence threshold; normal ambiguity remains downstream of NONE",
        }
    )
    return calibration, best[3]


def calibrate_model(bundle, encoder, arm, rows, refs):
    records = [
        {
            "reference": refs[row["id"]],
            "evidence": g51.evidence_for_request(bundle, encoder, row, arm),
        }
        for row in rows
    ]
    if arm["calibration_strategy"] == "joint-thresholds-fixed":
        return _base_calibration(records, arm["status_arbitration"])
    if arm["calibration_strategy"] == "ambiguity-rescue":
        return _rescue_calibration(records, int(arm["nomatch_recall_budget_pp"]))
    raise ValueError(arm["calibration_strategy"])


def _output_root():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", default="artifacts/results")
    args, _ = parser.parse_known_args()
    return Path(args.output)


def _postprocess(root):
    for summary_path in root.rglob("summary.json"):
        directory = summary_path.parent
        config_path = directory / "config.json"
        training_path = directory / "training.json"
        config = json.loads(config_path.read_text())
        training = json.loads(training_path.read_text())
        summary = json.loads(summary_path.read_text())
        arm = ARMS[config["arm"]]
        selected = json.loads((directory / "calibration.json").read_text())["selected"]

        config["research_question"] = (
            "can validation-only ambiguity-before-NONE arbitration recover cross-domain AMBIGUOUS requests "
            "without materially sacrificing canonical NO_MATCH and accepted recall"
        )
        config["status_arbitration"] = arm["status_arbitration"]
        config["nomatch_recall_budget_pp"] = arm["nomatch_recall_budget_pp"]
        training["gdm54_family"] = "ambiguity-before-none-arbitration"
        training["gdm54_calibration_strategy"] = arm["calibration_strategy"]
        training["gdm54_status_arbitration"] = arm["status_arbitration"]
        training["gdm54_nomatch_recall_budget_pp"] = arm["nomatch_recall_budget_pp"]
        training["canonical_gdm53_source_commit"] = CANONICAL_GDM53_SOURCE
        training["selected_calibration"] = selected

        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        training_path.write_text(json.dumps(training, indent=2, sort_keys=True) + "\n")
        summary["config"] = config
        summary["training"] = training
        summary["calibration_scope"] = (
            "disjoint Workspace/Build calibration-only cases; never used for optimizer updates or checkpoint selection"
        )
        summary["secondary_holdout_scope"] = (
            "new GDM54 Ticket/Account synthetic holdout; never used for optimizer/checkpoint/calibration selection"
        )
        summary["regression_scope"] = "corrected public test plus inspected GDM46-GDM53 holdouts"
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
        "request_from_evidence": g51.request_from_evidence,
    }
    old_g51_holdout = g52.g51.fresh_holdout
    old_g52_holdout = g52.fresh_holdout
    old_g53_holdout = g53.fresh_holdout

    def regression_g51_through_g53(task):
        rows51, refs51 = old_g51_holdout(task)
        rows52, refs52 = old_g52_holdout(task)
        rows53, refs53 = old_g53_holdout(task)
        return rows51 + rows52 + rows53, {**refs51, **refs52, **refs53}

    g52.FORMAT = FORMAT
    g52.SEEDS_DEFAULT = SEEDS_DEFAULT
    g52.ARMS = ARMS
    g52.calibration_set = calibration_set
    g52.fresh_holdout = fresh_holdout
    g52.calibrate_model = calibrate_model
    g52.g51.fresh_holdout = regression_g51_through_g53
    g51.request_from_evidence = request_from_evidence
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
        g51.request_from_evidence = old["request_from_evidence"]


_ORIGINAL_REQUEST_FROM_EVIDENCE = g51.request_from_evidence


if __name__ == "__main__":
    main()
