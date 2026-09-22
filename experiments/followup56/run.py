"""GDM56: relation-family balance and stability under fixed canonical architecture.

Canonical GDM55 established that relation-family expansion can recover additional
held-out AMBIGUOUS cases relative to a volume-matched control, but the gain is strongly
seed/task dependent and the bundled full curriculum can reduce ambiguity detection
while increasing incorrect publication. GDM56 therefore keeps the canonical GDM51
structured learned-head architecture, canonical GDM50 numerical execution, explicit
NONE capability objective, canonical GDM54 5pp ambiguity-rescue arbitration, 1e-4
frozen-feature boundary and deterministic GraphQL/Federation realization fixed.
Only ambiguity-head relation-family sampling and negative/domain factors change.

Five matched-budget arms factor the next hypothesis:
  * relation-expanded-control: exact canonical GDM55 relation-expanded curriculum;
  * family-balanced-rescue: equalized exposure across the four evaluated relation families;
  * family-balanced-counterfactual: balanced positives + 25% matched counterfactual negatives;
  * family-balanced-domain: balanced positives split across base and training-only domains;
  * family-balanced-counterfactual-domain: both factors at the same update budget.

Fresh Ledger/Session cases are calibration-only. Fresh Agreement/Device cases are the
secondary holdout and never participate in optimizer updates, checkpoint selection or
calibration design. No transformer fine-tuning occurs. Schema generation remains
catalog projection plus deterministic SDL/Federation realization. All compilation,
tests, training, validation and benchmarking belong in GitHub Actions.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from experiments.measured.contracts import catalog, sha
from experiments.followup50 import run as g50
from experiments.followup51 import run as g51
from experiments.followup52 import run as g52
from experiments.followup53 import run as g53
from experiments.followup54 import run as g54
from experiments.followup55 import run as g55

FORMAT = "gdm56-measured-v1"
SEEDS_DEFAULT = "5601,5602"
CANONICAL_GDM55_SOURCE = "db13313e7fd5021922768add76aaac617ec21208"

_BASE = {
    "family": "explicit-none-plus-ambiguity",
    "head": "mlp",
    "structured": True,
    "ambiguous_weight": 1.0,
    "hard_negative": False,
    "calibration_strategy": "ambiguity-rescue",
    "status_arbitration": "ambiguity-rescue",
    "nomatch_recall_budget_pp": 5,
}

ARMS = {
    "relation-expanded-control": {**_BASE, "ambiguity_curriculum": "relation-expanded-control"},
    "family-balanced-rescue": {**_BASE, "ambiguity_curriculum": "family-balanced"},
    "family-balanced-counterfactual": {**_BASE, "ambiguity_curriculum": "family-balanced-counterfactual"},
    "family-balanced-domain": {**_BASE, "ambiguity_curriculum": "family-balanced-domain"},
    "family-balanced-counterfactual-domain": {
        **_BASE,
        "ambiguity_curriculum": "family-balanced-counterfactual-domain",
    },
}

# The evaluated relation families. The first three are the canonical GDM51 relations;
# object-vs-supplier is the GDM55 relation that first recovered that held-out family.
FAMILY_TEMPLATES = {
    "lifecycle-time": [g51.AMBIGUITY_TEMPLATES[0]],
    "role": [g51.AMBIGUITY_TEMPLATES[1]],
    "representation": [g51.AMBIGUITY_TEMPLATES[2]],
    "object-vs-supplier": [g55.RELATION_EXTRA_TEMPLATES[0]],
}

TRAINING_ONLY_DOMAINS = [
    ("Beacon", "beacon"),
    ("Parcel", "parcel"),
    ("Memo", "memo"),
    ("Roster", "roster"),
]

CALIBRATION_LANGUAGE = [
    "public heading attached to this entity",
    "moment the entity was first committed to durable storage",
    "moment the entity was most recently revised in durable storage",
    "integer score recorded for each review",
    "readable names of people who authored reviews",
    "readable names of people who moderated reviews",
    "registered company name of the supplying organization",
    "stable identifiers of review authors",
]
CALIBRATION_UNSUPPORTED = [
    "physical bin coordinate for this entity",
    "reason an emergency access exception was approved",
    "international tariff class for this entity",
    "flag saying this entity is subject to legal preservation",
]
CALIBRATION_AMBIGUITIES = [
    ("role", "review person identity where author versus moderator remains unspecified"),
    ("lifecycle-time", "storage timestamp without resolving first commit versus latest revision"),
    ("representation", "review writer identity without choosing readable name versus durable identifier"),
    ("object-vs-supplier", "display name without resolving entity title versus supplier company name"),
]

HOLDOUT_LANGUAGE = [
    "customer-facing heading for this item",
    "instant this item was first persisted",
    "instant this item was most recently persisted",
    "whole-number score on every review",
    "names of people who wrote each review",
    "names of people who moderated each review",
    "official business name of this item's supplier",
    "persistent IDs of people who wrote each review",
]
HOLDOUT_UNSUPPORTED = [
    "shelf slot assigned to this item",
    "justification for a temporary privileged-access waiver",
    "customs category assigned to this item",
    "flag indicating the item is under evidence retention",
]
HOLDOUT_AMBIGUITIES = [
    ("role", "person tied to a review without telling whether they wrote or moderated it"),
    ("lifecycle-time", "saved-at moment without telling whether it is original or most recent"),
    ("representation", "review writer identity without telling whether it is a readable name or persistent ID"),
    ("object-vs-supplier", "name related to the item without telling whether item title or supplier business"),
]

_ORIGINAL_TRAIN_MODEL = g55._ORIGINAL_TRAIN_MODEL
_ORIGINAL_AMBIGUITY_TRAINING_EXAMPLES = g55._ORIGINAL_AMBIGUITY_TRAINING_EXAMPLES


def _synthetic_rows(task):
    rows = []
    for entity, root in TRAINING_ONLY_DOMAINS:
        cat = catalog(entity, root)
        uid = sha(["gdm56-training-only-domain", task, entity, root])[:24]
        options = copy.deepcopy(cat["options"])
        random.Random(uid).shuffle(options)
        rows.append({
            "id": uid,
            "task": task,
            "request": ("Return " if task == "operation" else "Expose capabilities for ")
            + f"training-only curriculum for the {root}.",
            "catalog": {**cat, "options": options},
        })
    return rows


def _family_groups(rows, domain_randomized=False):
    groups = []
    for family in ("role", "lifecycle-time", "representation", "object-vs-supplier"):
        source = ("gdm56-domain-positive:" if domain_randomized else "gdm56-positive:") + family
        groups.append(g55._ambiguity_examples(rows, FAMILY_TEMPLATES[family], source))
    return groups


def _family_for_clause(clause):
    for family, templates in FAMILY_TEMPLATES.items():
        for _, _, phrases in templates:
            if clause in phrases:
                return family
    return None


def ambiguity_curriculum_examples(task, public, refs, curriculum):
    control = list(g55.ambiguity_curriculum_examples(task, public, refs, "relation-expanded"))
    target_positive = sum(int(item[2]) == 1 for item in control)
    target_negative = sum(int(item[2]) == 0 for item in control)
    if curriculum == "relation-expanded-control":
        return control

    baseline = list(_ORIGINAL_AMBIGUITY_TRAINING_EXAMPLES(task, public, refs))
    base_negative = [item for item in baseline if int(item[2]) == 0]
    if not base_negative:
        raise AssertionError("canonical negative ambiguity curriculum unexpectedly empty")

    rows = g55._representative_train_rows(task, public)
    base_groups = _family_groups(rows, domain_randomized=False)
    if curriculum in {"family-balanced-domain", "family-balanced-counterfactual-domain"}:
        synthetic_groups = _family_groups(_synthetic_rows(task), domain_randomized=True)
        positive_groups = [a + b for a, b in zip(base_groups, synthetic_groups)]
    else:
        positive_groups = base_groups
    positives = g55._balanced_take(positive_groups, target_positive)

    if curriculum in {"family-balanced-counterfactual", "family-balanced-counterfactual-domain"}:
        counter_rows = rows + (_synthetic_rows(task) if curriculum.endswith("-domain") else [])
        counterfactual = g55._counterfactual_examples(counter_rows, "gdm56-counterfactual-negative")
        counter_count = target_negative // 4
        negatives = (
            g55._cycle_take(base_negative, target_negative - counter_count)
            + g55._cycle_take(counterfactual, counter_count)
        )
    elif curriculum in {"family-balanced", "family-balanced-domain"}:
        negatives = g55._cycle_take(base_negative, target_negative)
    else:
        raise ValueError(curriculum)
    return positives + negatives


def _curriculum_receipt(examples, curriculum):
    positive = [item for item in examples if int(item[2]) == 1]
    negative = [item for item in examples if int(item[2]) == 0]
    sources = Counter(item[4] for item in examples)
    family_counts = Counter()
    for item in positive:
        family = _family_for_clause(item[1])
        if family is None and ":" in item[4]:
            family = item[4].split(":", 1)[1]
        if family:
            family_counts[family] += 1
    return {
        "gdm56_family": "relation-family-balance-stability",
        "gdm56_ambiguity_curriculum": curriculum,
        "gdm56_change_scope": (
            "ambiguity-head relation-family balance/factorization only; fixed canonical GDM55 architecture, numerical path and GDM54 5pp rescue arbitration"
        ),
        "canonical_gdm55_source_commit": CANONICAL_GDM55_SOURCE,
        "gdm56_curriculum_examples": len(examples),
        "gdm56_curriculum_positive_examples": len(positive),
        "gdm56_curriculum_negative_examples": len(negative),
        "gdm56_curriculum_unique_positive_clauses": len({item[1] for item in positive}),
        "gdm56_curriculum_unique_negative_clauses": len({item[1] for item in negative}),
        "gdm56_curriculum_sources": dict(sorted(sources.items())),
        "gdm56_positive_family_counts": dict(sorted(family_counts.items())),
        "gdm56_counterfactual_negative_fraction": (
            sum("counterfactual" in item[4] for item in negative) / len(negative) if negative else 0.0
        ),
        "gdm56_training_only_domains": [f"{entity}/{root}" for entity, root in TRAINING_ONLY_DOMAINS],
    }


def train_model(bundle, encoder, arm, task, public, refs, seed, epochs, directory):
    curriculum = arm["ambiguity_curriculum"]
    examples = ambiguity_curriculum_examples(task, public, refs, curriculum)
    old_examples = g51.ambiguity_training_examples
    g51.ambiguity_training_examples = lambda _task, _public, _refs: list(examples)
    try:
        receipt = _ORIGINAL_TRAIN_MODEL(
            bundle, encoder, arm, task, public, refs, seed, epochs, directory
        )
    finally:
        g51.ambiguity_training_examples = old_examples
    receipt.update(_curriculum_receipt(examples, curriculum))
    return receipt


def calibration_set(task):
    rows, refs = g50._dataset(
        task,
        [("Ledger", "ledger"), ("Session", "session")],
        CALIBRATION_LANGUAGE,
        CALIBRATION_UNSUPPORTED,
        [phrase for _, phrase in CALIBRATION_AMBIGUITIES],
        "gdm56-calibration",
    )
    return g55._tag_ambiguity_families(rows, refs, CALIBRATION_AMBIGUITIES)


def fresh_holdout(task):
    rows, refs = g50._dataset(
        task,
        [("Agreement", "agreement"), ("Device", "device")],
        HOLDOUT_LANGUAGE,
        HOLDOUT_UNSUPPORTED,
        [phrase for _, phrase in HOLDOUT_AMBIGUITIES],
        "gdm56-holdout",
    )
    return g55._tag_ambiguity_families(rows, refs, HOLDOUT_AMBIGUITIES)


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
            "does relation-family balancing improve seed-stable cross-domain AMBIGUOUS recovery while preserving NO_MATCH/publication behavior under the canonical architecture"
        )
        config["ambiguity_curriculum"] = arm["ambiguity_curriculum"]
        training["gdm56_family"] = "relation-family-balance-stability"
        training["gdm56_ambiguity_curriculum"] = arm["ambiguity_curriculum"]
        training["gdm56_change_scope"] = (
            "ambiguity-head relation-family balance/factorization only; fixed canonical GDM55 architecture, numerical path and GDM54 5pp rescue arbitration"
        )
        training["canonical_gdm55_source_commit"] = CANONICAL_GDM55_SOURCE
        training["selected_calibration"] = selected

        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        training_path.write_text(json.dumps(training, indent=2, sort_keys=True) + "\n")
        summary["config"] = config
        summary["training"] = training
        summary["calibration_scope"] = (
            "disjoint Ledger/Session calibration-only cases; never used for optimizer updates or checkpoint selection"
        )
        summary["secondary_holdout_scope"] = (
            "new GDM56 Agreement/Device synthetic holdout; never used for optimizer/checkpoint/calibration selection"
        )
        summary["regression_scope"] = "corrected public test plus inspected GDM46-GDM55 holdouts"
        summary["secondary_holdout_ambiguity_family_metrics"] = g55._ambiguity_family_metrics(
            directory / "predictions-holdout.jsonl"
        )
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
        "train_model": g51.train_model,
    }
    old_g51_holdout = g52.g51.fresh_holdout
    old_g52_holdout = g52.fresh_holdout
    old_g53_holdout = g53.fresh_holdout
    old_g54_holdout = g54.fresh_holdout
    old_g55_holdout = g55.fresh_holdout

    def regression_g51_through_g55(task):
        rows51, refs51 = old_g51_holdout(task)
        rows52, refs52 = old_g52_holdout(task)
        rows53, refs53 = old_g53_holdout(task)
        rows54, refs54 = old_g54_holdout(task)
        rows55, refs55 = old_g55_holdout(task)
        return (
            rows51 + rows52 + rows53 + rows54 + rows55,
            {**refs51, **refs52, **refs53, **refs54, **refs55},
        )

    g52.FORMAT = FORMAT
    g52.SEEDS_DEFAULT = SEEDS_DEFAULT
    g52.ARMS = ARMS
    g52.calibration_set = calibration_set
    g52.fresh_holdout = fresh_holdout
    g52.calibrate_model = g54.calibrate_model
    g52.g51.fresh_holdout = regression_g51_through_g55
    g51.request_from_evidence = g54.request_from_evidence
    g51.train_model = train_model
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
        g51.train_model = old["train_model"]


if __name__ == "__main__":
    main()
