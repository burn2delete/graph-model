"""GDM55: broaden ambiguity curriculum under fixed canonical status arbitration.

Canonical GDM54 verified that changing NO_MATCH/AMBIGUOUS arbitration alone does not
solve cross-domain ambiguity. Ambiguity-first ordering recovered only a few additional
AMBIGUOUS requests while destroying NO_MATCH recall, and validation-selected rescue
thresholds preserved NO_MATCH behavior without increasing AMBIGUOUS recall. GDM55
therefore keeps the canonical GDM51 structured ambiguity-head architecture, canonical
GDM50 numerical execution, explicit NONE capability objective, GDM54 5pp ambiguity
rescue arbitration, 1e-4 frozen-feature boundary and deterministic schema realization
fixed. Only the ambiguity-head *training curriculum* changes.

The bounded ablation separates more optimizer exposure from semantic diversity:
  * canonical-rescue-control: exact canonical GDM51 ambiguity curriculum;
  * volume-matched-control: repeat canonical examples to the expanded update budget;
  * lexical-expanded-rescue: broader paraphrases for the same canonical relations;
  * relation-expanded-rescue: new relation families at the same expanded update budget;
  * full-curriculum-rescue: lexical + relation + counterfactual + training-only domain
    randomization at the same expanded update budget.

Fresh Environment/Pipeline cases are calibration-only. Fresh Campaign/Profile cases
are the secondary holdout and never participate in optimizer updates, checkpoint
selection or calibration design. No transformer fine-tuning occurs. Schema generation
remains catalog projection plus deterministic SDL/Federation realization. All
compilation, tests, training, validation and benchmarking belong in GitHub Actions.
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

FORMAT = "gdm55-measured-v1"
SEEDS_DEFAULT = "5501,5502"
CANONICAL_GDM54_SOURCE = "031c7f6087eb356f8baeeedcb2bf25ce9fe35cb4"

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
    "canonical-rescue-control": {**_BASE, "ambiguity_curriculum": "canonical"},
    "volume-matched-control": {**_BASE, "ambiguity_curriculum": "volume-matched"},
    "lexical-expanded-rescue": {**_BASE, "ambiguity_curriculum": "lexical-expanded"},
    "relation-expanded-rescue": {**_BASE, "ambiguity_curriculum": "relation-expanded"},
    "full-curriculum-rescue": {**_BASE, "ambiguity_curriculum": "full-curriculum"},
}

# Additional positive paraphrases for the three canonical GDM51 ambiguity relations.
# Exact calibration/holdout phrases below are deliberately excluded.
LEXICAL_EXTRA_TEMPLATES = [
    (("createdAt",), ("updatedAt",), [
        "lifecycle timestamp without resolving first persistence versus later revision",
        "write time without choosing original creation or most recent modification",
        "record time where initial registration versus latest update is unspecified",
    ]),
    (("reviews", "author", "name"), ("reviews", "moderator", "name"), [
        "review participant display name with writer versus moderator role unresolved",
        "name of a review person without choosing author or moderation role",
        "review-person name where authorship versus moderation is unspecified",
    ]),
    (("reviews", "author", "name"), ("reviews", "author", "id"), [
        "review writer identity without choosing readable name or durable identifier",
        "author identity field with display name versus stable ID unresolved",
        "identity of the review author without specifying name or identifier form",
    ]),
]

# A genuinely different naming relation plus a cross-role/cross-representation relation.
RELATION_EXTRA_TEMPLATES = [
    (("title",), ("supplier", "name"), [
        "name associated with the object without choosing object title or supplier name",
        "display label without resolving the record title versus provider company name",
        "human-facing name where the entity label versus supplier business is unspecified",
        "name for this record without saying record title or supplying organization",
        "displayed naming field with object-title versus supplier-name unresolved",
    ]),
    (("reviews", "author", "id"), ("reviews", "moderator", "name"), [
        "review participant identity without choosing author ID or moderator name",
        "review-person field with writer identifier versus moderator display name unresolved",
        "identity attached to the review without resolving author ID versus moderator name",
        "review participant reference where author identifier versus moderator name is unspecified",
    ]),
]

# Matched unambiguous near-negatives teach which lexical evidence resolves each ambiguity.
COUNTERFACTUAL_NEGATIVES = [
    (("createdAt",), [
        "original creation timestamp, specifically not the latest update time",
        "first persisted timestamp rather than the most recent rewrite",
    ]),
    (("updatedAt",), [
        "most recent update timestamp, specifically not the initial creation time",
        "latest persisted rewrite time rather than the first write",
    ]),
    (("reviews", "author", "name"), [
        "review author display name, specifically not the moderator name",
        "writer name for each review rather than that review's moderator",
        "review author readable name rather than the author's stable identifier",
    ]),
    (("reviews", "moderator", "name"), [
        "review moderator display name, specifically not the author name",
        "moderator name for each review rather than that review's writer",
    ]),
    (("reviews", "author", "id"), [
        "stable identifier of each review author rather than the author's display name",
        "review writer ID, specifically not the readable author name",
    ]),
    (("title",), [
        "the object's own public title rather than the supplier company name",
    ]),
    (("supplier", "name"), [
        "the supplier company's legal name rather than the object's public title",
    ]),
]

TRAINING_ONLY_DOMAINS = [
    ("Asset", "asset"),
    ("Document", "document"),
    ("Package", "package"),
    ("Service", "service"),
]

CALIBRATION_LANGUAGE = [
    "public heading shown for this entity",
    "instant the entity first entered durable storage",
    "instant the entity was most recently rewritten",
    "whole-number score carried by each review",
    "display names of people who authored reviews",
    "display names of people who moderated reviews",
    "official company name of the supplier",
    "persistent identifiers of review authors",
]
CALIBRATION_UNSUPPORTED = [
    "physical rack coordinate allocated to the entity",
    "reason a security exemption was granted",
    "customs classification assigned to the entity",
    "flag saying the entity is under preservation lock",
]
CALIBRATION_AMBIGUITIES = [
    ("role", "review participant identity with author versus moderator role unresolved"),
    ("lifecycle-time", "durable lifecycle timestamp with original-write versus latest-rewrite unresolved"),
    ("representation", "review author identity without choosing display name versus stable identifier"),
    ("object-vs-supplier", "name associated with the entity without choosing entity title versus supplier organization name"),
]

HOLDOUT_LANGUAGE = [
    "customer-visible heading attached to this record",
    "moment this record was first saved durably",
    "moment this record was most recently saved durably",
    "integer score attached to each review",
    "names of the people who wrote each review",
    "names of the people who moderated each review",
    "registered company name for the record's supplier",
    "durable IDs of the people who wrote each review",
]
HOLDOUT_UNSUPPORTED = [
    "warehouse bay coordinate assigned to this record",
    "explanation for a temporary security exception",
    "cross-border commodity classification for this record",
    "flag indicating this record is under archive freeze",
]
HOLDOUT_AMBIGUITIES = [
    ("role", "identity for someone on a review where writer versus moderator is left open"),
    ("lifecycle-time", "record persistence moment where first save versus newest save is left open"),
    ("representation", "identity of the review writer where readable name versus persistent ID is left open"),
    ("object-vs-supplier", "label or name related to the record where record title versus provider company is left open"),
]

_ORIGINAL_AMBIGUITY_TRAINING_EXAMPLES = g51.ambiguity_training_examples
_ORIGINAL_TRAIN_MODEL = g51.train_model
_ORIGINAL_REQUEST_FROM_EVIDENCE = g51.request_from_evidence


def _representative_train_rows(task, public):
    rows = {}
    for row in public["train"]:
        if row["task"] != task:
            continue
        rows.setdefault(row["catalog"]["root"], row)
    return [rows[key] for key in sorted(rows)]


def _synthetic_rows(task):
    rows = []
    for entity, root in TRAINING_ONLY_DOMAINS:
        cat = catalog(entity, root)
        uid = sha(["gdm55-training-only-domain", task, entity, root])[:24]
        options = copy.deepcopy(cat["options"])
        random.Random(uid).shuffle(options)
        cat = {**cat, "options": options}
        rows.append({
            "id": uid,
            "task": task,
            "request": ("Return " if task == "operation" else "Expose capabilities for ") + f"training-only curriculum for the {root}.",
            "catalog": cat,
        })
    return rows


def _ambiguity_examples(rows, templates, source):
    examples = []
    for row in rows:
        available = {g51._suffix(option) for option in row["catalog"]["options"]}
        for left, right, phrases in templates:
            if tuple(left) not in available or tuple(right) not in available:
                continue
            for phrase in phrases:
                examples.append((row, phrase, 1, None, source))
    return examples


def _counterfactual_examples(rows, source):
    examples = []
    for row in rows:
        options = row["catalog"]["options"]
        available = {g51._suffix(option) for option in options}
        for suffix, phrases in COUNTERFACTUAL_NEGATIVES:
            if tuple(suffix) not in available:
                continue
            target = g51._option_for_suffix(options, suffix)["id"]
            for phrase in phrases:
                examples.append((row, phrase, 0, target, source))
    return examples


def _cycle_take(items, count):
    if count <= 0:
        return []
    if not items:
        raise AssertionError("cannot sample empty curriculum pool")
    return [items[index % len(items)] for index in range(count)]


def _balanced_take(groups, count):
    groups = [list(group) for group in groups if group]
    if not groups:
        raise AssertionError("no curriculum groups")
    offsets = [0] * len(groups)
    out = []
    while len(out) < count:
        for index, group in enumerate(groups):
            if len(out) >= count:
                break
            out.append(group[offsets[index] % len(group)])
            offsets[index] += 1
    return out


def ambiguity_curriculum_examples(task, public, refs, curriculum):
    baseline = list(_ORIGINAL_AMBIGUITY_TRAINING_EXAMPLES(task, public, refs))
    base_positive = [example for example in baseline if int(example[2]) == 1]
    base_negative = [example for example in baseline if int(example[2]) == 0]
    if not base_positive or not base_negative:
        raise AssertionError("canonical ambiguity curriculum unexpectedly empty")
    if curriculum == "canonical":
        return baseline

    rows = _representative_train_rows(task, public)
    lexical = _ambiguity_examples(rows, LEXICAL_EXTRA_TEMPLATES, "gdm55-lexical-positive")
    relation = _ambiguity_examples(rows, RELATION_EXTRA_TEMPLATES, "gdm55-relation-positive")
    counterfactual = _counterfactual_examples(rows, "gdm55-counterfactual-negative")
    synthetic_rows = _synthetic_rows(task)
    synthetic_positive = _ambiguity_examples(
        synthetic_rows,
        list(g51.AMBIGUITY_TEMPLATES) + LEXICAL_EXTRA_TEMPLATES + RELATION_EXTRA_TEMPLATES,
        "gdm55-domain-randomized-positive",
    )
    synthetic_negative = _counterfactual_examples(
        synthetic_rows, "gdm55-domain-randomized-counterfactual-negative"
    )

    # Every expanded arm uses the same optimizer-update budget. The volume-matched
    # control therefore distinguishes additional exposure from new semantic diversity.
    target_positive = len(base_positive) * 2
    target_negative = len(base_negative) + 24

    if curriculum == "volume-matched":
        positives = _cycle_take(base_positive, target_positive)
        negatives = _cycle_take(base_negative, target_negative)
    elif curriculum == "lexical-expanded":
        positives = _cycle_take(base_positive + lexical, target_positive)
        negatives = _cycle_take(base_negative, target_negative)
    elif curriculum == "relation-expanded":
        positives = _cycle_take(base_positive + relation, target_positive)
        negatives = _cycle_take(base_negative, target_negative)
    elif curriculum == "full-curriculum":
        positives = _balanced_take(
            [base_positive, lexical, relation, synthetic_positive], target_positive
        )
        negatives = _balanced_take(
            [base_negative, counterfactual, synthetic_negative], target_negative
        )
    else:
        raise ValueError(curriculum)
    return positives + negatives


def _curriculum_receipt(examples, curriculum):
    positive = [item for item in examples if int(item[2]) == 1]
    negative = [item for item in examples if int(item[2]) == 0]
    sources = Counter(item[4] for item in examples)
    return {
        "gdm55_family": "ambiguity-curriculum-generalization",
        "gdm55_ambiguity_curriculum": curriculum,
        "gdm55_change_scope": "ambiguity-head training curriculum only; fixed GDM54 5pp ambiguity rescue arbitration",
        "canonical_gdm54_source_commit": CANONICAL_GDM54_SOURCE,
        "gdm55_curriculum_examples": len(examples),
        "gdm55_curriculum_positive_examples": len(positive),
        "gdm55_curriculum_negative_examples": len(negative),
        "gdm55_curriculum_unique_positive_clauses": len({item[1] for item in positive}),
        "gdm55_curriculum_unique_negative_clauses": len({item[1] for item in negative}),
        "gdm55_curriculum_sources": dict(sorted(sources.items())),
        "gdm55_training_only_domains": [f"{entity}/{root}" for entity, root in TRAINING_ONLY_DOMAINS],
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


def _tag_ambiguity_families(rows, refs, definitions):
    lookup = {phrase: family for family, phrase in definitions}
    for row in rows:
        if refs[row["id"]]["status"] != "AMBIGUOUS":
            continue
        matches = [family for phrase, family in lookup.items() if phrase in row["request"]]
        if len(matches) != 1:
            raise AssertionError((row["request"], matches))
        row["ambiguity_family"] = matches[0]
    return rows, refs


def calibration_set(task):
    rows, refs = g50._dataset(
        task,
        [("Environment", "environment"), ("Pipeline", "pipeline")],
        CALIBRATION_LANGUAGE,
        CALIBRATION_UNSUPPORTED,
        [phrase for _, phrase in CALIBRATION_AMBIGUITIES],
        "gdm55-calibration",
    )
    return _tag_ambiguity_families(rows, refs, CALIBRATION_AMBIGUITIES)


def fresh_holdout(task):
    rows, refs = g50._dataset(
        task,
        [("Campaign", "campaign"), ("Profile", "profile")],
        HOLDOUT_LANGUAGE,
        HOLDOUT_UNSUPPORTED,
        [phrase for _, phrase in HOLDOUT_AMBIGUITIES],
        "gdm55-holdout",
    )
    return _tag_ambiguity_families(rows, refs, HOLDOUT_AMBIGUITIES)


def _ambiguity_family_metrics(path):
    totals = defaultdict(lambda: {"correct": 0, "total": 0, "predicted_statuses": Counter()})
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        family = record.get("public", {}).get("ambiguity_family")
        if not family:
            continue
        reference = record["reference"]
        prediction = record["prediction"]
        if reference.get("status") != "AMBIGUOUS":
            raise AssertionError("ambiguity family attached to non-AMBIGUOUS reference")
        bucket = totals[family]
        bucket["total"] += 1
        bucket["correct"] += int(prediction.get("status") == "AMBIGUOUS")
        bucket["predicted_statuses"][prediction.get("status")] += 1
    return {
        family: {
            "correct": values["correct"],
            "total": values["total"],
            "accuracy": values["correct"] / values["total"] if values["total"] else 0.0,
            "predicted_statuses": dict(sorted(values["predicted_statuses"].items())),
        }
        for family, values in sorted(totals.items())
    }


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
            "does broader ambiguity-head training diversity improve high-confidence cross-domain AMBIGUOUS rescue "
            "without changing the capability objective, head architecture, numerical path or status arbitration"
        )
        config["ambiguity_curriculum"] = arm["ambiguity_curriculum"]
        training["gdm55_family"] = "ambiguity-curriculum-generalization"
        training["gdm55_ambiguity_curriculum"] = arm["ambiguity_curriculum"]
        training["gdm55_change_scope"] = (
            "ambiguity-head training curriculum only; fixed GDM54 5pp ambiguity rescue arbitration"
        )
        training["canonical_gdm54_source_commit"] = CANONICAL_GDM54_SOURCE
        training["selected_calibration"] = selected

        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        training_path.write_text(json.dumps(training, indent=2, sort_keys=True) + "\n")
        summary["config"] = config
        summary["training"] = training
        summary["calibration_scope"] = (
            "disjoint Environment/Pipeline calibration-only cases; never used for optimizer updates or checkpoint selection"
        )
        summary["secondary_holdout_scope"] = (
            "new GDM55 Campaign/Profile synthetic holdout; never used for optimizer/checkpoint/calibration selection"
        )
        summary["regression_scope"] = "corrected public test plus inspected GDM46-GDM54 holdouts"
        summary["secondary_holdout_ambiguity_family_metrics"] = _ambiguity_family_metrics(
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

    def regression_g51_through_g54(task):
        rows51, refs51 = old_g51_holdout(task)
        rows52, refs52 = old_g52_holdout(task)
        rows53, refs53 = old_g53_holdout(task)
        rows54, refs54 = old_g54_holdout(task)
        return (
            rows51 + rows52 + rows53 + rows54,
            {**refs51, **refs52, **refs53, **refs54},
        )

    g52.FORMAT = FORMAT
    g52.SEEDS_DEFAULT = SEEDS_DEFAULT
    g52.ARMS = ARMS
    g52.calibration_set = calibration_set
    g52.fresh_holdout = fresh_holdout
    g52.calibrate_model = g54.calibrate_model
    g52.g51.fresh_holdout = regression_g51_through_g54
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
