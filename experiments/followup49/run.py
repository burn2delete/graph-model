"""GDM49 measured pairwise open-set matching follow-up.

GDM48 showed that relative/listwise clause scores can recover answerable capabilities,
but open-set risk handling remains unstable: softmax probabilities are forced to name a
winner even when no catalog capability matches, and a single global threshold loses
recall as clause count grows. GDM49 keeps frozen DistilBERT + clause decomposition
fixed and tests an independent binary clause-to-capability matcher. Each catalog
option receives an absolute match score. Zero matched options means NO_MATCH, more
than one matched option means AMBIGUOUS, and exactly one match per clause means the
request is accepted.

All compilation, training, evaluation and timing execute only in GitHub Actions. The
transformer remains frozen; learned modules are feature-space adapters/heads, not
transformer fine-tuning. Schema output remains catalog projection plus deterministic
SDL/Federation realization, not unconstrained schema invention.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
from pathlib import Path
import random
import resource
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.measured.contracts import catalog, judge, sha
from experiments.measured.evidence import Compositor, aggregate, file_hash, timing, write_json
from experiments.measured.models import Encoder, Head, state_hash
from experiments.followup45.run import clauses, pair_features
from experiments.followup46.run import corrected_dataset, current_rss_kib, task_prefix, validate_emission
from experiments.followup47 import run as g47
from experiments.followup48 import run as g48

FORMAT = "gdm49-measured-v1"
SEEDS_DEFAULT = "4901,4902"

ARMS = {
    "listwise-hybrid-control": {
        "family": "listwise-control",
        "loss": "gdm48-cross-entropy-hard-negative",
        "calibration": "public-validation",
    },
    "binary-bce-publiccal": {
        "family": "binary-matcher",
        "loss": "balanced-bce",
        "hardneg": False,
        "focal": False,
        "calibration": "public-validation",
    },
    "binary-bce-expandedcal": {
        "family": "binary-matcher",
        "loss": "balanced-bce",
        "hardneg": False,
        "focal": False,
        "calibration": "expanded-validation-only",
    },
    "binary-hardneg-expandedcal": {
        "family": "binary-matcher",
        "loss": "balanced-bce+hard-negative-margin",
        "hardneg": True,
        "focal": False,
        "calibration": "expanded-validation-only",
    },
    "binary-focal-expandedcal": {
        "family": "binary-matcher",
        "loss": "balanced-focal-bce",
        "hardneg": False,
        "focal": True,
        "calibration": "expanded-validation-only",
    },
}

SEMANTIC_SUFFIXES = g47.SEMANTIC_SUFFIXES

# Training-only language. These phrases are intentionally distinct from GDM47/GDM48
# holdouts, the calibration-only phrases below, and the GDM49 secondary holdout.
AMBIGUOUS_TRAINING = [
    (
        "display name of a review participant without saying whether they wrote or moderated it",
        [("reviews", "author", "name"), ("reviews", "moderator", "name")],
    ),
    (
        "record timestamp without saying whether initial creation or latest revision is intended",
        [("createdAt",), ("updatedAt",)],
    ),
]
NO_MATCH_TRAINING = [
    "postal destination associated with the record",
    "reason the record was archived",
    "shipping charge attached to the record",
    "binary flag showing whether the record was deleted",
]

CALIBRATION_LANGUAGE = [
    "caption used as the public-facing name of the record",
    "time when the record was originally committed to storage",
    "time when the record was most recently modified in storage",
    "integer evaluation value on every review",
    "names of the people who authored the reviews",
    "names of the people who moderated the reviews",
    "registered company name of the record provider",
    "stable IDs of the people who authored the reviews",
]
CALIBRATION_UNSUPPORTED = [
    "warehouse aisle containing the record",
    "reason the record was cancelled",
    "tax amount associated with the record",
]
CALIBRATION_AMBIGUOUS = [
    "name of the review participant without choosing author or moderator",
    "record timestamp without choosing first creation or latest revision",
]

HOLDOUT_LANGUAGE = [
    "customer-facing caption attached to the record",
    "instant the record first entered durable storage",
    "instant of the newest durable modification to the record",
    "whole-number evaluation attached to every review",
    "names of the people who composed the reviews",
    "names of the people who supervised the reviews",
    "legal company name of the organization supplying the record",
    "persistent identifiers of the people who composed the reviews",
]
HOLDOUT_UNSUPPORTED = [
    "postal destination where the record should be delivered",
    "explanation for why the record was removed",
    "monetary shipping fee charged for the record",
    "boolean indicating whether the record has been archived",
]
HOLDOUT_AMBIGUOUS = [
    "name of a review participant where writer versus moderator is unspecified",
    "record timestamp where original creation versus newest revision is unspecified",
]


def _suffix(option):
    return tuple(option["path"][1:])


def _option_for_suffix(options, suffix):
    matches = [o for o in options if _suffix(o) == tuple(suffix)]
    if len(matches) != 1:
        raise AssertionError((suffix, len(matches)))
    return matches[0]


def _copy_catalog(base, options=None):
    out = {k: copy.deepcopy(v) for k, v in base.items() if k != "options"}
    out["options"] = copy.deepcopy(base["options"] if options is None else options)
    return out


class PairBundle(nn.Module):
    def __init__(self, pair_dim: int):
        super().__init__()
        self.matcher = Head(pair_dim, adapter=True)


def _subhash(module):
    return state_hash(copy.deepcopy(module.state_dict()))


def _pair_examples_from_rows(task, rows, refs):
    examples = []
    for row in rows:
        if row["task"] != task:
            continue
        ref = refs[row["id"]]
        parts = clauses(row)
        if ref["status"] == "accepted":
            if len(parts) != len(ref["paths"]):
                raise AssertionError("accepted clause/reference cardinality mismatch")
            for clause, target in zip(parts, ref["paths"]):
                examples.append((row, clause, {target}, "accepted"))
        elif ref["status"] == "NO_MATCH":
            for clause in parts:
                examples.append((row, clause, set(), "NO_MATCH"))
        # Existing AMBIGUOUS rows do not expose the intended competing targets. Do not
        # invent them from hidden test labels. Explicit training-only ambiguous examples
        # with known positive sets are added separately below.
    return examples


def training_curriculum(task, train_rows):
    catalogs = {}
    for row in train_rows:
        if row["task"] != task:
            continue
        key = (row["catalog"]["entity"], row["catalog"]["root"])
        catalogs.setdefault(key, copy.deepcopy(row["catalog"]))
    examples = []
    # Accepted semantic paraphrases from the earlier training-only curriculum.
    for row, ref in g47.semantic_curriculum(task, train_rows):
        parts = clauses(row)
        if len(parts) != 1 or len(ref["paths"]) != 1:
            raise AssertionError("semantic curriculum must be atomic")
        examples.append((row, parts[0], {ref["paths"][0]}, "accepted-curriculum"))
    for (entity, root), source in sorted(catalogs.items()):
        options = source["options"]
        for phrase, suffixes in AMBIGUOUS_TRAINING:
            positives = {_option_for_suffix(options, suffix)["id"] for suffix in suffixes}
            request = task_prefix(task) + phrase + f" for the {root}."
            uid = sha(["gdm49-train-ambiguous", task, entity, phrase])[:24]
            row = {"id": uid, "task": task, "request": request, "catalog": _copy_catalog(source)}
            examples.append((row, phrase, positives, "ambiguous-curriculum"))
        for phrase in NO_MATCH_TRAINING:
            request = task_prefix(task) + phrase + f" for the {root}."
            uid = sha(["gdm49-train-nomatch", task, entity, phrase])[:24]
            row = {"id": uid, "task": task, "request": request, "catalog": _copy_catalog(source)}
            examples.append((row, phrase, set(), "nomatch-curriculum"))
    return examples


def _binary_loss(bundle, encoder, item, clause, positives, arm):
    x, opts, _ = pair_features(encoder, item, clause)
    logits = bundle.matcher(x)
    labels = torch.tensor([float(o["id"] in positives) for o in opts], dtype=torch.float32)
    positive_count = int(labels.sum().item())
    negative_count = len(labels) - positive_count
    if positive_count:
        pos_weight = max(1.0, negative_count / max(1, positive_count))
        base = F.binary_cross_entropy_with_logits(
            logits, labels, reduction="none", pos_weight=torch.tensor(pos_weight)
        )
    else:
        base = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    if arm.get("focal"):
        probs = torch.sigmoid(logits)
        pt = probs * labels + (1.0 - probs) * (1.0 - labels)
        base = base * (1.0 - pt).pow(2.0)
    loss = base.mean()
    if arm.get("hardneg"):
        neg = logits[labels == 0]
        pos = logits[labels == 1]
        if len(neg):
            hardest = neg.max()
            if len(pos):
                loss = loss + 0.40 * F.softplus(hardest - pos.min() + 0.75)
            else:
                loss = loss + 0.25 * F.softplus(hardest + 1.0)
    return loss


def _accepted_top1_accuracy(bundle, encoder, rows, refs, task):
    correct = total = 0
    bundle.eval()
    with torch.no_grad():
        for row in rows:
            if row["task"] != task:
                continue
            ref = refs[row["id"]]
            if ref["status"] != "accepted":
                continue
            parts = clauses(row)
            if len(parts) != len(ref["paths"]):
                continue
            pred = []
            for clause in parts:
                x, opts, _ = pair_features(encoder, row, clause)
                logits = bundle.matcher(x)
                pred.append(opts[int(torch.argmax(logits))]["id"])
            total += 1
            correct += int(set(pred) == set(ref["paths"]))
    return correct, total


def _state_receipt(bundle, initial, selected, updates, selected_updates, selected_epoch, max_grad, train_examples):
    initial_hash = state_hash(initial)
    selected_hash = state_hash(selected)
    initial_matcher_hash = state_hash({k: v for k, v in initial.items() if k.startswith("matcher.")})
    selected_matcher_hash = state_hash({k: v for k, v in selected.items() if k.startswith("matcher.")})
    if initial_hash == selected_hash or initial_matcher_hash == selected_matcher_hash or max_grad <= 0:
        raise RuntimeError("Pairwise matcher training did not change selected weights")
    return {
        "optimizer_updates": updates,
        "matcher_optimizer_updates": updates,
        "selected_matcher_optimizer_updates": selected_updates,
        "selected_epoch": selected_epoch,
        "max_gradient_norm": max_grad,
        "initial_state_hash": initial_hash,
        "selected_state_hash": selected_hash,
        "initial_matcher_hash": initial_matcher_hash,
        "selected_matcher_hash": selected_matcher_hash,
        "train_pair_examples": train_examples,
        "independent_backbone_finetuning": False,
        "adapter_location": "frozen-embedding feature space",
        "risk_training_kind": "independent binary clause-option match supervision",
    }


def train_binary(bundle, encoder, arm, task, public, refs, seed, epochs, directory):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    initial = copy.deepcopy(bundle.state_dict())
    torch.save({"state": initial, "arm": arm, "task": task}, directory / "initial.pt")
    train_rows = [r for r in public["train"] if r["task"] == task]
    val_rows = [r for r in public["validation"] if r["task"] == task]
    base_examples = _pair_examples_from_rows(task, train_rows, refs["train"])
    curriculum = training_curriculum(task, train_rows)
    examples = base_examples + curriculum
    optimizer = torch.optim.AdamW(bundle.matcher.parameters(), lr=8e-4, weight_decay=1e-3)
    best_state = None; best_key = None; selected_updates = selected_epoch = 0
    updates = 0; max_grad = 0.0; history = []
    for epoch in range(1, epochs + 1):
        order = list(examples); random.Random(seed + epoch).shuffle(order)
        losses = []; bundle.train()
        for row, clause, positives, source in order:
            optimizer.zero_grad(set_to_none=True)
            loss = _binary_loss(bundle, encoder, row, clause, positives, arm)
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(bundle.matcher.parameters(), 5.0))
            if not math.isfinite(norm):
                raise FloatingPointError("nonfinite matcher gradient")
            optimizer.step(); updates += 1; max_grad = max(max_grad, norm); losses.append(float(loss.detach()))
        correct, total = _accepted_top1_accuracy(bundle, encoder, val_rows, refs["validation"], task)
        entry = {"epoch": epoch, "loss": float(np.mean(losses)), "optimizer_updates": updates,
                 "accepted_validation_top1_correct": correct, "accepted_validation_top1_total": total}
        history.append(entry)
        key = (correct / total if total else 0.0, -entry["loss"])
        if best_key is None or key > best_key:
            best_key = key; best_state = copy.deepcopy(bundle.state_dict())
            selected_updates = updates; selected_epoch = epoch
    bundle.load_state_dict(best_state)
    selected = copy.deepcopy(bundle.state_dict())
    torch.save({"state": selected, "arm": arm, "task": task, "encoder": encoder.meta, "seed": seed}, directory / "selected.pt")
    receipt = _state_receipt(bundle, initial, selected, updates, selected_updates, selected_epoch, max_grad, len(examples))
    write_json(directory / "training.json", {
        "family": "binary-matcher", "history": history,
        "base_pair_examples": len(base_examples), "curriculum_pair_examples": len(curriculum),
        **receipt,
    })
    return receipt


def _make_case(task, entity, root, phrase, status, targets=None, remove_suffix=None, tag="case"):
    base = catalog(entity, root); options = copy.deepcopy(base["options"])
    if remove_suffix is not None:
        remove_id = _option_for_suffix(options, remove_suffix)["id"]
        options = [o for o in options if o["id"] != remove_id]
    request = task_prefix(task) + phrase + f" for the {root}."
    uid = sha(["gdm49", tag, task, entity, phrase, status, remove_suffix])[:24]
    random.Random(uid).shuffle(options)
    row = {"id": uid, "task": task, "request": request, "catalog": _copy_catalog(base, options)}
    paths = []
    if targets:
        paths = [_option_for_suffix(base["options"], suffix)["id"] for suffix in targets]
    ref = {"status": status, "paths": paths, "fixture_seeds": [13, 47, 101]}
    return row, ref


def calibration_set(task):
    """Calibration-only synthetic cases; never used for gradient/checkpoint selection."""
    domains = [("Asset", "asset"), ("Booking", "booking")]
    rows, refs = [], {}
    combos = [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (0, 1), (4, 5), (3, 7)]
    for entity, root in domains:
        base = catalog(entity, root)
        for combo in combos:
            phrase = "; ".join(CALIBRATION_LANGUAGE[i] for i in combo)
            request = task_prefix(task) + phrase + f" for the {root}."
            uid = sha(["gdm49-cal", task, entity, combo, phrase])[:24]
            options = copy.deepcopy(base["options"]); random.Random(uid).shuffle(options)
            row = {"id": uid, "task": task, "request": request, "catalog": _copy_catalog(base, options)}
            refs[uid] = {"status": "accepted",
                         "paths": [_option_for_suffix(base["options"], SEMANTIC_SUFFIXES[i])["id"] for i in combo],
                         "fixture_seeds": [13, 47, 101]}
            rows.append(row)
        for i, suffix in enumerate(SEMANTIC_SUFFIXES):
            row, ref = _make_case(task, entity, root, CALIBRATION_LANGUAGE[i], "NO_MATCH", remove_suffix=suffix, tag="cal-missing")
            rows.append(row); refs[row["id"]] = ref
        for phrase in CALIBRATION_UNSUPPORTED:
            row, ref = _make_case(task, entity, root, phrase, "NO_MATCH", tag="cal-unsupported")
            rows.append(row); refs[row["id"]] = ref
        for phrase in CALIBRATION_AMBIGUOUS:
            row, ref = _make_case(task, entity, root, phrase, "AMBIGUOUS", tag="cal-ambiguous")
            rows.append(row); refs[row["id"]] = ref
    return rows, refs


def fresh_holdout(task):
    """Fresh GDM49 holdout with known-missing, unsupported, ambiguous and multi-clause cases."""
    domains = [("Shipment", "shipment"), ("Policy", "policy")]
    accepted_combos = [
        *((i,) for i in range(8)),
        (0, 1), (1, 2), (4, 5), (3, 4), (0, 6), (3, 7),
        (0, 3, 4), (1, 2, 6), (4, 5, 7), (0, 1, 6),
    ]
    composite_missing = [(0, 1), (1, 2), (4, 5), (3, 7)]
    rows, refs = [], {}
    for entity, root in domains:
        base = catalog(entity, root)
        for combo in accepted_combos:
            phrase = "; ".join(HOLDOUT_LANGUAGE[i] for i in combo)
            request = task_prefix(task) + phrase + f" for the {root}."
            uid = sha(["gdm49-holdout", task, entity, combo, phrase, "accepted"])[:24]
            options = copy.deepcopy(base["options"]); random.Random(uid).shuffle(options)
            rows.append({"id": uid, "task": task, "request": request, "catalog": _copy_catalog(base, options)})
            refs[uid] = {"status": "accepted",
                         "paths": [_option_for_suffix(base["options"], SEMANTIC_SUFFIXES[i])["id"] for i in combo],
                         "fixture_seeds": [13, 47, 101]}
        for i, suffix in enumerate(SEMANTIC_SUFFIXES):
            row, ref = _make_case(task, entity, root, HOLDOUT_LANGUAGE[i], "NO_MATCH", remove_suffix=suffix, tag="hold-missing")
            rows.append(row); refs[row["id"]] = ref
        for present_i, missing_i in composite_missing:
            options = copy.deepcopy(base["options"])
            remove_id = _option_for_suffix(options, SEMANTIC_SUFFIXES[missing_i])["id"]
            options = [o for o in options if o["id"] != remove_id]
            phrase = HOLDOUT_LANGUAGE[present_i] + "; " + HOLDOUT_LANGUAGE[missing_i]
            request = task_prefix(task) + phrase + f" for the {root}."
            uid = sha(["gdm49-holdout", task, entity, phrase, "composite-missing", missing_i])[:24]
            random.Random(uid).shuffle(options)
            rows.append({"id": uid, "task": task, "request": request, "catalog": _copy_catalog(base, options)})
            refs[uid] = {"status": "NO_MATCH", "paths": [], "fixture_seeds": [13, 47, 101]}
        for phrase in HOLDOUT_UNSUPPORTED:
            row, ref = _make_case(task, entity, root, phrase, "NO_MATCH", tag="hold-unsupported")
            rows.append(row); refs[row["id"]] = ref
        for phrase in HOLDOUT_AMBIGUOUS:
            row, ref = _make_case(task, entity, root, phrase, "AMBIGUOUS", tag="hold-ambiguous")
            rows.append(row); refs[row["id"]] = ref
        # Multi-clause ambiguity aggregation: one supported clause plus one ambiguous clause.
        for i, phrase in enumerate(HOLDOUT_AMBIGUOUS):
            composite = HOLDOUT_LANGUAGE[i] + "; " + phrase
            row, ref = _make_case(task, entity, root, composite, "AMBIGUOUS", tag="hold-composite-ambiguous")
            rows.append(row); refs[row["id"]] = ref
    return rows, refs


def _binary_clause_scores(bundle, encoder, item, clause, online=False):
    x, opts, raw = pair_features(encoder, item, clause, online=online)
    logits = bundle.matcher(x)
    probs = torch.sigmoid(logits)
    return opts, logits, probs, raw


def binary_infer(bundle, encoder, item, calibration, online=False):
    bundle.eval(); threshold = float(calibration["match_threshold"])
    selected = []; evidence = []; clause_statuses = []
    with torch.no_grad():
        for clause in clauses(item):
            opts, logits, probs, raw = _binary_clause_scores(bundle, encoder, item, clause, online=online)
            matched = [o["id"] for o, p in zip(opts, probs) if float(p) >= threshold]
            if len(matched) == 0:
                status = "NO_MATCH"
            elif len(matched) > 1:
                status = "AMBIGUOUS"
            else:
                status = "accepted"; selected.append(matched[0])
            clause_statuses.append(status)
            order = torch.argsort(probs, descending=True, stable=True)
            evidence.append({
                "clause": clause, "status": status, "matched": matched,
                "top": [{"id": opts[int(i)]["id"], "prob": float(probs[int(i)]),
                          "logit": float(logits[int(i)]), "raw": float(raw[int(i)])}
                         for i in order[:min(3, len(order))]],
            })
    if any(s == "NO_MATCH" for s in clause_statuses):
        request_status = "NO_MATCH"; selected = []
    elif any(s == "AMBIGUOUS" for s in clause_statuses):
        request_status = "AMBIGUOUS"; selected = []
    else:
        request_status = "accepted"; selected = sorted(set(selected))
    return {"status": request_status, "selected": selected,
            "pair_evidence": evidence, "calibration": calibration}


def _exact(prediction, reference):
    return prediction["status"] == reference["status"] and (
        reference["status"] != "accepted" or set(prediction.get("selected", [])) == set(reference["paths"])
    )


def _calibration_metrics(records, threshold):
    accepted_total = accepted_status = risk_total = risk_correct = exact_correct = incorrect = 0
    for row in records:
        prediction = binary_infer(row["bundle"], row["encoder"], row["public"],
                                  {"mode": "pair-threshold", "match_threshold": threshold,
                                   "selection_split": "validation"})
        ref = row["reference"]; ok = _exact(prediction, ref)
        exact_correct += int(ok)
        if ref["status"] == "accepted":
            accepted_total += 1; accepted_status += int(prediction["status"] == "accepted")
        else:
            risk_total += 1; risk_correct += int(prediction["status"] == ref["status"])
        incorrect += int(prediction["status"] == "accepted" and not ok)
    ar = accepted_status / accepted_total if accepted_total else 0.0
    rr = risk_correct / risk_total if risk_total else 0.0
    hm = 2 * ar * rr / (ar + rr) if ar + rr else 0.0
    return {"accepted_status_recall": ar, "risk_accuracy": rr, "selective_hmean": hm,
            "exact_accuracy": exact_correct / len(records) if records else 0.0,
            "incorrect_publication_rate": incorrect / len(records) if records else 0.0,
            "examples": len(records)}


def calibrate_binary(bundle, encoder, public_rows, public_refs, calibration_scope):
    rows = list(public_rows); refs = dict(public_refs)
    extra_hash = None
    if calibration_scope == "expanded-validation-only":
        extra_rows, extra_refs = calibration_set(public_rows[0]["task"] if public_rows else "operation")
        rows.extend(extra_rows); refs.update(extra_refs)
        extra_hash = sha({"rows": extra_rows, "references": extra_refs})
    records = [{"bundle": bundle, "encoder": encoder, "public": row, "reference": refs[row["id"]]} for row in rows]
    thresholds = [x / 100 for x in range(5, 96, 5)]
    best = None
    for threshold in thresholds:
        metrics = _calibration_metrics(records, threshold)
        key = (metrics["selective_hmean"], metrics["exact_accuracy"],
               -metrics["incorrect_publication_rate"], metrics["accepted_status_recall"])
        if best is None or key > best[0]:
            best = (key, threshold, metrics)
    calibration = {"mode": "pair-threshold", "match_threshold": best[1],
                   "selection_split": "validation", "calibration_scope": calibration_scope,
                   "expanded_calibration_sha256": extra_hash}
    return calibration, best[2]


def evaluate(rows, refs, infer_fn, compositor, path):
    records = []
    for item in rows:
        prediction = infer_fn(item)
        reference = refs[item["id"]]
        judgment = judge(item, reference, prediction)
        if item["task"] == "schema" and prediction["status"] == "accepted":
            composition = compositor.compose(item, prediction["selected"])
            judgment["composition"] = composition
            judgment["request_correct"] = judgment["request_correct"] and composition["success"] is True
            judgment["incorrect_publication"] = not judgment["request_correct"]
        records.append({"id": item["id"], "public": item, "reference": reference,
                        "prediction": prediction, "judgment": judgment})
    with open(path, "w") as handle:
        for record in records:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
    return records


def clause_metrics(records):
    tp = gold = predicted = 0
    for record in records:
        if record["reference"]["status"] != "accepted":
            continue
        g = set(record["reference"]["paths"])
        p = set(record["prediction"].get("selected", [])) if record["prediction"]["status"] == "accepted" else set()
        tp += len(g & p); gold += len(g); predicted += len(p)
    return {"target_recall": tp / gold if gold else None,
            "target_precision": tp / predicted if predicted else None,
            "targets": gold, "predicted_targets": predicted, "true_positive_targets": tp}


def _binary_train_and_calibrate(bundle, encoder, arm, task, public, refs, seed, epochs, directory):
    receipt = train_binary(bundle, encoder, arm, task, public, refs, seed, epochs, directory)
    val_rows = [r for r in public["validation"] if r["task"] == task]
    val_refs = {r["id"]: refs["validation"][r["id"]] for r in val_rows}
    calibration, metrics = calibrate_binary(bundle, encoder, val_rows, val_refs, arm["calibration"])
    write_json(directory / "calibration.json", {"selected": calibration, "validation_metrics": metrics})
    receipt.update({"selected_calibration": calibration,
                    "selected_calibration_validation_metrics": metrics,
                    "calibration_optimizer_updates": 0})
    training = json.loads((directory / "training.json").read_text())
    training.update(receipt); write_json(directory / "training.json", training)
    return receipt, calibration


def run_one(arm_name, task, seed, encoder, public, refs, data_report, compositor, root, epochs):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    arm = ARMS[arm_name]
    directory = root / f"{task}-{arm_name}-{encoder.name}-seed{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    config = {
        "id": directory.name, "arm": arm_name, "task": task, "backbone": encoder.name, **arm,
        "model_scope": "frozen pretrained encoder + learned feature-space adapter/head",
        "schema_scope": "catalog projection + deterministic SDL/Federation realization" if task == "schema" else None,
        "research_question": "listwise forced-choice scoring versus independent pairwise open-set clause matching",
    }
    write_json(directory / "config.json", config)
    rss_before = current_rss_kib(); started = time.perf_counter_ns()

    if arm["family"] == "listwise-control":
        bundle = g47.Bundle(encoder.dim * 5 + 4, encoder.dim)
        control_arm = {"gate": "clause-hybrid", "hardneg": True}
        receipt, calibration = g48.train(bundle, encoder, control_arm, task, public, refs, seed, epochs, directory)
        receipt["gdm49_family"] = "listwise-control"
        infer_fn = lambda item, online=False: g48.infer(bundle, encoder, control_arm, item, calibration, online=online)
    else:
        bundle = PairBundle(encoder.dim * 5 + 4)
        receipt, calibration = _binary_train_and_calibrate(bundle, encoder, arm, task, public, refs, seed, epochs, directory)
        receipt["gdm49_family"] = "binary-matcher"
        infer_fn = lambda item, online=False: binary_infer(bundle, encoder, item, calibration, online=online)

    training_ms = (time.perf_counter_ns() - started) / 1e6
    # Persist the family marker after GDM48 control training wrote its own receipt.
    training_file = json.loads((directory / "training.json").read_text())
    training_file.update({"gdm49_family": receipt["gdm49_family"]})
    write_json(directory / "training.json", training_file)

    base_rows = [r for r in public["test"] if r["task"] == task]
    base_refs = {r["id"]: refs["test"][r["id"]] for r in base_rows}
    g47_rows, g47_refs = g47.fresh_holdout(task)
    g48_rows, g48_refs = g48.fresh_holdout(task)
    regression_rows = base_rows + g47_rows + g48_rows
    regression_refs = {**base_refs, **g47_refs, **g48_refs}
    holdout_rows, holdout_refs = fresh_holdout(task)

    regression = evaluate(regression_rows, regression_refs, lambda x: infer_fn(x, False), compositor,
                          directory / "predictions-regression.jsonl")
    holdout = evaluate(holdout_rows, holdout_refs, lambda x: infer_fn(x, False), compositor,
                       directory / "predictions-holdout.jsonl")

    bench_rows = holdout_rows[:12]
    for row in bench_rows[:2]:
        validate_emission(row, infer_fn(row, True))
    samples = []; calls_before = encoder.calls
    for _ in range(2):
        for row in bench_rows:
            begin = time.perf_counter_ns(); pred = infer_fn(row, True)
            validate_emission(row, pred); samples.append((time.perf_counter_ns() - begin) / 1e6)
    calls_after = encoder.calls
    write_json(directory / "timings.json", {
        "generation_ms": samples, "warmup_requests": min(2, len(bench_rows)),
        "encoder_calls_measured": calls_after - calls_before,
        "scope": "single-request clause query encoding + open-set matching/gating + generation/validation; cached catalog; excludes download/index build/Rover/backend",
    })
    gc.collect(); rss_after = current_rss_kib()
    cal_rows, cal_refs = calibration_set(task)
    files = ["config.json", "training.json", "calibration.json", "initial.pt", "selected.pt",
             "predictions-regression.jsonl", "predictions-holdout.jsonl", "timings.json"]
    summary = {
        "format": FORMAT, "evidence_kind": "trained-feature-model", "config": config,
        "source_commit": os.environ.get("GITHUB_SHA", "unrecorded"), "run_id": os.environ.get("GITHUB_RUN_ID"),
        "seed": seed, "encoder": encoder.meta, "training": receipt, "training_and_selection_ms": training_ms,
        "dataset_sha256": data_report["dataset_sha256"], "dataset_scope": data_report["scope"],
        "calibration_only_sha256": sha({"rows": cal_rows, "references": cal_refs}),
        "calibration_scope": "public validation only" if arm["calibration"] == "public-validation" else "public validation plus disjoint synthetic calibration-only Asset/Booking cases",
        "secondary_holdout_sha256": sha({"rows": holdout_rows, "references": holdout_refs}),
        "secondary_holdout_scope": "new GDM49 Shipment/Policy synthetic holdout; never used for optimizer/checkpoint/calibration selection",
        "regression_scope": "corrected public test plus inspected GDM47 and GDM48 holdouts",
        "regression_examples": len(regression), "secondary_holdout_examples": len(holdout),
        "regression_metrics": aggregate(regression), "secondary_holdout_metrics": aggregate(holdout),
        "regression_clause_metrics": clause_metrics(regression), "secondary_holdout_clause_metrics": clause_metrics(holdout),
        "generation_latency": timing(samples),
        "latency_scope": "online query encoding + clause matching/status aggregation + generation/validation; cached catalog vectors",
        "memory": {"rss_before_training_kib": rss_before, "rss_after_evaluation_kib": rss_after,
                   "peak_worker_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "scope": "current process RSS plus worker high-water; worker reuses frozen backbone across arms"},
        "file_hashes": {name: file_hash(directory / name) for name in files},
    }
    write_json(directory / "summary.json", summary)
    print(json.dumps({"event": "completed_arm", "config": config["id"],
                      "holdout": summary["secondary_holdout_metrics"][task],
                      "clauses": summary["secondary_holdout_clause_metrics"],
                      "p50_ms": summary["generation_latency"]["p50_ms"]}), flush=True)
    return directory.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="distilbert", choices=["distilbert", "hash"])
    parser.add_argument("--task", required=True, choices=["operation", "schema"])
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--seeds", default=SEEDS_DEFAULT)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--output", default="artifacts/results")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise RuntimeError("Project execution belongs in GitHub Actions")
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    public, refs, data_report = corrected_dataset(root / "dataset")
    arms = [a for a in args.arms.split(",") if a]
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise ValueError(f"Unknown arms: {sorted(unknown)}")
    seeds = [int(s) for s in args.seeds.split(",") if s]
    if args.smoke:
        arms = ["binary-bce-publiccal"]; seeds = [4901]; args.epochs = 2
    write_json(root / "plan.json", {"backbone": args.backbone, "task": args.task,
                                     "arms": arms, "seeds": seeds, "epochs": args.epochs})
    encoder = Encoder(args.backbone); compositor = Compositor(root / "composition")
    schema_control = next(r for r in public["train"] if r["task"] == "schema" and refs["train"][r["id"]]["status"] == "accepted")
    all_paths = [o["id"] for o in schema_control["catalog"]["options"]]
    if not compositor.compose(schema_control, all_paths)["success"]:
        raise RuntimeError("Known-valid Federation composition control failed")
    completed, errors = [], []
    for arm in arms:
        for seed in seeds:
            try:
                completed.append(run_one(arm, args.task, seed, encoder, public, refs, data_report, compositor, root, args.epochs))
            except Exception:
                error = {"arm": arm, "task": args.task, "seed": seed, "traceback": traceback.format_exc()}
                errors.append(error); print(json.dumps({"event": "failed_arm", **error}), flush=True)
    write_json(root / "worker.json", {"complete": not errors, "completed": completed, "errors": errors,
                                       "expected": len(arms) * len(seeds), "encoder": encoder.meta})
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
