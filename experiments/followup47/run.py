"""GDM47 measured semantic-generalization and calibrated-risk follow-up.

All project execution is restricted to GitHub Actions. GDM47 keeps the verified
GDM46 DistilBERT + clause-decomposition setup fixed and tests five bounded changes:
validation-only status calibration, training-only semantic curriculum examples,
removing the raw query embedding from the risk gate, and the interaction with
hard-negative capability supervision.

The transformer remains frozen. Learned adapters/heads operate only on frozen
feature representations; this is not transformer fine-tuning. Schema generation
remains catalog projection plus deterministic SDL/Federation realization.
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

from experiments.measured.contracts import (
    api_sdl, audit, catalog, judge, make_operation, option_text, sha, subgraph_sdls,
)
from experiments.measured.evidence import Compositor, aggregate, file_hash, timing, write_json
from experiments.measured.models import Encoder, Head, state_hash
from experiments.followup45.run import clauses, pair_features
from experiments.followup46.run import (
    corrected_dataset, fresh_holdout as gdm46_holdout, task_prefix,
    balanced_status_rows, current_rss_kib, validate_emission,
)

FORMAT = "gdm47-measured-v1"
STATUS_TO_INDEX = {"accepted": 0, "NO_MATCH": 1, "AMBIGUOUS": 2}
INDEX_TO_STATUS = {v: k for k, v in STATUS_TO_INDEX.items()}

ARMS = {
    # Direct GDM46-style control on the new holdout.
    "control": {
        "curriculum": False, "hardneg": True,
        "status_features": "query+scores", "calibration": "argmax",
    },
    # Isolate validation-only class-threshold calibration.
    "calibrated": {
        "curriculum": False, "hardneg": True,
        "status_features": "query+scores", "calibration": "thresholds",
    },
    # Add training-only semantic contrast paraphrases.
    "curriculum-calibrated": {
        "curriculum": True, "hardneg": True,
        "status_features": "query+scores", "calibration": "thresholds",
    },
    # Remove raw request embedding from the risk gate; use only detached score statistics.
    "curriculum-scoreonly": {
        "curriculum": True, "hardneg": True,
        "status_features": "scores-only", "calibration": "thresholds",
    },
    # Isolate whether hard-negative capability loss is now harmful once the curriculum exists.
    "curriculum-nohardneg": {
        "curriculum": True, "hardneg": False,
        "status_features": "scores-only", "calibration": "thresholds",
    },
}

SEMANTIC_SUFFIXES = [
    ("title",),
    ("createdAt",),
    ("updatedAt",),
    ("reviews", "rating"),
    ("reviews", "author", "name"),
    ("reviews", "moderator", "name"),
    ("supplier", "name"),
    ("reviews", "author", "id"),
]

# Training-only curriculum. These strings are intentionally distinct from GDM46 and
# GDM47 secondary holdout wording. They are labels for synthetic training generation,
# not test-derived repair rules.
CURRICULUM = {
    ("title",): [
        "public display title", "customer-facing label", "headline assigned to the item",
    ],
    ("createdAt",): [
        "original registration timestamp", "time of initial persistence", "when the record was first created",
    ],
    ("updatedAt",): [
        "latest modification timestamp", "time of most recent change", "when the record was last revised",
    ],
    ("reviews", "rating"): [
        "numeric review rating", "score attached to each review", "number representing every evaluation",
    ],
    ("reviews", "author", "name"): [
        "display name of each review writer", "name of the person who authored each review", "reviewer author name",
    ],
    ("reviews", "moderator", "name"): [
        "display name of each review moderator", "name of the person supervising each review", "moderation contact name",
    ],
    ("supplier", "name"): [
        "supplier business name", "company providing the item", "vendor organization name",
    ],
    ("reviews", "author", "id"): [
        "identifier of each review writer", "review author ID", "stable key for the person who wrote each review",
    ],
}

# A fresh, uninspected synthetic holdout for GDM47. Once this batch is inspected it
# becomes regression evidence only.
FRESH_LANGUAGE = [
    "primary label presented at the top of the item",
    "moment the item first became persisted",
    "moment the item's newest stored change was made",
    "integer score attached to every review",
    "public names of the reviewers who wrote the feedback",
    "public names of the people responsible for review moderation",
    "business name of the organization supplying the item",
    "stable identifiers of the reviewers who wrote the feedback",
]


def _suffix(option):
    return tuple(option["path"][1:])


def _option_for_suffix(options, suffix):
    matches = [o for o in options if _suffix(o) == tuple(suffix)]
    if len(matches) != 1:
        raise AssertionError(f"Expected one option for suffix {suffix}, got {len(matches)}")
    return matches[0]


class Bundle(nn.Module):
    def __init__(self, pair_dim: int, encoder_dim: int):
        super().__init__()
        self.capability = Head(pair_dim, adapter=True)
        # Always allocate encoder_dim + 10 so score-only and query+score arms have
        # identical parameterization. Score-only arms concatenate a zero query vector.
        self.status = nn.Sequential(nn.Linear(encoder_dim + 10, 96), nn.GELU(), nn.Linear(96, 3))


def semantic_curriculum(task: str, train_rows):
    """Generate accepted, single-clause training examples from training catalogs only."""
    catalogs = {}
    for row in train_rows:
        if row["task"] != task:
            continue
        key = (row["catalog"]["entity"], row["catalog"]["root"])
        catalogs.setdefault(key, row["catalog"])
    examples = []
    for (entity, root), source in sorted(catalogs.items()):
        options = source["options"]
        for suffix, phrases in CURRICULUM.items():
            target = _option_for_suffix(options, suffix)["id"]
            for phrase in phrases:
                request = task_prefix(task) + phrase + f" for the {root}."
                uid = sha(["gdm47-curriculum", task, entity, root, suffix, phrase])[:24]
                cat = copy.deepcopy(source)
                random.Random(uid).shuffle(cat["options"])
                row = {"id": uid, "task": task, "request": request, "catalog": cat}
                ref = {"status": "accepted", "paths": [target], "fixture_seeds": [13, 47, 101]}
                examples.append((row, ref))
    return examples


def fresh_holdout(task: str):
    domains = [("Notebook", "notebook"), ("Invoice", "invoice")]
    combos = [
        *( (i,) for i in range(8) ),
        (0, 1), (1, 2), (4, 5), (3, 4), (0, 6), (3, 7),
        (0, 3, 4), (1, 2, 6), (4, 5, 7), (0, 1, 6),
    ]
    rows, refs = [], {}
    for entity, root in domains:
        base = catalog(entity, root)
        base_options = base["options"]
        for combo in combos:
            options = copy.deepcopy(base_options)
            targets = [_option_for_suffix(options, SEMANTIC_SUFFIXES[i])["id"] for i in combo]
            request = task_prefix(task) + "; ".join(FRESH_LANGUAGE[i] for i in combo) + f" for the {root}."
            uid = sha(["gdm47-holdout", entity, task, request, "accepted"])[:24]
            random.Random(uid).shuffle(options)
            cat = {k: copy.deepcopy(v) for k, v in base.items() if k != "options"}
            cat["options"] = options
            rows.append({"id": uid, "task": task, "request": request, "catalog": cat})
            refs[uid] = {"status": "accepted", "paths": targets, "fixture_seeds": [13, 47, 101]}

        # Test every capability as missing rather than only a hand-picked subset.
        for i, suffix in enumerate(SEMANTIC_SUFFIXES):
            options = copy.deepcopy(base_options)
            target = _option_for_suffix(options, suffix)["id"]
            options = [o for o in options if o["id"] != target]
            request = task_prefix(task) + FRESH_LANGUAGE[i] + f" for the {root}."
            uid = sha(["gdm47-holdout", entity, task, request, "missing", suffix])[:24]
            random.Random(uid).shuffle(options)
            cat = {k: copy.deepcopy(v) for k, v in base.items() if k != "options"}
            cat["options"] = options
            rows.append({"id": uid, "task": task, "request": request, "catalog": cat})
            refs[uid] = {"status": "NO_MATCH", "paths": [], "fixture_seeds": [13, 47, 101]}

        ambiguous = (
            "public name of the review-related person without identifying whether the writer or moderator role applies"
        )
        request = task_prefix(task) + ambiguous + f" for the {root}."
        uid = sha(["gdm47-holdout", entity, task, request, "ambiguous"])[:24]
        rows.append({"id": uid, "task": task, "request": request, "catalog": copy.deepcopy(base)})
        refs[uid] = {"status": "AMBIGUOUS", "paths": [], "fixture_seeds": [13, 47, 101]}
    return rows, refs


def _two(values):
    values = torch.sort(values, descending=True).values
    first = float(values[0]) if len(values) else 0.0
    second = float(values[1]) if len(values) > 1 else first
    return first, second, first - second


def risk_features(bundle, encoder, arm, item, online=False):
    q = encoder.query(item["request"], cached=not online)
    opts = item["catalog"]["options"]
    c = encoder.encode([option_text(o, True) for o in opts])
    raw_first, raw_second, raw_margin = _two(c @ q)
    clause_cache, maxima, margins, entropies = [], [], [], []
    with torch.no_grad():
        for clause in clauses(item):
            x, clause_opts, _ = pair_features(encoder, item, clause, online=online)
            logits = bundle.capability(x)
            first, second, margin = _two(logits)
            probs = torch.softmax(logits, dim=-1)
            entropy = float((-(probs * torch.log(probs.clamp_min(1e-12))).sum()) /
                            math.log(max(2, len(probs))))
            maxima.append(first); margins.append(margin); entropies.append(entropy)
            clause_cache.append((clause, clause_opts, logits.detach()))
    learned = [
        min(maxima) if maxima else 0.0,
        float(np.mean(maxima)) if maxima else 0.0,
        min(margins) if margins else 0.0,
        float(np.mean(margins)) if margins else 0.0,
        max(entropies) if entropies else 0.0,
    ]
    stats = torch.tensor([
        raw_first, raw_second, raw_margin, len(opts) / 10.0,
        *learned, len(clauses(item)) / 4.0,
    ], dtype=torch.float32)
    prefix = torch.zeros_like(q) if arm["status_features"] == "scores-only" else q
    return torch.cat([prefix, stats]), clause_cache


def select_capabilities(bundle, encoder, item, online=False, cache=None):
    selected, score_record = [], {}
    cached = {c: (opts, logits) for c, opts, logits in (cache or [])}
    for clause in clauses(item):
        if clause in cached:
            opts, logits = cached[clause]
        else:
            x, opts, _ = pair_features(encoder, item, clause, online=online)
            logits = bundle.capability(x)
        best = int(torch.argmax(logits))
        selected.append(opts[best]["id"])
        score_record[clause] = {o["id"]: float(s) for o, s in zip(opts, logits)}
    return sorted(set(selected)), score_record


def status_from_probs(probs, calibration):
    if calibration.get("mode") == "argmax":
        return INDEX_TO_STATUS[int(torch.tensor(probs).argmax())]
    no_t = float(calibration["no_match_threshold"])
    amb_t = float(calibration["ambiguous_threshold"])
    candidates = []
    if probs[1] >= no_t:
        candidates.append((probs[1] / max(no_t, 1e-9), "NO_MATCH"))
    if probs[2] >= amb_t:
        candidates.append((probs[2] / max(amb_t, 1e-9), "AMBIGUOUS"))
    return max(candidates)[1] if candidates else "accepted"


def infer(bundle, encoder, arm, item, calibration, online=False):
    bundle.eval()
    with torch.no_grad():
        features, cache = risk_features(bundle, encoder, arm, item, online=online)
        logits = bundle.status(features)
        probs = torch.softmax(logits, -1).tolist()
        status = status_from_probs(probs, calibration)
        result = {"status": status, "selected": [], "status_scores": probs,
                  "calibration": calibration}
        if status != "accepted":
            return result
        selected, scores = select_capabilities(bundle, encoder, item, online=online, cache=cache)
        result.update({"selected": selected, "scores": scores})
        return result


def exact(prediction, reference):
    return prediction["status"] == reference["status"] and (
        reference["status"] != "accepted" or set(prediction["selected"]) == set(reference["paths"])
    )


def capability_loss(bundle, encoder, item, ref, hardneg):
    parts = clauses(item)
    if len(parts) != len(ref["paths"]):
        raise AssertionError("Public clause cardinality differs from accepted reference cardinality")
    loss = torch.tensor(0.0)
    for clause, target in zip(parts, ref["paths"]):
        x, opts, _ = pair_features(encoder, item, clause)
        logits = bundle.capability(x)
        target_index = next(i for i, o in enumerate(opts) if o["id"] == target)
        loss = loss + F.cross_entropy(logits[None, :], torch.tensor([target_index]))
        if hardneg and len(opts) > 1:
            negatives = torch.cat([logits[:target_index], logits[target_index + 1:]])
            hardest = negatives.max()
            loss = loss + 0.45 * F.softplus(hardest - logits[target_index] + 0.6)
    return loss


def _subhash(module):
    return state_hash(copy.deepcopy(module.state_dict()))


def _validation_records(bundle, encoder, arm, rows, refs):
    out = []
    bundle.eval()
    with torch.no_grad():
        for row in rows:
            features, cache = risk_features(bundle, encoder, arm, row)
            probs = torch.softmax(bundle.status(features), -1).tolist()
            selected, _ = select_capabilities(bundle, encoder, row, cache=cache)
            out.append({"reference": refs[row["id"]], "probs": probs, "selected": selected})
    return out


def _calibration_metrics(records, calibration):
    accepted_total = accepted_status_correct = risks = risk_correct = exact_correct = incorrect_publications = 0
    for record in records:
        ref = record["reference"]
        status = status_from_probs(record["probs"], calibration)
        pred = {"status": status, "selected": record["selected"] if status == "accepted" else []}
        is_exact = exact(pred, ref)
        exact_correct += int(is_exact)
        if ref["status"] == "accepted":
            accepted_total += 1; accepted_status_correct += int(status == "accepted")
        else:
            risks += 1; risk_correct += int(status == ref["status"])
        incorrect_publications += int(status == "accepted" and not is_exact)
    accepted_recall = accepted_status_correct / accepted_total if accepted_total else 0.0
    risk_accuracy = risk_correct / risks if risks else 0.0
    hmean = (2 * accepted_recall * risk_accuracy / (accepted_recall + risk_accuracy)
             if accepted_recall + risk_accuracy else 0.0)
    return {
        "accepted_status_recall": accepted_recall,
        "risk_accuracy": risk_accuracy,
        "selective_hmean": hmean,
        "exact_accuracy": exact_correct / len(records) if records else 0.0,
        "incorrect_publication_rate": incorrect_publications / len(records) if records else 0.0,
        "examples": len(records),
    }


def calibrate(records, mode):
    if mode == "argmax":
        calibration = {"mode": "argmax"}
        return calibration, _calibration_metrics(records, calibration)
    thresholds = [x / 100 for x in range(30, 96, 5)]
    best = None
    for no_t in thresholds:
        for amb_t in thresholds:
            calibration = {"mode": "thresholds", "no_match_threshold": no_t,
                           "ambiguous_threshold": amb_t, "selection_split": "validation"}
            metrics = _calibration_metrics(records, calibration)
            key = (metrics["selective_hmean"], metrics["exact_accuracy"],
                   -metrics["incorrect_publication_rate"], metrics["accepted_status_recall"])
            if best is None or key > best[0]:
                best = (key, calibration, metrics)
    return best[1], best[2]


def train(bundle, encoder, arm, task, public, refs, seed, epochs, directory):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    initial = copy.deepcopy(bundle.state_dict())
    initial_hash = state_hash(initial)
    initial_cap_hash, initial_status_hash = _subhash(bundle.capability), _subhash(bundle.status)
    torch.save({"state": initial, "arm": arm, "task": task}, directory / "initial.pt")

    train_rows = [r for r in public["train"] if r["task"] == task]
    val_rows = [r for r in public["validation"] if r["task"] == task]
    accepted_train = [(r, refs["train"][r["id"]]) for r in train_rows
                      if refs["train"][r["id"]]["status"] == "accepted"]
    accepted_val = [r for r in val_rows if refs["validation"][r["id"]]["status"] == "accepted"]
    curriculum = semantic_curriculum(task, train_rows) if arm["curriculum"] else []
    capability_examples = accepted_train + curriculum
    history = {"capability": [], "status": []}
    cap_updates = status_updates = 0; max_grad = 0.0

    optimizer = torch.optim.AdamW(bundle.capability.parameters(), lr=8e-4, weight_decay=1e-3)
    best_cap = None; best_cap_key = None; selected_cap_updates = 0; cap_selected_epoch = 0
    for epoch in range(1, epochs + 1):
        order = list(capability_examples); random.Random(seed + epoch).shuffle(order)
        losses = []; bundle.train()
        for row, ref in order:
            optimizer.zero_grad(set_to_none=True)
            loss = capability_loss(bundle, encoder, row, ref, arm["hardneg"])
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(bundle.capability.parameters(), 5.0))
            if not math.isfinite(norm): raise FloatingPointError("Nonfinite capability gradient")
            optimizer.step(); cap_updates += 1; max_grad = max(max_grad, norm); losses.append(float(loss.detach()))
        bundle.eval(); correct = 0
        with torch.no_grad():
            for row in accepted_val:
                pred, _ = select_capabilities(bundle, encoder, row)
                correct += int(set(pred) == set(refs["validation"][row["id"]]["paths"]))
        entry = {"epoch": epoch, "loss": float(np.mean(losses)),
                 "accepted_validation_correct": correct,
                 "accepted_validation_examples": len(accepted_val),
                 "optimizer_updates": cap_updates}
        history["capability"].append(entry)
        key = (correct, -entry["loss"])
        if best_cap_key is None or key > best_cap_key:
            best_cap_key = key; best_cap = copy.deepcopy(bundle.capability.state_dict())
            selected_cap_updates = cap_updates; cap_selected_epoch = epoch
        print(json.dumps({"event": "capability_epoch", "arm": arm, "task": task, "seed": seed, **entry}), flush=True)
    bundle.capability.load_state_dict(best_cap)
    for p in bundle.capability.parameters(): p.requires_grad = False

    optimizer = torch.optim.AdamW(bundle.status.parameters(), lr=8e-4, weight_decay=1e-3)
    best_status = None; best_key = None; best_calibration = None; best_cal_metrics = None
    selected_status_updates = 0; status_selected_epoch = 0
    for epoch in range(1, epochs + 1):
        order = balanced_status_rows(train_rows, refs["train"], seed + epoch)
        losses = []; bundle.train()
        for row in order:
            optimizer.zero_grad(set_to_none=True)
            features, _ = risk_features(bundle, encoder, arm, row)
            logits = bundle.status(features)
            y = torch.tensor(STATUS_TO_INDEX[refs["train"][row["id"]]["status"]])
            loss = F.cross_entropy(logits[None, :], y[None])
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(bundle.status.parameters(), 5.0))
            if not math.isfinite(norm): raise FloatingPointError("Nonfinite status gradient")
            optimizer.step(); status_updates += 1; max_grad = max(max_grad, norm); losses.append(float(loss.detach()))
        records = _validation_records(bundle, encoder, arm, val_rows, refs["validation"])
        calibration, metrics = calibrate(records, arm["calibration"])
        entry = {"epoch": epoch, "loss": float(np.mean(losses)), "optimizer_updates": status_updates,
                 "calibration": calibration, "validation_metrics": metrics}
        history["status"].append(entry)
        key = (metrics["selective_hmean"], metrics["exact_accuracy"],
               -metrics["incorrect_publication_rate"], metrics["accepted_status_recall"], -entry["loss"])
        if best_key is None or key > best_key:
            best_key = key; best_status = copy.deepcopy(bundle.status.state_dict())
            best_calibration = copy.deepcopy(calibration); best_cal_metrics = copy.deepcopy(metrics)
            selected_status_updates = status_updates; status_selected_epoch = epoch
        print(json.dumps({"event": "status_epoch", "arm": arm, "task": task, "seed": seed, **entry}), flush=True)
    bundle.status.load_state_dict(best_status)
    for p in bundle.capability.parameters(): p.requires_grad = True

    selected = copy.deepcopy(bundle.state_dict())
    selected_hash = state_hash(selected)
    selected_cap_hash, selected_status_hash = _subhash(bundle.capability), _subhash(bundle.status)
    if selected_hash == initial_hash or max_grad <= 0:
        raise RuntimeError("Training failed to change selected feature model")
    if selected_cap_hash == initial_cap_hash or selected_status_hash == initial_status_hash:
        raise RuntimeError("Both capability and status heads must change")
    torch.save({"state": selected, "arm": arm, "task": task,
                "encoder": encoder.meta, "seed": seed}, directory / "selected.pt")
    write_json(directory / "training.json", history)
    write_json(directory / "calibration.json", {
        "selected": best_calibration, "validation_metrics": best_cal_metrics,
        "selected_epoch": status_selected_epoch,
    })
    return {
        "optimizer_updates": cap_updates + status_updates,
        "capability_optimizer_updates": cap_updates,
        "status_optimizer_updates": status_updates,
        "selected_capability_optimizer_updates": selected_cap_updates,
        "selected_status_optimizer_updates": selected_status_updates,
        "selected_epoch": {"capability": cap_selected_epoch, "status": status_selected_epoch},
        "max_gradient_norm": max_grad,
        "initial_state_hash": initial_hash, "selected_state_hash": selected_hash,
        "initial_capability_hash": initial_cap_hash, "selected_capability_hash": selected_cap_hash,
        "initial_status_hash": initial_status_hash, "selected_status_hash": selected_status_hash,
        "train_examples": len(train_rows), "validation_examples": len(val_rows),
        "curriculum_examples": len(curriculum),
        "trainable_parameters": sum(p.numel() for p in bundle.parameters()),
        "independent_backbone_finetuning": False,
        "adapter_location": "frozen-embedding feature space",
        "status_gate": arm["status_features"], "calibration_mode": arm["calibration"],
        "dynamic_hard_negative": arm["hardneg"],
        "selected_calibration": best_calibration,
        "selected_calibration_validation_metrics": best_cal_metrics,
    }, best_calibration


def evaluate(rows, refs, bundle, encoder, arm, calibration, compositor, path):
    records = []
    for item in rows:
        prediction = infer(bundle, encoder, arm, item, calibration)
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
        gold_set = set(record["reference"]["paths"])
        pred_set = set(record["prediction"].get("selected", [])) if record["prediction"]["status"] == "accepted" else set()
        tp += len(gold_set & pred_set); gold += len(gold_set); predicted += len(pred_set)
    return {
        "target_recall": tp / gold if gold else None,
        "target_precision": tp / predicted if predicted else None,
        "targets": gold, "predicted_targets": predicted, "true_positive_targets": tp,
    }


def run_one(arm_name, task, seed, encoder, public, refs, data_report, compositor, root, epochs):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    arm = ARMS[arm_name]
    directory = root / f"{task}-{arm_name}-{encoder.name}-seed{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    config = {
        "id": directory.name, "arm": arm_name, "task": task, "backbone": encoder.name, **arm,
        "model_scope": "frozen pretrained encoder + learned feature adapter/heads",
        "schema_scope": "catalog projection + deterministic SDL/Federation realization" if task == "schema" else None,
        "benchmark_correction": "task-consistent accepted/NO_MATCH/AMBIGUOUS syntax; GDM46 holdout moved to regression",
    }
    write_json(directory / "config.json", config)
    bundle = Bundle(encoder.dim * 5 + 4, encoder.dim)
    rss_before = current_rss_kib(); started = time.perf_counter_ns()
    receipt, calibration = train(bundle, encoder, arm, task, public, refs, seed, epochs, directory)
    training_ms = (time.perf_counter_ns() - started) / 1e6

    base_reg_rows = [r for r in public["test"] if r["task"] == task]
    base_reg_refs = {r["id"]: refs["test"][r["id"]] for r in base_reg_rows}
    prior_rows, prior_refs = gdm46_holdout(task)
    regression_rows = base_reg_rows + prior_rows
    regression_refs = {**base_reg_refs, **prior_refs}
    holdout_rows, holdout_refs = fresh_holdout(task)

    regression = evaluate(regression_rows, regression_refs, bundle, encoder, arm, calibration, compositor,
                          directory / "predictions-regression.jsonl")
    holdout = evaluate(holdout_rows, holdout_refs, bundle, encoder, arm, calibration, compositor,
                       directory / "predictions-holdout.jsonl")

    bench_rows = holdout_rows[:12]
    for row in bench_rows[:2]:
        validate_emission(row, infer(bundle, encoder, arm, row, calibration, online=True))
    samples = []; calls_before = encoder.calls
    for _ in range(2):
        for row in bench_rows:
            begin = time.perf_counter_ns()
            pred = infer(bundle, encoder, arm, row, calibration, online=True)
            validate_emission(row, pred)
            samples.append((time.perf_counter_ns() - begin) / 1e6)
    calls_after = encoder.calls
    write_json(directory / "timings.json", {
        "generation_ms": samples, "warmup_requests": min(2, len(bench_rows)),
        "encoder_calls_measured": calls_after - calls_before,
        "scope": "single-request calibrated status + clause query encoding, cached catalog, learned heads, GraphQL rendering/validation; excludes download, index build, Rover and fixture backend",
    })
    gc.collect(); rss_after = current_rss_kib()
    files = ["config.json", "training.json", "calibration.json", "initial.pt", "selected.pt",
             "predictions-regression.jsonl", "predictions-holdout.jsonl", "timings.json"]
    summary = {
        "format": FORMAT, "evidence_kind": "trained-feature-model", "config": config,
        "source_commit": os.environ.get("GITHUB_SHA", "unrecorded"), "run_id": os.environ.get("GITHUB_RUN_ID"),
        "seed": seed, "encoder": encoder.meta, "training": receipt,
        "training_and_selection_ms": training_ms,
        "dataset_sha256": data_report["dataset_sha256"], "dataset_scope": data_report["scope"],
        "curriculum_sha256": sha(CURRICULUM) if arm["curriculum"] else None,
        "benchmark_correction": data_report["benchmark_correction"],
        "secondary_holdout_sha256": sha({"rows": holdout_rows, "references": holdout_refs}),
        "secondary_holdout_scope": "new GDM47 synthetic holdout; not used for optimizer/checkpoint/calibration selection; not human-authored OOD",
        "regression_scope": "corrected public test plus previously inspected GDM46 secondary holdout",
        "regression_examples": len(regression), "secondary_holdout_examples": len(holdout),
        "regression_metrics": aggregate(regression), "secondary_holdout_metrics": aggregate(holdout),
        "regression_clause_metrics": clause_metrics(regression),
        "secondary_holdout_clause_metrics": clause_metrics(holdout),
        "generation_latency": timing(samples), "latency_scope": "online query encoding + calibrated generation/validation; cached catalog vectors",
        "memory": {"rss_before_training_kib": rss_before, "rss_after_evaluation_kib": rss_after,
                   "peak_worker_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "scope": "current process RSS plus worker high-water; worker reuses frozen backbone across arms"},
        "file_hashes": {name: file_hash(directory / name) for name in files},
    }
    write_json(directory / "summary.json", summary)
    metrics = summary["secondary_holdout_metrics"][task]
    print(json.dumps({"event": "completed_arm", "config": config["id"], "seed": seed,
                      "holdout": metrics, "clauses": summary["secondary_holdout_clause_metrics"],
                      "p50_ms": summary["generation_latency"]["p50_ms"]}), flush=True)
    return directory.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="distilbert", choices=["distilbert", "hash"])
    parser.add_argument("--task", required=True, choices=["operation", "schema"])
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--seeds", default="4701,4702")
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
    if unknown: raise ValueError(f"Unknown arms: {sorted(unknown)}")
    seeds = [int(s) for s in args.seeds.split(",") if s]
    if args.smoke:
        arms = ["curriculum-scoreonly"]; seeds = [4701]; args.epochs = 2
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
                completed.append(run_one(arm, args.task, seed, encoder, public, refs,
                                         data_report, compositor, root, args.epochs))
            except Exception:
                error = {"arm": arm, "task": args.task, "seed": seed,
                         "traceback": traceback.format_exc()}
                errors.append(error); print(json.dumps({"event": "failed_arm", **error}), flush=True)
    write_json(root / "worker.json", {"complete": not errors, "completed": completed,
                                       "errors": errors, "expected": len(arms) * len(seeds),
                                       "encoder": encoder.meta})
    if errors: raise SystemExit(1)


if __name__ == "__main__":
    main()
