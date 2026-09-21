"""GDM48 measured per-clause risk-gating follow-up.

GDM47 verified that capability selection can be precise when the request survives the
risk gate, but its request-global gate frequently rejects answerable requests. GDM48
keeps frozen DistilBERT + clause decomposition fixed and tests whether status should
be derived from validation-calibrated per-clause evidence instead of a request-global
3-way classifier.

All compilation, training, evaluation and benchmarking are restricted to GitHub
Actions. The transformer remains frozen. Schema output remains catalog projection
plus deterministic SDL/Federation realization, not unconstrained schema invention.
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
from torch.nn import functional as F

from experiments.measured.contracts import api_sdl, catalog, judge, make_operation, sha, subgraph_sdls
from experiments.measured.evidence import Compositor, aggregate, file_hash, timing, write_json
from experiments.measured.models import Encoder, state_hash
from experiments.followup45.run import clauses, pair_features
from experiments.followup46.run import balanced_status_rows, corrected_dataset, current_rss_kib, task_prefix, validate_emission
from experiments.followup47 import run as g47

FORMAT = "gdm48-measured-v1"
ARMS = {
    "global-scoreonly": {"gate": "global-scoreonly", "hardneg": True},
    "clause-prob": {"gate": "clause-prob", "hardneg": True},
    "clause-logit": {"gate": "clause-logit", "hardneg": True},
    "clause-hybrid": {"gate": "clause-hybrid", "hardneg": True},
    "clause-hybrid-nohardneg": {"gate": "clause-hybrid", "hardneg": False},
}

FRESH_LANGUAGE = [
    "headline customers see for the record",
    "timestamp marking the record's first durable save",
    "timestamp of the most recent persisted revision",
    "whole-number assessment recorded on each review",
    "visible names of the people who composed each review",
    "visible names of the people who oversee each review",
    "registered business name of the provider behind the record",
    "persistent keys identifying the people who composed each review",
]
SEMANTIC_SUFFIXES = g47.SEMANTIC_SUFFIXES


def _suffix(option):
    return tuple(option["path"][1:])


def _option_for_suffix(options, suffix):
    matches = [o for o in options if _suffix(o) == tuple(suffix)]
    if len(matches) != 1:
        raise AssertionError((suffix, len(matches)))
    return matches[0]


def fresh_holdout(task: str):
    """Fresh GDM48 holdout, including multi-clause risk aggregation cases."""
    domains = [("Device", "device"), ("Reservation", "reservation")]
    accepted_combos = [
        *((i,) for i in range(8)),
        (0, 1), (1, 2), (4, 5), (3, 4), (0, 6), (3, 7),
        (0, 3, 4), (1, 2, 6), (4, 5, 7), (0, 1, 6),
    ]
    composite_missing = [(0, 1), (1, 2), (4, 5), (3, 7)]
    rows, refs = [], {}
    for entity, root in domains:
        base = catalog(entity, root)
        base_options = base["options"]
        for combo in accepted_combos:
            options = copy.deepcopy(base_options)
            targets = [_option_for_suffix(options, SEMANTIC_SUFFIXES[i])["id"] for i in combo]
            request = task_prefix(task) + "; ".join(FRESH_LANGUAGE[i] for i in combo) + f" for the {root}."
            uid = sha(["gdm48-holdout", entity, task, request, "accepted"])[:24]
            random.Random(uid).shuffle(options)
            cat = {k: copy.deepcopy(v) for k, v in base.items() if k != "options"}; cat["options"] = options
            rows.append({"id": uid, "task": task, "request": request, "catalog": cat})
            refs[uid] = {"status": "accepted", "paths": targets, "fixture_seeds": [13, 47, 101]}

        for i, suffix in enumerate(SEMANTIC_SUFFIXES):
            options = copy.deepcopy(base_options)
            target = _option_for_suffix(options, suffix)["id"]
            options = [o for o in options if o["id"] != target]
            request = task_prefix(task) + FRESH_LANGUAGE[i] + f" for the {root}."
            uid = sha(["gdm48-holdout", entity, task, request, "missing", suffix])[:24]
            random.Random(uid).shuffle(options)
            cat = {k: copy.deepcopy(v) for k, v in base.items() if k != "options"}; cat["options"] = options
            rows.append({"id": uid, "task": task, "request": request, "catalog": cat})
            refs[uid] = {"status": "NO_MATCH", "paths": [], "fixture_seeds": [13, 47, 101]}

        for present_i, missing_i in composite_missing:
            options = copy.deepcopy(base_options)
            target = _option_for_suffix(options, SEMANTIC_SUFFIXES[missing_i])["id"]
            options = [o for o in options if o["id"] != target]
            request = task_prefix(task) + FRESH_LANGUAGE[present_i] + "; " + FRESH_LANGUAGE[missing_i] + f" for the {root}."
            uid = sha(["gdm48-holdout", entity, task, request, "composite-missing", missing_i])[:24]
            random.Random(uid).shuffle(options)
            cat = {k: copy.deepcopy(v) for k, v in base.items() if k != "options"}; cat["options"] = options
            rows.append({"id": uid, "task": task, "request": request, "catalog": cat})
            refs[uid] = {"status": "NO_MATCH", "paths": [], "fixture_seeds": [13, 47, 101]}

        ambiguous = "visible name of the review participant without specifying whether they wrote or moderated the review"
        request = task_prefix(task) + ambiguous + f" for the {root}."
        uid = sha(["gdm48-holdout", entity, task, request, "ambiguous"])[:24]
        rows.append({"id": uid, "task": task, "request": request, "catalog": copy.deepcopy(base)})
        refs[uid] = {"status": "AMBIGUOUS", "paths": [], "fixture_seeds": [13, 47, 101]}

        request = task_prefix(task) + FRESH_LANGUAGE[0] + "; " + ambiguous + f" for the {root}."
        uid = sha(["gdm48-holdout", entity, task, request, "composite-ambiguous"])[:24]
        rows.append({"id": uid, "task": task, "request": request, "catalog": copy.deepcopy(base)})
        refs[uid] = {"status": "AMBIGUOUS", "paths": [], "fixture_seeds": [13, 47, 101]}
    return rows, refs


def _top2(values):
    order = torch.argsort(values, descending=True, stable=True)
    first_i = int(order[0])
    second_i = int(order[1]) if len(order) > 1 else first_i
    return first_i, second_i, float(values[first_i]), float(values[second_i])


def clause_evidence(bundle, encoder, item, gate, online=False):
    selected, records = [], []
    for clause in clauses(item):
        x, opts, raw = pair_features(encoder, item, clause, online=online)
        logits = bundle.capability(x)
        probs = torch.softmax(logits, dim=-1)
        first_i, second_i, first_logit, second_logit = _top2(logits)
        _, _, first_raw, second_raw = _top2(raw)
        first_prob = float(probs[first_i]); second_prob = float(probs[second_i])
        selected.append(opts[first_i]["id"])
        records.append({
            "clause": clause,
            "best_id": opts[first_i]["id"],
            "second_id": opts[second_i]["id"],
            "prob_top": first_prob,
            "prob_margin": first_prob - second_prob,
            "logit_top": first_logit,
            "logit_margin": first_logit - second_logit,
            "raw_top": first_raw,
            "raw_margin": first_raw - second_raw,
            "scores": {o["id"]: float(s) for o, s in zip(opts, logits)},
        })
    if gate == "clause-prob":
        support = min((r["prob_top"] for r in records), default=0.0)
        ambiguity = min((r["prob_margin"] for r in records), default=0.0)
    elif gate == "clause-logit":
        support = min((r["logit_top"] for r in records), default=-1e9)
        ambiguity = min((r["logit_margin"] for r in records), default=0.0)
    elif gate == "clause-hybrid":
        support = min((0.65 * r["prob_top"] + 0.35 * ((r["raw_top"] + 1.0) / 2.0) for r in records), default=0.0)
        ambiguity = min((0.70 * r["prob_margin"] + 0.30 * min(1.0, max(0.0, r["raw_margin"] * 2.0)) for r in records), default=0.0)
    else:
        raise ValueError(gate)
    return {"support": support, "ambiguity": ambiguity, "selected": sorted(set(selected)), "clauses": records}


def clause_status(evidence, calibration):
    if evidence["support"] < float(calibration["support_threshold"]):
        return "NO_MATCH"
    if evidence["ambiguity"] < float(calibration["ambiguity_threshold"]):
        return "AMBIGUOUS"
    return "accepted"


def _threshold_candidates(values):
    values = sorted(set(float(v) for v in values if math.isfinite(float(v))))
    if not values:
        return [0.0]
    if len(values) > 12:
        idx = sorted(set(round(i * (len(values) - 1) / 11) for i in range(12)))
        values = [values[i] for i in idx]
    eps = max(1e-6, (max(values) - min(values)) * 1e-5)
    out = [values[0] - eps, values[-1] + eps]
    out.extend(values)
    out.extend((a + b) / 2 for a, b in zip(values, values[1:]))
    return sorted(set(out))


def _gate_metrics(records, calibration):
    accepted_total = accepted_status_correct = risk_total = risk_correct = exact_correct = incorrect = 0
    for r in records:
        status = clause_status(r["evidence"], calibration)
        ref = r["reference"]
        pred = {"status": status, "selected": r["evidence"]["selected"] if status == "accepted" else []}
        ok = g47.exact(pred, ref)
        exact_correct += int(ok)
        if ref["status"] == "accepted":
            accepted_total += 1; accepted_status_correct += int(status == "accepted")
        else:
            risk_total += 1; risk_correct += int(status == ref["status"])
        incorrect += int(status == "accepted" and not ok)
    ar = accepted_status_correct / accepted_total if accepted_total else 0.0
    rr = risk_correct / risk_total if risk_total else 0.0
    hm = 2 * ar * rr / (ar + rr) if ar + rr else 0.0
    return {"accepted_status_recall": ar, "risk_accuracy": rr, "selective_hmean": hm,
            "exact_accuracy": exact_correct / len(records) if records else 0.0,
            "incorrect_publication_rate": incorrect / len(records) if records else 0.0,
            "examples": len(records)}


def calibrate_clause_gate(bundle, encoder, gate, rows, refs):
    records = []
    for row in rows:
        records.append({"reference": refs[row["id"]], "evidence": clause_evidence(bundle, encoder, row, gate)})
    support = _threshold_candidates([r["evidence"]["support"] for r in records])
    ambiguity = _threshold_candidates([r["evidence"]["ambiguity"] for r in records])
    best = None
    for st in support:
        for at in ambiguity:
            cal = {"mode": "clause-thresholds", "gate": gate, "support_threshold": st,
                   "ambiguity_threshold": at, "selection_split": "validation"}
            metrics = _gate_metrics(records, cal)
            key = (metrics["selective_hmean"], metrics["exact_accuracy"],
                   -metrics["incorrect_publication_rate"], metrics["accepted_status_recall"])
            if best is None or key > best[0]:
                best = (key, copy.deepcopy(cal), copy.deepcopy(metrics))
    return best[1], best[2]


def infer(bundle, encoder, arm, item, calibration, online=False):
    if arm["gate"] == "global-scoreonly":
        return g47.infer(bundle, encoder, {"status_features": "scores-only"}, item, calibration, online=online)
    evidence = clause_evidence(bundle, encoder, item, arm["gate"], online=online)
    status = clause_status(evidence, calibration)
    return {"status": status, "selected": evidence["selected"] if status == "accepted" else [],
            "gate_evidence": evidence, "calibration": calibration}


def _subhash(module):
    return state_hash(copy.deepcopy(module.state_dict()))


def train(bundle, encoder, arm, task, public, refs, seed, epochs, directory):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    initial = copy.deepcopy(bundle.state_dict())
    initial_hash = state_hash(initial)
    initial_cap_hash, initial_status_hash = _subhash(bundle.capability), _subhash(bundle.status)
    torch.save({"state": initial, "arm": arm, "task": task}, directory / "initial.pt")

    train_rows = [r for r in public["train"] if r["task"] == task]
    val_rows = [r for r in public["validation"] if r["task"] == task]
    accepted_train = [(r, refs["train"][r["id"]]) for r in train_rows if refs["train"][r["id"]]["status"] == "accepted"]
    accepted_val = [r for r in val_rows if refs["validation"][r["id"]]["status"] == "accepted"]
    curriculum = g47.semantic_curriculum(task, train_rows)
    capability_examples = accepted_train + curriculum
    history = {"capability": [], "risk": []}
    cap_updates = risk_updates = 0; max_grad = 0.0

    optimizer = torch.optim.AdamW(bundle.capability.parameters(), lr=8e-4, weight_decay=1e-3)
    best_cap = None; best_key = None; selected_cap_updates = 0; selected_cap_epoch = 0
    for epoch in range(1, epochs + 1):
        order = list(capability_examples); random.Random(seed + epoch).shuffle(order)
        losses = []; bundle.train()
        for row, ref in order:
            optimizer.zero_grad(set_to_none=True)
            loss = g47.capability_loss(bundle, encoder, row, ref, arm["hardneg"])
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(bundle.capability.parameters(), 5.0))
            if not math.isfinite(norm): raise FloatingPointError("Nonfinite capability gradient")
            optimizer.step(); cap_updates += 1; max_grad = max(max_grad, norm); losses.append(float(loss.detach()))
        bundle.eval(); correct = 0
        with torch.no_grad():
            for row in accepted_val:
                pred, _ = g47.select_capabilities(bundle, encoder, row)
                correct += int(set(pred) == set(refs["validation"][row["id"]]["paths"]))
        entry = {"epoch": epoch, "loss": float(np.mean(losses)), "optimizer_updates": cap_updates,
                 "accepted_validation_correct": correct, "accepted_validation_examples": len(accepted_val)}
        history["capability"].append(entry)
        key = (correct, -entry["loss"])
        if best_key is None or key > best_key:
            best_key = key; best_cap = copy.deepcopy(bundle.capability.state_dict())
            selected_cap_updates = cap_updates; selected_cap_epoch = epoch
        print(json.dumps({"event": "capability_epoch", "arm": arm, "task": task, "seed": seed, **entry}), flush=True)
    bundle.capability.load_state_dict(best_cap)
    selected_status_updates = 0; selected_status_epoch = None

    if arm["gate"] == "global-scoreonly":
        for p in bundle.capability.parameters(): p.requires_grad = False
        optimizer = torch.optim.AdamW(bundle.status.parameters(), lr=8e-4, weight_decay=1e-3)
        best_status = None; best_status_key = None; best_cal = None; best_cal_metrics = None
        for epoch in range(1, epochs + 1):
            order = balanced_status_rows(train_rows, refs["train"], seed + epoch)
            losses = []; bundle.train()
            for row in order:
                optimizer.zero_grad(set_to_none=True)
                features, _ = g47.risk_features(bundle, encoder, {"status_features": "scores-only"}, row)
                logits = bundle.status(features)
                y = torch.tensor(g47.STATUS_TO_INDEX[refs["train"][row["id"]]["status"]])
                loss = F.cross_entropy(logits[None, :], y[None])
                loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(bundle.status.parameters(), 5.0))
                if not math.isfinite(norm): raise FloatingPointError("Nonfinite status gradient")
                optimizer.step(); risk_updates += 1; max_grad = max(max_grad, norm); losses.append(float(loss.detach()))
            records = g47._validation_records(bundle, encoder, {"status_features": "scores-only"}, val_rows, refs["validation"])
            cal, metrics = g47.calibrate(records, "thresholds")
            entry = {"epoch": epoch, "loss": float(np.mean(losses)), "optimizer_updates": risk_updates,
                     "calibration": cal, "validation_metrics": metrics}
            history["risk"].append(entry)
            key = (metrics["selective_hmean"], metrics["exact_accuracy"], -metrics["incorrect_publication_rate"], metrics["accepted_status_recall"], -entry["loss"])
            if best_status_key is None or key > best_status_key:
                best_status_key = key; best_status = copy.deepcopy(bundle.status.state_dict())
                best_cal = copy.deepcopy(cal); best_cal_metrics = copy.deepcopy(metrics)
                selected_status_updates = risk_updates; selected_status_epoch = epoch
        bundle.status.load_state_dict(best_status)
        for p in bundle.capability.parameters(): p.requires_grad = True
        calibration, validation_metrics = best_cal, best_cal_metrics
    else:
        calibration, validation_metrics = calibrate_clause_gate(bundle, encoder, arm["gate"], val_rows, refs["validation"])
        history["risk"].append({"mode": "validation-only deterministic calibration", "calibration": calibration,
                                "validation_metrics": validation_metrics, "optimizer_updates": 0})

    selected = copy.deepcopy(bundle.state_dict())
    selected_hash = state_hash(selected)
    selected_cap_hash, selected_status_hash = _subhash(bundle.capability), _subhash(bundle.status)
    if selected_hash == initial_hash or selected_cap_hash == initial_cap_hash or max_grad <= 0:
        raise RuntimeError("Capability training failed to change selected feature model")
    if arm["gate"] == "global-scoreonly" and selected_status_hash == initial_status_hash:
        raise RuntimeError("Global status control did not train status head")
    if arm["gate"] != "global-scoreonly" and selected_status_hash != initial_status_hash:
        raise RuntimeError("Deterministic clause gate unexpectedly changed unused status head")

    torch.save({"state": selected, "arm": arm, "task": task, "encoder": encoder.meta, "seed": seed}, directory / "selected.pt")
    write_json(directory / "training.json", history)
    write_json(directory / "calibration.json", {"selected": calibration, "validation_metrics": validation_metrics,
                                                  "status_optimizer_updates": risk_updates})
    return {
        "optimizer_updates": cap_updates + risk_updates,
        "capability_optimizer_updates": cap_updates,
        "risk_optimizer_updates": risk_updates,
        "selected_capability_optimizer_updates": selected_cap_updates,
        "selected_risk_optimizer_updates": selected_status_updates,
        "selected_epoch": {"capability": selected_cap_epoch, "risk": selected_status_epoch},
        "max_gradient_norm": max_grad,
        "initial_state_hash": initial_hash, "selected_state_hash": selected_hash,
        "initial_capability_hash": initial_cap_hash, "selected_capability_hash": selected_cap_hash,
        "initial_status_hash": initial_status_hash, "selected_status_hash": selected_status_hash,
        "train_examples": len(train_rows), "validation_examples": len(val_rows),
        "curriculum_examples": len(curriculum), "trainable_parameters": sum(p.numel() for p in bundle.parameters()),
        "independent_backbone_finetuning": False, "adapter_location": "frozen-embedding feature space",
        "gate_kind": arm["gate"], "dynamic_hard_negative": arm["hardneg"],
        "risk_training_kind": "learned global score-only head" if arm["gate"] == "global-scoreonly" else "validation-only deterministic clause thresholds",
        "selected_calibration": calibration, "selected_calibration_validation_metrics": validation_metrics,
    }, calibration


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
        if record["reference"]["status"] != "accepted": continue
        gold_set = set(record["reference"]["paths"])
        pred_set = set(record["prediction"].get("selected", [])) if record["prediction"]["status"] == "accepted" else set()
        tp += len(gold_set & pred_set); gold += len(gold_set); predicted += len(pred_set)
    return {"target_recall": tp / gold if gold else None, "target_precision": tp / predicted if predicted else None,
            "targets": gold, "predicted_targets": predicted, "true_positive_targets": tp}


def run_one(arm_name, task, seed, encoder, public, refs, data_report, compositor, root, epochs):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    arm = ARMS[arm_name]
    directory = root / f"{task}-{arm_name}-{encoder.name}-seed{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    config = {"id": directory.name, "arm": arm_name, "task": task, "backbone": encoder.name, **arm,
              "model_scope": "frozen pretrained encoder + learned feature adapter/head",
              "schema_scope": "catalog projection + deterministic SDL/Federation realization" if task == "schema" else None,
              "research_question": "request-global risk gate versus validation-calibrated per-clause support/ambiguity gating"}
    write_json(directory / "config.json", config)
    bundle = g47.Bundle(encoder.dim * 5 + 4, encoder.dim)
    rss_before = current_rss_kib(); started = time.perf_counter_ns()
    receipt, calibration = train(bundle, encoder, arm, task, public, refs, seed, epochs, directory)
    training_ms = (time.perf_counter_ns() - started) / 1e6

    base_rows = [r for r in public["test"] if r["task"] == task]
    base_refs = {r["id"]: refs["test"][r["id"]] for r in base_rows}
    g46_rows, g46_refs = g47.gdm46_holdout(task)
    g47_rows, g47_refs = g47.fresh_holdout(task)
    regression_rows = base_rows + g46_rows + g47_rows
    regression_refs = {**base_refs, **g46_refs, **g47_refs}
    holdout_rows, holdout_refs = fresh_holdout(task)

    regression = evaluate(regression_rows, regression_refs, bundle, encoder, arm, calibration, compositor,
                          directory / "predictions-regression.jsonl")
    holdout = evaluate(holdout_rows, holdout_refs, bundle, encoder, arm, calibration, compositor,
                       directory / "predictions-holdout.jsonl")

    bench_rows = holdout_rows[:12]
    for row in bench_rows[:2]: validate_emission(row, infer(bundle, encoder, arm, row, calibration, online=True))
    samples = []; calls_before = encoder.calls
    for _ in range(2):
        for row in bench_rows:
            begin = time.perf_counter_ns(); pred = infer(bundle, encoder, arm, row, calibration, online=True)
            validate_emission(row, pred); samples.append((time.perf_counter_ns() - begin) / 1e6)
    calls_after = encoder.calls
    write_json(directory / "timings.json", {"generation_ms": samples, "warmup_requests": min(2, len(bench_rows)),
              "encoder_calls_measured": calls_after - calls_before,
              "scope": "single-request clause query encoding + calibrated risk gate + generation/validation; cached catalog; excludes download/index build/Rover/backend"})
    gc.collect(); rss_after = current_rss_kib()
    files = ["config.json", "training.json", "calibration.json", "initial.pt", "selected.pt",
             "predictions-regression.jsonl", "predictions-holdout.jsonl", "timings.json"]
    summary = {
        "format": FORMAT, "evidence_kind": "trained-feature-model", "config": config,
        "source_commit": os.environ.get("GITHUB_SHA", "unrecorded"), "run_id": os.environ.get("GITHUB_RUN_ID"),
        "seed": seed, "encoder": encoder.meta, "training": receipt, "training_and_selection_ms": training_ms,
        "dataset_sha256": data_report["dataset_sha256"], "dataset_scope": data_report["scope"],
        "curriculum_sha256": sha({"paths": [list(k) for k in g47.CURRICULUM], "phrases": [g47.CURRICULUM[k] for k in g47.CURRICULUM]}),
        "secondary_holdout_sha256": sha({"rows": holdout_rows, "references": holdout_refs}),
        "secondary_holdout_scope": "new GDM48 synthetic holdout with single- and multi-clause risk aggregation; never used for optimizer/checkpoint/calibration selection",
        "regression_scope": "corrected public test plus inspected GDM46 and GDM47 holdouts",
        "regression_examples": len(regression), "secondary_holdout_examples": len(holdout),
        "regression_metrics": aggregate(regression), "secondary_holdout_metrics": aggregate(holdout),
        "regression_clause_metrics": clause_metrics(regression), "secondary_holdout_clause_metrics": clause_metrics(holdout),
        "generation_latency": timing(samples), "latency_scope": "online query encoding + calibrated generation/validation; cached catalog vectors",
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
    parser.add_argument("--seeds", default="4801,4802")
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
        arms = ["clause-hybrid"]; seeds = [4801]; args.epochs = 2
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
    if errors: raise SystemExit(1)


if __name__ == "__main__":
    main()
