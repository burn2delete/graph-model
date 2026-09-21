"""GDM46 measured risk-gating follow-up.

All project execution is restricted to GitHub Actions. GDM46 keeps the strongest
GDM45 screening choices fixed (DistilBERT + clause decomposition) and isolates
risk-gating plus dynamic hard-negative hypotheses on corrected task-consistent data.
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
    api_sdl, audit, catalog, dataset, judge, make_operation, option_text, sha,
    subgraph_sdls,
)
from experiments.measured.evidence import Compositor, aggregate, file_hash, timing, write_json
from experiments.measured.models import Encoder, Head, state_hash
from experiments.followup45.run import clauses, pair_features

FORMAT = "gdm46-measured-v1"
ARMS = {
    "joint-raw": {"training": "joint", "status_features": "raw", "hardneg": False},
    "balanced-raw": {"training": "staged-balanced", "status_features": "raw", "hardneg": False},
    "balanced-learned": {"training": "staged-balanced", "status_features": "learned", "hardneg": False},
    "balanced-learned-hardneg": {"training": "staged-balanced", "status_features": "learned", "hardneg": True},
}
STATUS_TO_INDEX = {"accepted": 0, "NO_MATCH": 1, "AMBIGUOUS": 2}
INDEX_TO_STATUS = {v: k for k, v in STATUS_TO_INDEX.items()}
AMBIGUOUS_LANGUAGE = {
    "train": "name of the person related to each review without saying whether that person authored or moderated it",
    "validation": "display name of each review person without choosing the author role or moderator role",
    "test": "display name of the feedback person without specifying writer versus moderator",
}


class Bundle(nn.Module):
    def __init__(self, pair_dim: int, encoder_dim: int):
        super().__init__()
        self.capability = Head(pair_dim, adapter=True)
        # Query embedding + 10 measured/catalog statistics. This remains a learned
        # feature-space head over a frozen encoder, not transformer fine-tuning.
        self.status = nn.Sequential(nn.Linear(encoder_dim + 10, 96), nn.GELU(), nn.Linear(96, 3))


def task_prefix(task: str) -> str:
    return "Return " if task == "operation" else "Expose capabilities for "


def corrected_dataset(directory):
    """Write a corrected public/reference dataset with no task-prefix risk shortcut."""
    public, refs = dataset()
    for split, rows in public.items():
        rebuilt_refs = {}
        for row in rows:
            old_id = row["id"]
            ref = refs[split][old_id]
            if ref["status"] != "accepted":
                root = row["catalog"]["root"]
                if ref["status"] == "NO_MATCH":
                    text = row["request"]
                    if text.startswith("Return "):
                        text = text[len("Return "):]
                    elif text.startswith("Expose capabilities for "):
                        text = text[len("Expose capabilities for "):]
                    row["request"] = task_prefix(row["task"]) + text
                else:
                    row["request"] = (
                        task_prefix(row["task"]) + AMBIGUOUS_LANGUAGE[split] + f" for the {root}."
                    )
                row["id"] = sha([
                    "gdm46-corrected", split, row["catalog"]["entity"], row["task"],
                    ref["status"], row["request"],
                ])[:24]
            rebuilt_refs[row["id"]] = ref
        refs[split] = rebuilt_refs
    report = audit(public, refs)
    report["benchmark_correction"] = (
        "Risk examples use the same task-specific request prefix as accepted examples; "
        "the GDM45 schema prefix shortcut is removed."
    )
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    for split in public:
        (directory / f"{split}.public.json").write_text(json.dumps(public[split], indent=2))
        (directory / f"{split}.references.json").write_text(json.dumps(refs[split], indent=2))
    (directory / "audit.json").write_text(json.dumps(report, indent=2))
    return public, refs, report


def fresh_holdout(task: str):
    """New synthetic holdout; never used for optimizer or checkpoint selection."""
    domains = [("Playlist", "playlist"), ("Shipment", "shipment")]
    language = [
        "heading visible to people using the interface",
        "timestamp from the first insertion of this record",
        "timestamp from the latest revision of this record",
        "numeric score on each evaluation entry",
        "names of the people who created the reviews",
        "names of the people who oversee the reviews",
        "name of the vendor providing this item",
        "identifiers of the people who created the reviews",
    ]
    combos = [(i,) for i in range(8)] + [(0, 1), (1, 2), (4, 5), (3, 4),
             (0, 6), (3, 7), (0, 3, 4), (1, 2, 6)]
    rows, refs = [], {}
    for entity, root in domains:
        for combo in combos:
            cat = catalog(entity, root)
            opts = cat.pop("options")
            request = task_prefix(task) + "; ".join(language[i] for i in combo) + f" for the {root}."
            uid = sha(["gdm46-holdout", entity, task, request])[:24]
            targets = [opts[i]["id"] for i in combo]
            random.Random(uid).shuffle(opts)
            rows.append({"id": uid, "task": task, "request": request,
                         "catalog": {**cat, "options": opts}})
            refs[uid] = {"status": "accepted", "paths": targets, "fixture_seeds": [13, 47, 101]}
        for concept in [1, 3, 4, 6]:
            cat = catalog(entity, root)
            target = cat["options"][concept]["id"]
            cat["options"] = [o for o in cat["options"] if o["id"] != target]
            random.Random(entity + task + str(concept)).shuffle(cat["options"])
            request = task_prefix(task) + language[concept] + f" for the {root}."
            uid = sha(["gdm46-holdout", entity, task, request, "missing"])[:24]
            rows.append({"id": uid, "task": task, "request": request, "catalog": cat})
            refs[uid] = {"status": "NO_MATCH", "paths": [], "fixture_seeds": [13, 47, 101]}
        request = (task_prefix(task) +
                   "visible name of the review participant without identifying whether the author or moderator relationship applies" +
                   f" for the {root}.")
        uid = sha(["gdm46-holdout", entity, task, request, "ambiguous"])[:24]
        rows.append({"id": uid, "task": task, "request": request, "catalog": catalog(entity, root)})
        refs[uid] = {"status": "AMBIGUOUS", "paths": [], "fixture_seeds": [13, 47, 101]}
    return rows, refs


def _two(values):
    values = torch.sort(values, descending=True).values
    first = float(values[0]) if len(values) else 0.0
    second = float(values[1]) if len(values) > 1 else first
    return first, second, first - second


def risk_features(bundle, encoder, arm, item, online=False):
    """Return status features and reusable learned clause scores.

    The learned statistics are detached from capability training. Status selection can
    use them, but status gradients never alter the capability policy in staged arms.
    """
    q = encoder.query(item["request"], cached=not online)
    opts = item["catalog"]["options"]
    c = encoder.encode([option_text(o, True) for o in opts])
    raw_first, raw_second, raw_margin = _two(c @ q)
    clause_cache, maxima, margins, entropies = [], [], [], []
    if arm["status_features"] == "learned":
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
    if maxima:
        learned = [min(maxima), float(np.mean(maxima)), min(margins), float(np.mean(margins)), max(entropies)]
    else:
        learned = [0.0] * 5
    stats = torch.tensor([
        raw_first, raw_second, raw_margin, len(opts) / 10.0,
        *learned, len(clauses(item)) / 4.0,
    ], dtype=torch.float32)
    return torch.cat([q, stats]), clause_cache


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


def infer(bundle, encoder, arm, item, online=False):
    bundle.eval()
    with torch.no_grad():
        features, cache = risk_features(bundle, encoder, arm, item, online=online)
        status_logits = bundle.status(features)
        status = INDEX_TO_STATUS[int(status_logits.argmax())]
        result = {"status": status, "selected": [],
                  "status_scores": torch.softmax(status_logits, -1).tolist()}
        if status != "accepted":
            return result
        selected, scores = select_capabilities(bundle, encoder, item, online=online, cache=cache)
        result.update({"selected": selected, "scores": scores})
        return result


def exact(prediction, reference):
    return prediction["status"] == reference["status"] and (
        reference["status"] != "accepted" or set(prediction["selected"]) == set(reference["paths"])
    )


def _subhash(module):
    return state_hash(copy.deepcopy(module.state_dict()))


def balanced_status_rows(rows, refs, seed):
    groups = {k: [] for k in STATUS_TO_INDEX}
    for row in rows:
        groups[refs[row["id"]]["status"]].append(row)
    if any(not v for v in groups.values()):
        raise AssertionError("Every status class must exist in training")
    target = max(len(v) for v in groups.values())
    out = []
    for label, group in groups.items():
        ordered = list(group)
        random.Random(seed + STATUS_TO_INDEX[label] * 1009).shuffle(ordered)
        out.extend(ordered[i % len(ordered)] for i in range(target))
    random.Random(seed + 7919).shuffle(out)
    return out


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


def _status_macro(bundle, encoder, arm, rows, refs):
    by_class = {k: [0, 0] for k in STATUS_TO_INDEX}
    exact_total = 0
    for row in rows:
        ref = refs[row["id"]]
        pred = infer(bundle, encoder, arm, row)
        label = ref["status"]
        by_class[label][1] += 1
        by_class[label][0] += int(pred["status"] == label)
        exact_total += int(exact(pred, ref))
    macro = float(np.mean([c / n for c, n in by_class.values() if n]))
    return macro, exact_total, by_class


def train(bundle, encoder, arm, task, public, refs, seed, epochs, directory):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    initial = copy.deepcopy(bundle.state_dict())
    initial_hash = state_hash(initial)
    initial_cap_hash, initial_status_hash = _subhash(bundle.capability), _subhash(bundle.status)
    torch.save({"state": initial, "arm": arm, "task": task}, directory / "initial.pt")
    train_rows = [r for r in public["train"] if r["task"] == task]
    val_rows = [r for r in public["validation"] if r["task"] == task]
    accepted_train = [r for r in train_rows if refs["train"][r["id"]]["status"] == "accepted"]
    accepted_val = [r for r in val_rows if refs["validation"][r["id"]]["status"] == "accepted"]
    history = {"capability": [], "status": [], "joint": []}
    cap_updates = status_updates = 0
    max_grad = 0.0

    if arm["training"] == "joint":
        optimizer = torch.optim.AdamW(bundle.parameters(), lr=8e-4, weight_decay=1e-3)
        class_weights = torch.tensor([1.0, 3.0, 5.0])
        best_state = None; best_key = None; selected_updates = 0; selected_epoch = 0
        for epoch in range(1, epochs + 1):
            order = list(train_rows); random.Random(seed + epoch).shuffle(order)
            losses = []; bundle.train()
            for row in order:
                ref = refs["train"][row["id"]]
                optimizer.zero_grad(set_to_none=True)
                features, _ = risk_features(bundle, encoder, arm, row)
                logits = bundle.status(features)
                y = torch.tensor(STATUS_TO_INDEX[ref["status"]])
                loss = F.cross_entropy(logits[None, :], y[None], weight=class_weights)
                if ref["status"] == "accepted":
                    loss = loss + capability_loss(bundle, encoder, row, ref, False)
                loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(bundle.parameters(), 5.0))
                if not math.isfinite(norm): raise FloatingPointError("Nonfinite gradient")
                optimizer.step(); cap_updates += int(ref["status"] == "accepted"); status_updates += 1
                max_grad = max(max_grad, norm); losses.append(float(loss.detach()))
            bundle.eval(); macro, exact_total, counts = _status_macro(bundle, encoder, arm, val_rows, refs["validation"])
            entry = {"epoch": epoch, "loss": float(np.mean(losses)), "status_macro_accuracy": macro,
                     "validation_exact": exact_total, "validation_examples": len(val_rows),
                     "status_counts": counts}
            history["joint"].append(entry)
            key = (exact_total, macro, -entry["loss"])
            if best_key is None or key > best_key:
                best_key = key; best_state = copy.deepcopy(bundle.state_dict())
                selected_updates = status_updates; selected_epoch = epoch
            print(json.dumps({"event": "joint_epoch", "arm": arm, "task": task, "seed": seed, **entry}), flush=True)
        bundle.load_state_dict(best_state)
        selected_cap_updates = cap_updates
        selected_status_updates = selected_updates
    else:
        # Phase 1: capability policy only, selected only on answerable validation cases.
        optimizer = torch.optim.AdamW(bundle.capability.parameters(), lr=8e-4, weight_decay=1e-3)
        best_cap = None; best_cap_key = None; selected_cap_updates = 0; cap_selected_epoch = 0
        for epoch in range(1, epochs + 1):
            order = list(accepted_train); random.Random(seed + epoch).shuffle(order)
            losses = []; bundle.train()
            for row in order:
                optimizer.zero_grad(set_to_none=True)
                ref = refs["train"][row["id"]]
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
            entry = {"epoch": epoch, "loss": float(np.mean(losses)), "accepted_validation_correct": correct,
                     "accepted_validation_examples": len(accepted_val), "optimizer_updates": cap_updates}
            history["capability"].append(entry)
            key = (correct, -entry["loss"])
            if best_cap_key is None or key > best_cap_key:
                best_cap_key = key; best_cap = copy.deepcopy(bundle.capability.state_dict())
                selected_cap_updates = cap_updates; cap_selected_epoch = epoch
            print(json.dumps({"event": "capability_epoch", "arm": arm, "task": task, "seed": seed, **entry}), flush=True)
        bundle.capability.load_state_dict(best_cap)
        for p in bundle.capability.parameters(): p.requires_grad = False

        # Phase 2: status head only, with deterministic class balancing.
        optimizer = torch.optim.AdamW(bundle.status.parameters(), lr=8e-4, weight_decay=1e-3)
        best_status = None; best_status_key = None; selected_status_updates = 0; status_selected_epoch = 0
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
            bundle.eval(); macro, exact_total, counts = _status_macro(bundle, encoder, arm, val_rows, refs["validation"])
            entry = {"epoch": epoch, "loss": float(np.mean(losses)), "status_macro_accuracy": macro,
                     "validation_exact": exact_total, "validation_examples": len(val_rows),
                     "status_counts": counts, "optimizer_updates": status_updates}
            history["status"].append(entry)
            key = (macro, exact_total, -entry["loss"])
            if best_status_key is None or key > best_status_key:
                best_status_key = key; best_status = copy.deepcopy(bundle.status.state_dict())
                selected_status_updates = status_updates; status_selected_epoch = epoch
            print(json.dumps({"event": "status_epoch", "arm": arm, "task": task, "seed": seed, **entry}), flush=True)
        bundle.status.load_state_dict(best_status)
        for p in bundle.capability.parameters(): p.requires_grad = True
        selected_epoch = {"capability": cap_selected_epoch, "status": status_selected_epoch}

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
    return {
        "optimizer_updates": cap_updates + status_updates,
        "capability_optimizer_updates": cap_updates,
        "status_optimizer_updates": status_updates,
        "selected_capability_optimizer_updates": selected_cap_updates,
        "selected_status_optimizer_updates": selected_status_updates,
        "selected_epoch": selected_epoch,
        "max_gradient_norm": max_grad,
        "initial_state_hash": initial_hash,
        "selected_state_hash": selected_hash,
        "initial_capability_hash": initial_cap_hash,
        "selected_capability_hash": selected_cap_hash,
        "initial_status_hash": initial_status_hash,
        "selected_status_hash": selected_status_hash,
        "train_examples": len(train_rows),
        "validation_examples": len(val_rows),
        "trainable_parameters": sum(p.numel() for p in bundle.parameters()),
        "independent_backbone_finetuning": False,
        "adapter_location": "frozen-embedding feature space",
        "status_gate": arm["status_features"],
        "training_mode": arm["training"],
        "dynamic_hard_negative": arm["hardneg"],
    }


def validate_emission(item, prediction):
    from graphql import build_schema, parse, validate
    if prediction["status"] != "accepted": return
    if item["task"] == "schema":
        build_schema(api_sdl(item, prediction["selected"])); subgraph_sdls(item, prediction["selected"])
    else:
        validate(build_schema(api_sdl(item)), parse(make_operation(item, prediction["selected"])))


def evaluate(rows, refs, bundle, encoder, arm, compositor, path):
    records = []
    for item in rows:
        prediction = infer(bundle, encoder, arm, item)
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
        for record in records: handle.write(json.dumps(record, allow_nan=False) + "\n")
    return records


def current_rss_kib():
    statm = Path("/proc/self/statm")
    if not statm.exists(): return None
    pages = int(statm.read_text().split()[1])
    return pages * (os.sysconf("SC_PAGE_SIZE") // 1024)


def run_one(arm_name, task, seed, encoder, public, refs, data_report, compositor, root, epochs):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    arm = ARMS[arm_name]
    directory = root / f"{task}-{arm_name}-{encoder.name}-seed{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    config = {"id": directory.name, "arm": arm_name, "task": task, "backbone": encoder.name,
              **arm, "model_scope": "frozen pretrained encoder + learned feature adapter/heads",
              "schema_scope": "catalog projection + deterministic SDL/Federation realization" if task == "schema" else None,
              "benchmark_correction": "task-consistent accepted/NO_MATCH/AMBIGUOUS request syntax"}
    write_json(directory / "config.json", config)
    bundle = Bundle(encoder.dim * 5 + 4, encoder.dim)
    rss_before = current_rss_kib(); started = time.perf_counter_ns()
    receipt = train(bundle, encoder, arm, task, public, refs, seed, epochs, directory)
    training_ms = (time.perf_counter_ns() - started) / 1e6

    regression_rows = [r for r in public["test"] if r["task"] == task]
    regression_refs = {r["id"]: refs["test"][r["id"]] for r in regression_rows}
    holdout_rows, holdout_refs = fresh_holdout(task)
    regression = evaluate(regression_rows, regression_refs, bundle, encoder, arm, compositor,
                          directory / "predictions-regression.jsonl")
    holdout = evaluate(holdout_rows, holdout_refs, bundle, encoder, arm, compositor,
                       directory / "predictions-holdout.jsonl")

    bench_rows = holdout_rows[:12]
    for row in bench_rows[:2]: validate_emission(row, infer(bundle, encoder, arm, row, online=True))
    samples = []; calls_before = encoder.calls
    for _ in range(2):
        for row in bench_rows:
            begin = time.perf_counter_ns(); pred = infer(bundle, encoder, arm, row, online=True)
            validate_emission(row, pred); samples.append((time.perf_counter_ns() - begin) / 1e6)
    calls_after = encoder.calls
    write_json(directory / "timings.json", {
        "generation_ms": samples, "warmup_requests": min(2, len(bench_rows)),
        "encoder_calls_measured": calls_after - calls_before,
        "scope": "single-request status + clause query encoding, cached catalog, learned heads, GraphQL rendering/validation; excludes download, index build, Rover and fixture backend",
    })
    gc.collect(); rss_after = current_rss_kib()
    files = ["config.json", "training.json", "initial.pt", "selected.pt",
             "predictions-regression.jsonl", "predictions-holdout.jsonl", "timings.json"]
    summary = {
        "format": FORMAT, "evidence_kind": "trained-feature-model", "config": config,
        "source_commit": os.environ.get("GITHUB_SHA", "unrecorded"), "run_id": os.environ.get("GITHUB_RUN_ID"),
        "seed": seed, "encoder": encoder.meta, "training": receipt,
        "training_and_selection_ms": training_ms,
        "dataset_sha256": data_report["dataset_sha256"], "dataset_scope": data_report["scope"],
        "benchmark_correction": data_report["benchmark_correction"],
        "secondary_holdout_sha256": sha({"rows": holdout_rows, "references": holdout_refs}),
        "secondary_holdout_scope": "new synthetic follow-up holdout; not used for optimizer/checkpoint selection; not human-authored OOD",
        "regression_examples": len(regression), "secondary_holdout_examples": len(holdout),
        "regression_metrics": aggregate(regression), "secondary_holdout_metrics": aggregate(holdout),
        "generation_latency": timing(samples), "latency_scope": "online query encoding + generation/validation; cached catalog vectors",
        "memory": {"rss_before_training_kib": rss_before, "rss_after_evaluation_kib": rss_after,
                   "peak_worker_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "scope": "current process RSS plus worker high-water; worker reuses frozen backbone across arms"},
        "file_hashes": {name: file_hash(directory / name) for name in files},
    }
    write_json(directory / "summary.json", summary)
    metrics = summary["secondary_holdout_metrics"][task]
    print(json.dumps({"event": "completed_arm", "config": config["id"], "seed": seed,
                      "holdout": metrics, "p50_ms": summary["generation_latency"]["p50_ms"]}), flush=True)
    return directory.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="distilbert", choices=["distilbert", "hash"])
    parser.add_argument("--task", required=True, choices=["operation", "schema"])
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--seeds", default="4601,4602")
    parser.add_argument("--epochs", type=int, default=10)
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
    if args.smoke: arms = ["balanced-learned"]; seeds = [4601]; args.epochs = 2
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
                error = {"arm": arm, "task": args.task, "seed": seed, "traceback": traceback.format_exc()}
                errors.append(error); print(json.dumps({"event": "failed_arm", **error}), flush=True)
    write_json(root / "worker.json", {"complete": not errors, "completed": completed,
                                       "errors": errors, "expected": len(arms) * len(seeds),
                                       "encoder": encoder.meta})
    if errors: raise SystemExit(1)


if __name__ == "__main__":
    main()
