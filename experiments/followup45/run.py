"""GDM45 measured follow-up.

Tests the principal failure theory from the verified GDM41-44 repair:
whole-request independent capability scoring collapses on multi-capability requests.
GDM45 compares a whole-request cardinality-aware baseline with clause decomposition,
hard-negative supervision, and per-clause local retrieval. All transformer backbones
remain frozen; learned components are feature-space adapters/heads only.

Project runs are intentionally restricted to GitHub Actions.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import resource
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.measured.contracts import (
    LANGUAGE, SPECIAL, api_sdl, catalog, judge, make_operation, option_text,
    sha, subgraph_sdls, write_dataset,
)
from experiments.measured.evidence import Compositor, aggregate, file_hash, timing, write_json
from experiments.measured.models import Encoder, Head, state_hash

FORMAT = "gdm45-measured-v1"
ARMS = {
    "whole-risk": {"kind": "whole", "hardneg": False, "top_k": 0},
    "clause-risk": {"kind": "clause", "hardneg": False, "top_k": 0},
    "clause-hardneg": {"kind": "clause", "hardneg": True, "top_k": 0},
    "clause-top2": {"kind": "clause", "hardneg": False, "top_k": 2},
    "clause-top4": {"kind": "clause", "hardneg": False, "top_k": 4},
}
STATUS_TO_INDEX = {"accepted": 0, "NO_MATCH": 1, "AMBIGUOUS": 2}
INDEX_TO_STATUS = {v: k for k, v in STATUS_TO_INDEX.items()}


class Bundle(nn.Module):
    def __init__(self, pair_dim: int, encoder_dim: int):
        super().__init__()
        # Explicit feature-space adapter. This is NOT transformer fine-tuning.
        self.capability = Head(pair_dim, adapter=True)
        self.status = nn.Sequential(
            nn.Linear(encoder_dim + 4, 96), nn.GELU(), nn.Linear(96, 3)
        )


def clauses(item):
    """Return public atomic clauses for an accepted-style request.

    This function never receives or inspects the reference answer. Cardinality is
    derived from semicolon-separated public request syntax.
    """
    request = item["request"].strip()
    root = item["catalog"]["root"]
    suffix = f" for the {root}."
    prefixes = ("Expose capabilities for ", "Return ")
    prefix = next((p for p in prefixes if request.startswith(p)), None)
    if prefix is None or not request.endswith(suffix):
        return [request]
    body = request[len(prefix):-len(suffix)]
    return [part.strip() for part in body.split("; ") if part.strip()]


def pair_features(encoder, item, query_text, online=False):
    opts = item["catalog"]["options"]
    q = encoder.query(query_text, cached=not online)
    c = encoder.encode([option_text(o, True) for o in opts])
    qx = q.expand_as(c)
    centered = c - c.mean(0, keepdim=True)
    state = torch.tensor([
        [len(o["path"]) / 5.0, float(item["task"] == "schema"),
         len(o.get("coordinates", [])) / 5.0, float(o["path"][-1] in {"name", "id"})]
        for o in opts
    ], dtype=torch.float32)
    x = torch.cat([qx, c, qx * c, (qx - c).abs(), centered, state], dim=-1)
    cosine = c @ q
    return x, opts, cosine


def status_features(encoder, item, online=False):
    q = encoder.query(item["request"], cached=not online)
    opts = item["catalog"]["options"]
    c = encoder.encode([option_text(o, True) for o in opts])
    sims = torch.sort(c @ q, descending=True).values
    first = float(sims[0]) if len(sims) else 0.0
    second = float(sims[1]) if len(sims) > 1 else first
    stats = torch.tensor([first, second, first - second, len(opts) / 10.0], dtype=torch.float32)
    return torch.cat([q, stats])


def confusing_ids(target_id, opts):
    by_suffix = {tuple(o["path"][1:]): o["id"] for o in opts}
    target = next((o for o in opts if o["id"] == target_id), None)
    if target is None:
        return []
    suffix = tuple(target["path"][1:])
    table = {
        ("createdAt",): [("updatedAt",)],
        ("updatedAt",): [("createdAt",)],
        ("title",): [("supplier", "name")],
        ("supplier", "name"): [("title",)],
        ("reviews", "author", "name"): [("reviews", "moderator", "name"), ("reviews", "author", "id")],
        ("reviews", "moderator", "name"): [("reviews", "author", "name")],
        ("reviews", "author", "id"): [("reviews", "author", "name")],
        ("reviews", "rating"): [("reviews", "author", "id")],
    }
    return [by_suffix[s] for s in table.get(suffix, []) if s in by_suffix]


def exact(prediction, reference):
    return prediction["status"] == reference["status"] and (
        reference["status"] != "accepted" or set(prediction["selected"]) == set(reference["paths"])
    )


def infer(bundle, encoder, arm, item, online=False):
    bundle.eval()
    with torch.no_grad():
        status_logits = bundle.status(status_features(encoder, item, online=online))
        status = INDEX_TO_STATUS[int(status_logits.argmax())]
        if status != "accepted":
            return {"status": status, "selected": [], "retrieved": [],
                    "status_scores": torch.softmax(status_logits, -1).tolist()}

        selected, retrieved, all_scores = [], [], {}
        if arm["kind"] == "whole":
            x, opts, cosine = pair_features(encoder, item, item["request"], online=online)
            scores = bundle.capability(x)
            count = max(1, len(clauses(item)))
            order = torch.argsort(scores, descending=True, stable=True)[:min(count, len(opts))].tolist()
            selected = [opts[i]["id"] for i in order]
            all_scores["whole"] = {o["id"]: float(s) for o, s in zip(opts, scores)}
        else:
            for clause in clauses(item):
                x, opts, cosine = pair_features(encoder, item, clause, online=online)
                allowed = list(range(len(opts)))
                if arm["top_k"]:
                    allowed = torch.argsort(cosine, descending=True, stable=True)[:min(arm["top_k"], len(opts))].tolist()
                    retrieved.append({"clause": clause, "ids": [opts[i]["id"] for i in allowed]})
                scores = bundle.capability(x)
                best = max(allowed, key=lambda i: float(scores[i]))
                selected.append(opts[best]["id"])
                all_scores[clause] = {opts[i]["id"]: float(scores[i]) for i in allowed}
        return {"status": "accepted", "selected": sorted(set(selected)),
                "retrieved": retrieved, "scores": all_scores,
                "status_scores": torch.softmax(status_logits, -1).tolist()}


def fresh_holdout(task):
    """Secondary holdout not used for checkpoint or hyperparameter selection."""
    domains = [("Album", "album"), ("Invoice", "invoice")]
    language = [
        "label presented to end users",
        "time this record was first persisted",
        "time of the newest change to this record",
        "scores attached to feedback entries",
        "display names of people who authored the feedback",
        "display names of people who moderated the feedback",
        "name of the upstream supplying organization",
        "identifiers of people who authored the feedback",
    ]
    combos = [(i,) for i in range(8)] + [(0, 1), (1, 2), (4, 5), (3, 4),
             (0, 6), (3, 7), (0, 3, 4), (1, 2, 6)]
    rows, refs = [], {}
    for entity, root in domains:
        for combo in combos:
            cat = catalog(entity, root)
            opts = cat.pop("options")
            req_prefix = "Return " if task == "operation" else "Expose capabilities for "
            request = req_prefix + "; ".join(language[i] for i in combo) + f" for the {root}."
            uid = sha(["gdm45-holdout", entity, task, request])[:24]
            targets = [opts[i]["id"] for i in combo]
            random.Random(uid).shuffle(opts)
            rows.append({"id": uid, "task": task, "request": request,
                         "catalog": {**cat, "options": opts}})
            refs[uid] = {"status": "accepted", "paths": targets, "fixture_seeds": [13, 47, 101]}
        for concept in [1, 3, 4, 6]:
            cat = catalog(entity, root)
            target = cat["options"][concept]["id"]
            cat["options"] = [o for o in cat["options"] if o["id"] != target]
            request = "Return " + language[concept] + f" for the {root}."
            uid = sha(["gdm45-holdout", entity, task, request, "missing"])[:24]
            rows.append({"id": uid, "task": task, "request": request, "catalog": cat})
            refs[uid] = {"status": "NO_MATCH", "paths": [], "fixture_seeds": [13, 47, 101]}
        request = "Show the display name of the feedback person without assuming whether they authored or moderated it."
        uid = sha(["gdm45-holdout", entity, task, request, "ambiguous"])[:24]
        rows.append({"id": uid, "task": task, "request": request, "catalog": catalog(entity, root)})
        refs[uid] = {"status": "AMBIGUOUS", "paths": [], "fixture_seeds": [13, 47, 101]}
    return rows, refs


def train(bundle, encoder, arm, task, public, refs, seed, epochs, directory):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    initial = copy.deepcopy(bundle.state_dict())
    initial_hash = state_hash(initial)
    torch.save({"state": initial, "arm": arm, "task": task}, directory / "initial.pt")
    optimizer = torch.optim.AdamW(bundle.parameters(), lr=8e-4, weight_decay=1e-3)
    class_weights = torch.tensor([1.0, 3.0, 5.0])
    train_rows = [r for r in public["train"] if r["task"] == task]
    val_rows = [r for r in public["validation"] if r["task"] == task]
    history, updates, max_grad = [], 0, 0.0
    best_state, best_key, best_epoch, selected_updates = None, None, None, 0

    for epoch in range(1, epochs + 1):
        bundle.train(); order = list(train_rows); random.Random(seed + epoch).shuffle(order)
        losses = []
        for item in order:
            ref = refs["train"][item["id"]]
            optimizer.zero_grad(set_to_none=True)
            status_logits = bundle.status(status_features(encoder, item))
            status_y = torch.tensor(STATUS_TO_INDEX[ref["status"]])
            loss = F.cross_entropy(status_logits[None, :], status_y[None], weight=class_weights)

            if ref["status"] == "accepted":
                if arm["kind"] == "whole":
                    x, opts, _ = pair_features(encoder, item, item["request"])
                    logits = bundle.capability(x)
                    positives = set(ref["paths"])
                    y = torch.tensor([float(o["id"] in positives) for o in opts])
                    loss = loss + F.binary_cross_entropy_with_logits(logits, y, pos_weight=torch.tensor(2.0))
                else:
                    parts = clauses(item)
                    if len(parts) != len(ref["paths"]):
                        raise AssertionError("Public clause cardinality differs from accepted reference cardinality")
                    for clause, target in zip(parts, ref["paths"]):
                        x, opts, _ = pair_features(encoder, item, clause)
                        logits = bundle.capability(x)
                        target_index = next(i for i, o in enumerate(opts) if o["id"] == target)
                        loss = loss + F.cross_entropy(logits[None, :], torch.tensor([target_index]))
                        if arm["hardneg"]:
                            negative_ids = confusing_ids(target, opts)
                            negative_indices = [i for i, o in enumerate(opts) if o["id"] in negative_ids]
                            if negative_indices:
                                neg = logits[negative_indices]
                                loss = loss + .35 * F.softplus(neg - logits[target_index] + .5).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(bundle.parameters(), 5.0))
            if not math.isfinite(norm):
                raise FloatingPointError("Nonfinite gradient")
            max_grad = max(max_grad, norm)
            optimizer.step(); updates += 1; losses.append(float(loss.detach()))

        bundle.eval()
        correct = sum(exact(infer(bundle, encoder, arm, row), refs["validation"][row["id"]]) for row in val_rows)
        mean_loss = float(np.mean(losses))
        entry = {"epoch": epoch, "loss": mean_loss, "optimizer_updates": updates,
                 "validation_correct": correct, "validation_examples": len(val_rows)}
        history.append(entry)
        key = (correct, -mean_loss)
        if best_key is None or key > best_key:
            best_key = key; best_state = copy.deepcopy(bundle.state_dict())
            best_epoch = epoch; selected_updates = updates
        print(json.dumps({"event": "epoch", "arm": arm, "task": task, "seed": seed, **entry}), flush=True)

    if best_state is None or state_hash(best_state) == initial_hash or max_grad <= 0:
        raise RuntimeError("Training failed to change selected feature model")
    bundle.load_state_dict(best_state); bundle.eval()
    torch.save({"state": best_state, "arm": arm, "task": task,
                "encoder": encoder.meta, "seed": seed}, directory / "selected.pt")
    write_json(directory / "training.json", history)
    return {
        "optimizer_updates": updates,
        "selected_optimizer_updates": selected_updates,
        "selected_epoch": best_epoch,
        "max_gradient_norm": max_grad,
        "initial_state_hash": initial_hash,
        "selected_state_hash": state_hash(best_state),
        "train_examples": len(train_rows),
        "validation_examples": len(val_rows),
        "trainable_parameters": sum(p.numel() for p in bundle.parameters()),
        "independent_backbone_finetuning": False,
        "adapter_location": "frozen-embedding feature space",
        "status_gate": "explicit learned 3-way accepted/NO_MATCH/AMBIGUOUS head",
    }


def validate_emission(item, prediction):
    from graphql import build_schema, parse, validate
    if prediction["status"] != "accepted":
        return
    if item["task"] == "schema":
        build_schema(api_sdl(item, prediction["selected"]))
        subgraph_sdls(item, prediction["selected"])
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
        for record in records:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
    return records


def current_rss_kib():
    statm = Path("/proc/self/statm")
    if not statm.exists():
        return None
    pages = int(statm.read_text().split()[1])
    return pages * (os.sysconf("SC_PAGE_SIZE") // 1024)


def run_one(arm_name, task, seed, encoder, public, refs, data_report, compositor, root, epochs):
    arm = ARMS[arm_name]
    directory = root / f"{task}-{arm_name}-{encoder.name}-seed{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    config = {"id": directory.name, "arm": arm_name, "task": task, "backbone": encoder.name,
              **arm, "model_scope": "frozen pretrained encoder + learned feature adapter/heads",
              "schema_scope": "catalog projection + deterministic SDL/Federation realization" if task == "schema" else None}
    write_json(directory / "config.json", config)
    pair_dim = encoder.dim * 5 + 4
    bundle = Bundle(pair_dim, encoder.dim)
    rss_before = current_rss_kib()
    started = time.perf_counter_ns()
    receipt = train(bundle, encoder, arm, task, public, refs, seed, epochs, directory)
    training_ms = (time.perf_counter_ns() - started) / 1e6

    regression_rows = [r for r in public["test"] if r["task"] == task]
    regression_refs = {r["id"]: refs["test"][r["id"]] for r in regression_rows}
    holdout_rows, holdout_refs = fresh_holdout(task)
    regression = evaluate(regression_rows, regression_refs, bundle, encoder, arm, compositor,
                          directory / "predictions-regression.jsonl")
    holdout = evaluate(holdout_rows, holdout_refs, bundle, encoder, arm, compositor,
                       directory / "predictions-holdout.jsonl")

    # Single-request online timing on secondary holdout; fresh query encoding is required.
    bench_rows = holdout_rows[:12]
    for row in bench_rows[:2]:
        validate_emission(row, infer(bundle, encoder, arm, row, online=True))
    samples = []
    calls_before = encoder.calls
    for _ in range(2):
        for row in bench_rows:
            begin = time.perf_counter_ns()
            pred = infer(bundle, encoder, arm, row, online=True)
            validate_emission(row, pred)
            samples.append((time.perf_counter_ns() - begin) / 1e6)
    calls_after = encoder.calls
    write_json(directory / "timings.json", {
        "generation_ms": samples,
        "warmup_requests": min(2, len(bench_rows)),
        "encoder_calls_measured": calls_after - calls_before,
        "scope": "single-request status encoding + whole/clause query encoding + cached catalog + learned heads + GraphQL rendering/validation; excludes model download, corpus build, Rover and fixture backend",
    })
    gc.collect()
    rss_after = current_rss_kib()
    files = ["config.json", "training.json", "initial.pt", "selected.pt",
             "predictions-regression.jsonl", "predictions-holdout.jsonl", "timings.json"]
    summary = {
        "format": FORMAT,
        "evidence_kind": "trained-feature-model",
        "config": config,
        "source_commit": os.environ.get("GITHUB_SHA", "unrecorded"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "seed": seed,
        "encoder": encoder.meta,
        "training": receipt,
        "training_and_selection_ms": training_ms,
        "dataset_sha256": data_report["dataset_sha256"],
        "dataset_scope": data_report["scope"],
        "secondary_holdout_sha256": sha({"rows": holdout_rows, "references": holdout_refs}),
        "secondary_holdout_scope": "synthetic follow-up holdout; held out from optimizer/checkpoint selection; not human-authored OOD",
        "regression_examples": len(regression),
        "secondary_holdout_examples": len(holdout),
        "regression_metrics": aggregate(regression),
        "secondary_holdout_metrics": aggregate(holdout),
        "generation_latency": timing(samples),
        "latency_scope": "online query encoding + generation/validation; cached catalog vectors",
        "memory": {
            "rss_before_training_kib": rss_before,
            "rss_after_evaluation_kib": rss_after,
            "peak_worker_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "scope": "current process RSS plus worker high-water; workers reuse one frozen backbone across arms",
        },
        "file_hashes": {name: file_hash(directory / name) for name in files},
    }
    write_json(directory / "summary.json", summary)
    print(json.dumps({"event": "completed_arm", "config": config["id"], "seed": seed,
                      "regression": summary["regression_metrics"],
                      "holdout": summary["secondary_holdout_metrics"],
                      "p50_ms": summary["generation_latency"]["p50_ms"]}), flush=True)
    return directory.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=["distilbert", "mmbert", "hash"])
    parser.add_argument("--task", required=True, choices=["operation", "schema"])
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--seeds", default="4501,4502")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output", default="artifacts/results")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise RuntimeError("Project execution belongs in GitHub Actions")
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    public, refs, data_report = write_dataset(root / "dataset")
    arms = [a for a in args.arms.split(",") if a]
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise ValueError(f"Unknown arms: {sorted(unknown)}")
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.smoke:
        arms = ["clause-risk"]; seeds = [4501]; args.epochs = 2
    write_json(root / "plan.json", {"backbone": args.backbone, "task": args.task,
                                     "arms": arms, "seeds": seeds, "epochs": args.epochs})
    encoder = Encoder(args.backbone)
    compositor = Compositor(root / "composition")
    # Known-valid composition control before model-generated schemas are interpreted.
    schema_control = next(r for r in public["train"] if r["task"] == "schema")
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
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
