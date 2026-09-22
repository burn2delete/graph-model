"""Actions-only GDM50 learned-head numerical reproducibility diagnostic.

This is diagnostic instrumentation, not a new experiment arm. It reproduces the
first epoch of the existing GDM50 schema null-calibrated training path and records
exact tensor/state hashes at each learned-head stage so two independent hosted
runners can identify whether the first cross-run divergence occurs in the canonical
input features, learned-head forward pass, backward gradients, clipping, or AdamW
update. No holdout or benchmark selection is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from experiments.followup50 import execute as repro
from experiments.followup50 import execute_deterministic as det
from experiments.followup50 import run as g50
from experiments.measured.models import state_hash

SOURCE_COMMIT = "c84054403c95471f15fb6b6d8363a8f12a0caf4f"
ARM_NAME = "null-calibrated"
TASK = "schema"
SEED = 5001
EPOCH = 1


def tensor_hash(t: torch.Tensor) -> str:
    t = t.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(t.dtype).encode())
    h.update(json.dumps(list(t.shape)).encode())
    h.update(t.numpy().tobytes())
    return h.hexdigest()


def named_grad_hash(module: torch.nn.Module) -> tuple[str, dict[str, str | None]]:
    h = hashlib.sha256()
    pieces: dict[str, str | None] = {}
    for name, param in sorted(module.named_parameters()):
        grad = param.grad
        digest = None if grad is None else tensor_hash(grad)
        pieces[name] = digest
        h.update(name.encode())
        h.update((digest or "NONE").encode())
    return h.hexdigest(), pieces


def stable_value_hash(value: Any) -> str:
    h = hashlib.sha256()

    def visit(x: Any) -> None:
        if isinstance(x, torch.Tensor):
            h.update(b"tensor")
            h.update(tensor_hash(x).encode())
        elif isinstance(x, dict):
            h.update(b"dict")
            for k in sorted(x, key=lambda y: str(y)):
                h.update(str(k).encode())
                visit(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(type(x).__name__.encode())
            for item in x:
                visit(item)
        elif isinstance(x, float):
            h.update(b"float")
            h.update(struct.pack("!d", x))
        elif isinstance(x, (int, bool, str)) or x is None:
            h.update(repr(x).encode())
        else:
            h.update(repr(x).encode())

    visit(value)
    return h.hexdigest()


def optimizer_state_hash(optimizer: torch.optim.Optimizer) -> str:
    state = optimizer.state_dict()
    return stable_value_hash(state)


def loss_bits(loss: torch.Tensor) -> str:
    value = float(loss.detach().cpu())
    return struct.pack("!d", value).hex()


def prepare() -> tuple[g50.NullBundle, repro.ReproducibleEncoder, list, dict, dict]:
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise RuntimeError("GDM50 diagnostic execution belongs in GitHub Actions")
    repro.configure_reproducibility()
    # Match the measured GDM50 cache-correct feature path and head implementation.
    g50.features_with_none = repro.features_with_none_cached
    encoder = det.ReproducibleEncoder("distilbert")
    public, refs, report = g50.corrected_dataset(Path("diagnostic-dataset"))
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    bundle = g50.NullBundle(encoder.dim * 5 + 4)
    examples = g50.training_examples(TASK, public, refs)
    order = list(examples)
    random.Random(SEED + EPOCH).shuffle(order)
    return bundle, encoder, order, public, report


def run_trace(output: Path) -> None:
    bundle, encoder, order, _public, data_report = prepare()
    arm = g50.ARMS[ARM_NAME]
    optimizer = det.SingleTensorAdamW(
        bundle.capability.parameters(), lr=8e-4, weight_decay=1e-3
    )
    initial_state = state_hash(bundle.capability.state_dict())
    trace = []

    for step, (row, clause, positives, source) in enumerate(order, 1):
        optimizer.zero_grad(set_to_none=True)
        x, opts, _raw = g50.features_with_none(encoder, row, clause)
        pre_state = state_hash(bundle.capability.state_dict())
        input_hash = tensor_hash(x)
        logits = bundle.capability(x)
        logits_hash = tensor_hash(logits)
        ids = [o["id"] for o in opts]
        target = torch.zeros_like(logits)
        if positives:
            indices = [ids.index(p) for p in positives]
            target[indices] = 1.0 / len(indices)
            target_source = "real"
        else:
            target[-1] = 1.0
            target_source = "none"
        loss = -(target * F.log_softmax(logits, dim=-1)).sum()
        if target_source == "none":
            loss = loss * float(arm["none_balance"])
        if len(positives) > 1:
            loss = loss * 1.15
        if arm.get("contrast"):
            pos = logits[target > 0]
            neg = logits[target == 0]
            if len(pos) and len(neg):
                loss = loss + 0.35 * F.softplus(neg.max() - pos.min() + 0.65)

        forward = {
            "input_hash": input_hash,
            "target_hash": tensor_hash(target),
            "pre_state_hash": pre_state,
            "logits_hash": logits_hash,
            "loss_bits": loss_bits(loss),
            "loss": float(loss.detach()),
        }
        loss.backward()
        grad_before_hash, grad_before = named_grad_hash(bundle.capability)
        norm = float(det.single_tensor_clip_grad_norm_(bundle.capability.parameters(), 5.0))
        if not math.isfinite(norm):
            raise FloatingPointError("nonfinite capability gradient")
        grad_after_hash, grad_after = named_grad_hash(bundle.capability)
        optimizer.step()
        post_state = state_hash(bundle.capability.state_dict())
        opt_hash = optimizer_state_hash(optimizer)

        trace.append(
            {
                "step": step,
                "row_id": row["id"],
                "clause": clause,
                "positives": list(positives),
                "source": source,
                "encoder_canonical_corpus_sha256": encoder.meta["feature_corpus"]["canonical_text_feature_sha256"],
                "encoder_raw_corpus_sha256": encoder.meta["feature_corpus"]["raw_text_feature_sha256"],
                **forward,
                "grad_before_clip_hash": grad_before_hash,
                "grad_before_clip": grad_before,
                "grad_norm_bits": struct.pack("!d", norm).hex(),
                "grad_norm": norm,
                "grad_after_clip_hash": grad_after_hash,
                "grad_after_clip": grad_after,
                "post_state_hash": post_state,
                "optimizer_state_hash": opt_hash,
            }
        )

    doc = {
        "format": "gdm50-head-repro-trace-v1",
        "source_commit": SOURCE_COMMIT,
        "workflow_commit": os.environ.get("GITHUB_SHA"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "job": os.environ.get("GITHUB_JOB"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "runner_arch": os.environ.get("RUNNER_ARCH"),
        "runner_os": os.environ.get("RUNNER_OS"),
        "task": TASK,
        "arm": ARM_NAME,
        "seed": SEED,
        "epoch": EPOCH,
        "updates": len(trace),
        "initial_capability_hash": initial_state,
        "dataset_sha256": data_report["dataset_sha256"],
        "encoder": encoder.meta,
        "head_contract": {
            "adamw_foreach": False,
            "adamw_fused": False,
            "clip_grad_norm_foreach": False,
        },
        "trace": trace,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "event": "gdm50_head_trace_complete",
        "updates": len(trace),
        "initial_capability_hash": initial_state,
        "final_capability_hash": trace[-1]["post_state_hash"],
        "canonical_feature_corpus": encoder.meta["feature_corpus"]["canonical_text_feature_sha256"],
    }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_trace(Path(args.output))


if __name__ == "__main__":
    main()
