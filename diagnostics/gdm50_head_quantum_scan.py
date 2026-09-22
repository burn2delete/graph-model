"""Actions-only candidate scan for GDM50 learned-head parameter canonicalization.

The preceding three-run diagnostic localized the first cross-run difference to the
AdamW parameter update: canonical inputs, pre-update parameters, forward/loss,
backward gradients, clipping, and optimizer state all matched. This diagnostic does
not use validation/holdout targets and does not change a measured experiment. It
replays the existing schema/null-calibrated/seed-5001 first training epoch with a
small, explicitly recorded set of post-optimizer parameter grids. The goal is only
to identify the smallest candidate boundary that makes the learned-head state exact
across independent hosted runners before any GDM50 architecture repair is adopted.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch

from experiments.followup50 import execute as repro
from experiments.followup50 import execute_deterministic as det
from experiments.followup50 import run as g50
from experiments.measured.models import state_hash

SOURCE_COMMIT = "c84054403c95471f15fb6b6d8363a8f12a0caf4f"
TASK = "schema"
ARM_NAME = "null-calibrated"
SEED = 5001
EPOCH = 1
CANDIDATE_QUANTA = (1e-8, 1e-7, 1e-6, 1e-5)


def canonicalize_parameters(module: torch.nn.Module, quantum: float) -> None:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.round(parameter / quantum) * quantum)


def build_context():
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise RuntimeError("GDM50 diagnostic execution belongs in GitHub Actions")
    repro.configure_reproducibility()
    g50.features_with_none = repro.features_with_none_cached
    encoder = det.ReproducibleEncoder("distilbert")
    public, refs, report = g50.corrected_dataset(Path("diagnostic-dataset"))
    examples = g50.training_examples(TASK, public, refs)
    order = list(examples)
    random.Random(SEED + EPOCH).shuffle(order)
    return encoder, order, report


def run_candidate(encoder, order, quantum: float):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    bundle = g50.NullBundle(encoder.dim * 5 + 4)
    initial = state_hash(bundle.capability.state_dict())
    optimizer = det.SingleTensorAdamW(
        bundle.capability.parameters(), lr=8e-4, weight_decay=1e-3
    )
    max_grad = 0.0
    for row, clause, positives, _source in order:
        optimizer.zero_grad(set_to_none=True)
        loss = g50.null_loss(bundle, encoder, row, clause, positives, g50.ARMS[ARM_NAME])
        loss.backward()
        norm = float(det.single_tensor_clip_grad_norm_(bundle.capability.parameters(), 5.0))
        if not math.isfinite(norm):
            raise FloatingPointError("nonfinite capability gradient")
        optimizer.step()
        canonicalize_parameters(bundle.capability, quantum)
        max_grad = max(max_grad, norm)
    return {
        "quantum": quantum,
        "initial_capability_hash": initial,
        "final_capability_hash": state_hash(bundle.capability.state_dict()),
        "optimizer_state_hash": det.stable_value_hash(optimizer.state_dict()) if hasattr(det, "stable_value_hash") else None,
        "max_gradient_norm": max_grad,
        "updates": len(order),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    encoder, order, report = build_context()
    results = [run_candidate(encoder, order, q) for q in CANDIDATE_QUANTA]
    doc = {
        "format": "gdm50-head-quantum-scan-v1",
        "source_commit": SOURCE_COMMIT,
        "workflow_commit": os.environ.get("GITHUB_SHA"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "task": TASK,
        "arm": ARM_NAME,
        "seed": SEED,
        "epoch": EPOCH,
        "dataset_sha256": report["dataset_sha256"],
        "canonical_feature_probe_sha256": encoder.meta["reproducibility_probe"]["canonical_feature_sha256"],
        "candidate_quanta": list(CANDIDATE_QUANTA),
        "results": results,
        "selection_scope": "diagnostic first-epoch exact cross-run state reproducibility only; no validation or holdout selection",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event":"gdm50_head_quantum_scan_complete","results":results}, sort_keys=True))


if __name__ == "__main__":
    main()
