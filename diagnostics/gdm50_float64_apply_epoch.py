"""Actions-only full first-epoch test of a high-precision AdamW parameter application.

Prior GDM50 diagnostics established that cross-run drift first appears in AdamW's
float32 denominator / parameter application even when canonical frozen features,
forward/loss, backward gradients, clipping, optimizer moments, and pre-step weights
match exactly. The component diagnostic also showed that recomputing the first final
parameter application in float64 and casting back to float32 was exact on three
independent hosted runners.

This diagnostic extends that candidate across the entire first training epoch for the
existing schema/null-calibrated/seed-5001 path. It does not use validation or holdout
selection and does not modify the measured GDM50 implementation.
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


class Float64ApplyAdamW(det.SingleTensorAdamW):
    """Keep AdamW moments in their native dtype, but apply the final step in float64."""

    @torch.no_grad()
    def step(self, closure=None):
        snapshots = {
            id(p): p.detach().clone()
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        }
        loss = super().step(closure)
        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            amsgrad = bool(group["amsgrad"])
            wd_factor = 1.0 - lr * weight_decay
            for p in group["params"]:
                if p.grad is None:
                    continue
                before = snapshots[id(p)]
                state = self.state[p]
                step = int(state["step"].item())
                bc1 = 1.0 - beta1 ** step
                bc2 = 1.0 - beta2 ** step
                source_sq = state["max_exp_avg_sq"] if amsgrad else state["exp_avg_sq"]
                p64 = before.double() * wd_factor
                denom64 = source_sq.double().sqrt() / (bc2 ** 0.5)
                denom64.add_(eps)
                p64.addcdiv_(state["exp_avg"].double(), denom64, value=-(lr / bc1))
                p.copy_(p64.to(dtype=p.dtype))
        return loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise RuntimeError("GDM50 diagnostic execution belongs in GitHub Actions")
    repro.configure_reproducibility()
    g50.features_with_none = repro.features_with_none_cached
    encoder = det.ReproducibleEncoder("distilbert")
    public, refs, data_report = g50.corrected_dataset(Path("diagnostic-dataset"))
    examples = g50.training_examples(TASK, public, refs)
    order = list(examples)
    random.Random(SEED + EPOCH).shuffle(order)

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    bundle = g50.NullBundle(encoder.dim * 5 + 4)
    initial = state_hash(bundle.capability.state_dict())
    optimizer = Float64ApplyAdamW(
        bundle.capability.parameters(), lr=8e-4, weight_decay=1e-3
    )
    losses = []
    max_grad = 0.0
    checkpoints = []
    for step, (row, clause, positives, source) in enumerate(order, 1):
        optimizer.zero_grad(set_to_none=True)
        loss = g50.null_loss(bundle, encoder, row, clause, positives, g50.ARMS[ARM_NAME])
        loss.backward()
        norm = float(det.single_tensor_clip_grad_norm_(bundle.capability.parameters(), 5.0))
        if not math.isfinite(norm):
            raise FloatingPointError("nonfinite capability gradient")
        optimizer.step()
        losses.append(float(loss.detach()))
        max_grad = max(max_grad, norm)
        if step in {1, 2, 4, 8, 16, 32, 64, len(order)}:
            checkpoints.append({"step": step, "capability_hash": state_hash(bundle.capability.state_dict())})

    doc = {
        "format": "gdm50-float64-apply-epoch-v1",
        "source_commit": SOURCE_COMMIT,
        "workflow_commit": os.environ.get("GITHUB_SHA"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "task": TASK,
        "arm": ARM_NAME,
        "seed": SEED,
        "epoch": EPOCH,
        "dataset_sha256": data_report["dataset_sha256"],
        "canonical_feature_probe_sha256": encoder.meta["reproducibility_probe"]["canonical_feature_sha256"],
        "canonical_feature_corpus_sha256": encoder.meta["feature_corpus"]["canonical_text_feature_sha256"],
        "initial_capability_hash": initial,
        "final_capability_hash": state_hash(bundle.capability.state_dict()),
        "updates": len(order),
        "mean_loss": float(np.mean(losses)),
        "max_gradient_norm": max_grad,
        "checkpoints": checkpoints,
        "candidate_contract": {
            "optimizer_family": "AdamW",
            "moment_update_dtype": "float32",
            "moment_update_path": "PyTorch 2.8 single-tensor AdamW foreach=false fused=false",
            "final_parameter_application_dtype": "float64",
            "stored_parameter_dtype": "float32",
            "frozen_feature_quantum_unchanged": 1e-4,
            "selection_scope": "first training epoch reproducibility diagnostic only; no validation or holdout selection",
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "event": "gdm50_float64_apply_epoch_complete",
        "initial_capability_hash": initial,
        "final_capability_hash": doc["final_capability_hash"],
        "updates": len(order),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
