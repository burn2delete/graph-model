"""Actions-only decomposition of the first divergent GDM50 AdamW update.

The preceding diagnostic established that canonical input, pre-update parameters,
forward/loss, backward gradients, gradient clipping, and AdamW optimizer state agree
across hosted runners while the post-step parameter state differs. This script
reconstructs the first schema/null-calibrated/seed-5001 update into its PyTorch 2.8
single-tensor AdamW components and records stage hashes. It also records a candidate
where only the final parameter application is evaluated in float64 then cast back to
float32. No validation/holdout data is used and no measured GDM50 model is changed.
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
LR = 8e-4
WEIGHT_DECAY = 1e-3
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8


def aggregate(named: dict[str, torch.Tensor]) -> str:
    return state_hash({k: v.detach().cpu() for k, v in named.items()})


def prepare():
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise RuntimeError("GDM50 diagnostic execution belongs in GitHub Actions")
    repro.configure_reproducibility()
    g50.features_with_none = repro.features_with_none_cached
    encoder = det.ReproducibleEncoder("distilbert")
    public, refs, report = g50.corrected_dataset(Path("diagnostic-dataset"))
    examples = g50.training_examples(TASK, public, refs)
    order = list(examples)
    random.Random(SEED + 1).shuffle(order)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    bundle = g50.NullBundle(encoder.dim * 5 + 4)
    return bundle, encoder, order, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    bundle, encoder, order, report = prepare()
    row, clause, positives, source = order[0]
    optimizer = det.SingleTensorAdamW(
        bundle.capability.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    optimizer.zero_grad(set_to_none=True)
    loss = g50.null_loss(bundle, encoder, row, clause, positives, g50.ARMS[ARM_NAME])
    loss.backward()
    norm = float(det.single_tensor_clip_grad_norm_(bundle.capability.parameters(), 5.0))
    if not math.isfinite(norm):
        raise FloatingPointError("nonfinite gradient")

    p0 = {name: p.detach().clone() for name, p in bundle.capability.named_parameters()}
    grads = {name: p.grad.detach().clone() for name, p in bundle.capability.named_parameters()}
    exp_avg = {}
    exp_avg_sq = {}
    weight_decayed = {}
    denominator = {}
    standard_reconstruction = {}
    float64_application = {}
    bias_correction1 = 1.0 - BETA1
    bias_correction2 = 1.0 - BETA2
    step_size = LR / bias_correction1
    bc2_sqrt = bias_correction2 ** 0.5
    wd_factor = 1.0 - LR * WEIGHT_DECAY

    for name in sorted(p0):
        p = p0[name]
        g = grads[name]
        m = torch.zeros_like(p)
        v = torch.zeros_like(p)
        m.lerp_(g, 1 - BETA1)
        v.mul_(BETA2).addcmul_(g, g, value=1 - BETA2)
        wd = p.clone().mul_(wd_factor)
        denom = (v.sqrt() / bc2_sqrt).add_(EPS)
        reconstructed = wd.clone().addcdiv_(m, denom, value=-step_size)
        hp = (
            p.double() * wd_factor
            - step_size * (m.double() / (v.double().sqrt() / bc2_sqrt + EPS))
        ).float()
        exp_avg[name] = m
        exp_avg_sq[name] = v
        weight_decayed[name] = wd
        denominator[name] = denom
        standard_reconstruction[name] = reconstructed
        float64_application[name] = hp

    stage_hashes = {
        "pre_parameter": aggregate(p0),
        "clipped_gradient": aggregate(grads),
        "exp_avg": aggregate(exp_avg),
        "exp_avg_sq": aggregate(exp_avg_sq),
        "weight_decay_parameter": aggregate(weight_decayed),
        "denominator": aggregate(denominator),
        "standard_addcdiv_parameter": aggregate(standard_reconstruction),
        "float64_parameter_application": aggregate(float64_application),
    }

    optimizer.step()
    actual = state_hash(bundle.capability.state_dict())
    stage_hashes["actual_single_tensor_adamw_parameter"] = actual
    if actual != stage_hashes["standard_addcdiv_parameter"]:
        raise RuntimeError("manual PyTorch 2.8 first-step reconstruction does not match actual AdamW parameter state")

    opt_state = optimizer.state_dict()
    state_tensors = {}
    for param_index, state in sorted(opt_state["state"].items()):
        for key, value in sorted(state.items()):
            if isinstance(value, torch.Tensor):
                state_tensors[f"{param_index}:{key}"] = value
    stage_hashes["actual_optimizer_state_tensors"] = aggregate(state_tensors)

    doc = {
        "format": "gdm50-adamw-component-trace-v1",
        "source_commit": SOURCE_COMMIT,
        "workflow_commit": os.environ.get("GITHUB_SHA"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "task": TASK,
        "arm": ARM_NAME,
        "seed": SEED,
        "dataset_sha256": report["dataset_sha256"],
        "row_id": row["id"],
        "clause": clause,
        "positives": list(positives),
        "source": source,
        "loss": float(loss.detach()),
        "gradient_norm": norm,
        "canonical_feature_probe_sha256": encoder.meta["reproducibility_probe"]["canonical_feature_sha256"],
        "canonical_feature_corpus_sha256": encoder.meta["feature_corpus"]["canonical_text_feature_sha256"],
        "hyperparameters": {
            "lr": LR, "weight_decay": WEIGHT_DECAY, "beta1": BETA1,
            "beta2": BETA2, "eps": EPS, "step": 1,
        },
        "stage_hashes": stage_hashes,
        "interpretation": "standard_addcdiv_parameter is asserted equal to the actual single-tensor AdamW post-step state on this runner; float64_parameter_application changes only numerical evaluation of the final parameter application candidate",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event":"gdm50_adamw_component_trace_complete","stage_hashes":stage_hashes}, sort_keys=True))


if __name__ == "__main__":
    main()
