"""GDM50 downstream learned-head reproducibility wrapper.

The canonical frozen-feature corpus is stable across hosted runners at the recorded
1e-4 feature boundary, but exact GDM50 reruns still diverged during learned-head
training. Actions diagnostics localized the first schema divergence to AdamW update
step 1: canonical inputs, pre-update parameters, forward/loss bits, backward
gradients, clipping, float32 moments, and decoupled weight-decay input matched, while
the float32 bias-corrected denominator and resulting parameter application diverged.

A three-run component diagnostic showed that keeping native float32 AdamW moments but
evaluating the final decoupled-weight-decay plus adaptive parameter application in
float64, then storing parameters back in float32, produced the same post-step hash.
A subsequent three-run full-first-epoch diagnostic (Actions run 35677710836) verified
exact checkpoint-sequence and final-capability equality across all 258 updates.

This measured repair therefore preserves the optimizer family, loss, moment updates,
feature quantum, and task semantics while explicitly changing only the numerical
precision of final AdamW parameter application. AdamW remains single-tensor
(foreach=false, fused=false), clipping remains non-foreach, and evidence records the
numerical architecture change and diagnostic provenance.
"""
from __future__ import annotations

import torch

from experiments.followup50 import execute as base


_BaseAdamW = torch.optim.AdamW
_base_clip_grad_norm = torch.nn.utils.clip_grad_norm_


class HighPrecisionParameterApplyAdamW(_BaseAdamW):
    """AdamW with float32 moments and float64 final parameter application.

    PyTorch 2.8 single-tensor AdamW performs the canonical moment/state update first.
    We snapshot each pre-step float32 parameter, allow the parent optimizer to update
    moment state, then deterministically overwrite the stored parameter by recomputing
    decoupled weight decay plus the bias-corrected adaptive update in float64. The
    resulting parameter is cast back to its original float32 dtype.
    """

    def __init__(self, *args, **kwargs):
        requested_foreach = kwargs.get("foreach")
        requested_fused = kwargs.get("fused")
        if requested_foreach not in (None, False):
            raise RuntimeError("GDM50 reproducibility contract forbids foreach AdamW")
        if requested_fused not in (None, False):
            raise RuntimeError("GDM50 reproducibility contract forbids fused AdamW")
        kwargs["foreach"] = False
        kwargs["fused"] = False
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise RuntimeError("GDM50 high-precision AdamW contract does not permit closures")
        snapshots = {}
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.dtype != torch.float32:
                    raise RuntimeError(
                        "GDM50 high-precision AdamW contract requires float32 learned-head parameters"
                    )
                snapshots[id(parameter)] = parameter.detach().clone()

        loss = super().step(closure)

        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            beta1, beta2 = group["betas"]
            beta1 = float(beta1)
            beta2 = float(beta2)
            eps = float(group["eps"])
            amsgrad = bool(group["amsgrad"])
            wd_factor = 1.0 - lr * weight_decay

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                before = snapshots[id(parameter)]
                state = self.state[parameter]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                if exp_avg.dtype != torch.float32 or exp_avg_sq.dtype != torch.float32:
                    raise RuntimeError(
                        "GDM50 high-precision AdamW contract requires float32 moment state"
                    )
                step = int(state["step"].item())
                if step <= 0:
                    raise RuntimeError("GDM50 AdamW state step must be positive after parent update")
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                source_sq = state["max_exp_avg_sq"] if amsgrad else exp_avg_sq
                if source_sq.dtype != torch.float32:
                    raise RuntimeError(
                        "GDM50 high-precision AdamW contract requires float32 second-moment state"
                    )

                parameter64 = before.double() * wd_factor
                denominator64 = source_sq.double().sqrt() / (bias_correction2 ** 0.5)
                denominator64.add_(eps)
                parameter64.addcdiv_(
                    exp_avg.double(),
                    denominator64,
                    value=-(lr / bias_correction1),
                )
                parameter.copy_(parameter64.to(dtype=parameter.dtype))
        return loss


def single_tensor_clip_grad_norm_(
    parameters,
    max_norm,
    norm_type=2.0,
    error_if_nonfinite=False,
    foreach=None,
):
    """Use the non-foreach clipping path while preserving clip_grad_norm_ math."""
    if foreach not in (None, False):
        raise RuntimeError("GDM50 reproducibility contract forbids foreach gradient clipping")
    return _base_clip_grad_norm(
        parameters,
        max_norm,
        norm_type=norm_type,
        error_if_nonfinite=error_if_nonfinite,
        foreach=False,
    )


class ReproducibleEncoder(base.ReproducibleEncoder):
    """Attach the learned-head numerical execution contract to every artifact."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta = dict(self.meta)
        self.meta["head_training_reproducibility"] = {
            "adamw_foreach": False,
            "adamw_fused": False,
            "clip_grad_norm_foreach": False,
            "moment_update_dtype": "float32",
            "final_parameter_application_dtype": "float64->float32",
            "stored_parameter_dtype": "float32",
            "feature_quantum_unchanged": base.FEATURE_CANONICALIZATION_QUANTUM,
            "numerical_architecture_change": True,
            "model_objective_changed": False,
            "verified_denominator_drift_diagnostic_run": 35677485996,
            "verified_full_epoch_candidate_run": 35677710836,
            "verified_full_epoch_checkpoint_sequences_exact_match": True,
            "verified_full_epoch_final_capability_exact_match": True,
            "reason": (
                "Actions diagnostics localized first cross-run drift to the float32 "
                "AdamW bias-corrected denominator/parameter application; full-first-epoch "
                "three-run evidence verified exact reproducibility when only final "
                "decoupled-weight-decay plus adaptive parameter application is evaluated "
                "in float64 and stored back as float32"
            ),
        }


def main():
    original_adamw = torch.optim.AdamW
    original_clip = torch.nn.utils.clip_grad_norm_
    original_encoder = base.ReproducibleEncoder
    torch.optim.AdamW = HighPrecisionParameterApplyAdamW
    torch.nn.utils.clip_grad_norm_ = single_tensor_clip_grad_norm_
    base.ReproducibleEncoder = ReproducibleEncoder
    try:
        base.main()
    finally:
        torch.optim.AdamW = original_adamw
        torch.nn.utils.clip_grad_norm_ = original_clip
        base.ReproducibleEncoder = original_encoder


if __name__ == "__main__":
    main()
