"""GDM50 downstream learned-head reproducibility wrapper.

The canonical frozen-feature corpus is already stable across hosted runners at the
recorded 1e-4 feature boundary, but exact reruns still diverged during schema head
training. This wrapper therefore constrains PyTorch's downstream optimizer and
gradient-clipping implementation paths without changing the intended model,
optimizer, loss, feature quantum, or task semantics.

In particular, AdamW is forced onto its single-tensor implementation and gradient
norm clipping is forced off the foreach implementation. These controls target the
remaining CPU-kernel reduction/update nondeterminism while preserving the same
mathematical training procedure. Evidence records the execution contract.
"""
from __future__ import annotations

import torch

from experiments.followup50 import execute as base


_BaseAdamW = torch.optim.AdamW
_base_clip_grad_norm = torch.nn.utils.clip_grad_norm_


class SingleTensorAdamW(_BaseAdamW):
    """AdamW with explicitly non-foreach, non-fused CPU update kernels."""

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
    """Attach the downstream learned-head execution contract to every artifact."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta = dict(self.meta)
        self.meta["head_training_reproducibility"] = {
            "adamw_foreach": False,
            "adamw_fused": False,
            "clip_grad_norm_foreach": False,
            "feature_quantum_unchanged": base.FEATURE_CANONICALIZATION_QUANTUM,
            "model_semantics_changed": False,
            "reason": "canonical frozen feature corpus matched across reruns while downstream schema head training diverged; force single-tensor CPU optimizer/reduction paths",
        }


def main():
    original_adamw = torch.optim.AdamW
    original_clip = torch.nn.utils.clip_grad_norm_
    original_encoder = base.ReproducibleEncoder
    torch.optim.AdamW = SingleTensorAdamW
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
