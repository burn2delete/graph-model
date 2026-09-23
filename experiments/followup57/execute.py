"""Canonical deterministic execution wrapper for GDM57.

GDM57 inherits the canonical GDM50 numerical repair and the canonical GDM56
capability/curriculum/arbitration line. Only the top-two semantic representation
exposed to the learned ambiguity head changes.
"""
from __future__ import annotations

import torch

from experiments.followup50 import execute as g50exec
from experiments.followup50 import execute_deterministic as g50det
from experiments.followup52 import run as g52
from experiments.followup56.execute import Encoder as GDM56Encoder
from experiments.followup57 import run as g57

SOURCE50 = "62d2720800f5d98e1521e0200766a80e998ebe9b"
SOURCE51 = "f52baf82222947ca00139437e90fe9b5428bd9ae"
SOURCE54 = "031c7f6087eb356f8baeeedcb2bf25ce9fe35cb4"
SOURCE55 = "db13313e7fd5021922768add76aaac617ec21208"
SOURCE56 = "168317f6495b05e04597ae084b75be6890fe0621"


class Encoder(GDM56Encoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        head = dict(self.meta["head_training_reproducibility"])
        head["gdm57_execution_scope"] = "capability_and_ambiguity_heads"
        head["canonical_gdm50_numerical_source_commit"] = SOURCE50
        head["canonical_gdm51_architecture_source_commit"] = SOURCE51
        head["canonical_gdm54_source_commit"] = SOURCE54
        head["canonical_gdm55_source_commit"] = SOURCE55
        head["canonical_gdm56_source_commit"] = SOURCE56
        head["gdm57_change_scope"] = (
            "ambiguity-head top-two semantic input representation only; fixed canonical GDM56 family-balanced curriculum, numerical path and GDM54 5pp rescue arbitration"
        )
        self.meta = dict(self.meta)
        self.meta["head_training_reproducibility"] = head


def main():
    g50exec.configure_reproducibility()
    old_adamw = torch.optim.AdamW
    old_clip = torch.nn.utils.clip_grad_norm_
    old_encoder = g52.Encoder
    old_threads = torch.set_num_threads
    old_interop = torch.set_num_interop_threads
    torch.optim.AdamW = g50det.HighPrecisionParameterApplyAdamW
    torch.nn.utils.clip_grad_norm_ = g50det.single_tensor_clip_grad_norm_
    g52.Encoder = Encoder
    torch.set_num_threads = lambda _requested: old_threads(1)

    def deterministic_interop(_requested):
        if torch.get_num_interop_threads() != 1:
            raise RuntimeError("GDM57 reproducibility contract requires one inter-op thread")
        return None

    torch.set_num_interop_threads = deterministic_interop
    try:
        g57.main()
    finally:
        torch.optim.AdamW = old_adamw
        torch.nn.utils.clip_grad_norm_ = old_clip
        g52.Encoder = old_encoder
        torch.set_num_threads = old_threads
        torch.set_num_interop_threads = old_interop


if __name__ == "__main__":
    main()
