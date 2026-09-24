"""Canonical deterministic execution wrapper for operation-only GDM64."""
from __future__ import annotations

import torch

from experiments.followup50 import execute as g50exec
from experiments.followup50 import execute_deterministic as g50det
from experiments.followup52 import run as g52
from experiments.followup56.execute import Encoder as GDM56Encoder
from experiments.followup64 import run as g64

SOURCE50 = "62d2720800f5d98e1521e0200766a80e998ebe9b"
SOURCE51 = "f52baf82222947ca00139437e90fe9b5428bd9ae"
SOURCE54 = "031c7f6087eb356f8baeeedcb2bf25ce9fe35cb4"
SOURCE56 = "168317f6495b05e04597ae084b75be6890fe0621"
SOURCE59 = "035f44836d850b24f4bebe5cd499a572b6052752"
SOURCE62 = "a3cfa70a570f931e2c78553ca9d72aeb5ed65e79"
SOURCE63 = "43a6d16652a3fd63c155f46e99b42c757c7601a1"
AUDIT63 = 35988393672
PROMOTION63 = "4efc41f161fadd59f563f491f251ab6e058fcb71"

CHANGE_SCOPE = (
    "operation-only schema-coordinate semantic factorization inside fixed top5 ranked ambiguity slots; deterministic parent/role and leaf/representation views derived only from provided GraphQL paths; fixed dimensions, ambiguity-head capacity, capability training, GDM56 family-balanced curriculum, canonical numerical path and GDM54 5pp rescue arbitration"
)


class Encoder(GDM56Encoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        head = dict(self.meta["head_training_reproducibility"])
        head["gdm64_execution_scope"] = "operation_capability_and_ambiguity_heads_only"
        head["schema_generation_scope"] = "suspended-frozen"
        head["canonical_gdm50_numerical_source_commit"] = SOURCE50
        head["canonical_gdm51_architecture_source_commit"] = SOURCE51
        head["canonical_gdm54_source_commit"] = SOURCE54
        head["canonical_gdm56_source_commit"] = SOURCE56
        head["canonical_gdm59_source_commit"] = SOURCE59
        head["canonical_gdm62_source_commit"] = SOURCE62
        head["canonical_gdm63_source_commit"] = SOURCE63
        head["canonical_gdm63_audit_run"] = AUDIT63
        head["canonical_gdm63_promotion_commit"] = PROMOTION63
        head["gdm64_change_scope"] = CHANGE_SCOPE
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
            raise RuntimeError("GDM64 reproducibility contract requires one inter-op thread")
        return None

    torch.set_num_interop_threads = deterministic_interop
    try:
        g64.main()
    finally:
        torch.optim.AdamW = old_adamw
        torch.nn.utils.clip_grad_norm_ = old_clip
        g52.Encoder = old_encoder
        torch.set_num_threads = old_threads
        torch.set_num_interop_threads = old_interop


if __name__ == "__main__":
    main()
