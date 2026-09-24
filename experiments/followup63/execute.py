"""Canonical deterministic execution wrapper for operation-only GDM63."""
from __future__ import annotations

import torch

from experiments.followup50 import execute as g50exec
from experiments.followup50 import execute_deterministic as g50det
from experiments.followup52 import run as g52
from experiments.followup56.execute import Encoder as GDM56Encoder
from experiments.followup63 import run as g63

SOURCE50 = "62d2720800f5d98e1521e0200766a80e998ebe9b"
SOURCE51 = "f52baf82222947ca00139437e90fe9b5428bd9ae"
SOURCE54 = "031c7f6087eb356f8baeeedcb2bf25ce9fe35cb4"
SOURCE56 = "168317f6495b05e04597ae084b75be6890fe0621"
SOURCE59 = "035f44836d850b24f4bebe5cd499a572b6052752"
SOURCE60 = "c04784ec33983829f6fe55506bcb74e0f40442b2"
SOURCE61 = "f22fc36caf7327ed3d0d1f7dd2898ea31d837f3a"
SOURCE62 = "a3cfa70a570f931e2c78553ca9d72aeb5ed65e79"
AUDIT62 = 35965921793
PROMOTION62 = "7d925c87f80ecb69bf2ea38bfac9960faf08f9cd"

CHANGE_SCOPE = (
    "operation-only small learned rank-preserving cross-candidate ambiguity interaction over fixed canonical q*ci+abs(q-ci) top5 slots; matched architecture/parameter count across all five arms; fixed capability training, GDM56 family-balanced curriculum, canonical numerical path and GDM54 5pp rescue arbitration"
)


class Encoder(GDM56Encoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        head = dict(self.meta["head_training_reproducibility"])
        head["gdm63_execution_scope"] = "operation_capability_and_ambiguity_heads_only"
        head["schema_generation_scope"] = "suspended-frozen"
        head["canonical_gdm50_numerical_source_commit"] = SOURCE50
        head["canonical_gdm51_architecture_source_commit"] = SOURCE51
        head["canonical_gdm54_source_commit"] = SOURCE54
        head["canonical_gdm56_source_commit"] = SOURCE56
        head["canonical_gdm59_source_commit"] = SOURCE59
        head["canonical_gdm60_source_commit"] = SOURCE60
        head["canonical_gdm61_source_commit"] = SOURCE61
        head["canonical_gdm62_source_commit"] = SOURCE62
        head["canonical_gdm62_audit_run"] = AUDIT62
        head["canonical_gdm62_promotion_commit"] = PROMOTION62
        head["gdm63_change_scope"] = CHANGE_SCOPE
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
            raise RuntimeError("GDM63 reproducibility contract requires one inter-op thread")
        return None

    torch.set_num_interop_threads = deterministic_interop
    try:
        g63.main()
    finally:
        torch.optim.AdamW = old_adamw
        torch.nn.utils.clip_grad_norm_ = old_clip
        g52.Encoder = old_encoder
        torch.set_num_threads = old_threads
        torch.set_num_interop_threads = old_interop


if __name__ == "__main__":
    main()
