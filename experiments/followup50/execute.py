"""Canonical GDM50 entrypoint with corrected timing and cross-run CPU reproducibility.

GDM50's original explicit-NONE path called ``features_with_none(..., online=True)``
for timing. The implementation used ``cached=not online`` for both the request and
catalog option embeddings, so timed requests re-encoded every catalog option even
though the recorded latency scope claimed cached catalog vectors.

The first cache-corrected rerun exposed a second measurement problem: identical
seeds, model revision, packages, dataset hashes and initial checkpoint hashes could
produce materially different selected checkpoints on the multi-thread CPU path.
Forcing one PyTorch intra-op thread helped, but exact same-seed Actions reruns still
showed hosted-runner frozen-encoder numerical drift that could amplify through
feature-head training, checkpoint selection and validation calibration.

This repair keeps the transformer frozen and does not change the downstream model
family. It makes the feature boundary explicit and reproducible: normalized frozen
encoder outputs are projected onto a fixed 1e-4 decimal grid before they are cached,
used for training, calibration, evaluation or timed inference. That controlled
feature canonicalization is part of the measured architecture and is recorded in
every result artifact together with raw and canonical fixed-probe hashes. The grid
is intentionally much coarser than the previously observed ~4.77e-7 cross-run drift
while remaining small relative to normalized embedding magnitudes.

The wrapper also disables oneDNN, fixes PyTorch intra-op/inter-op execution to one
thread and records the workflow's OpenMP/MKL controls. Historical thread setters are
masked while ``g50.main`` runs so they cannot mutate the established contract.
"""
from __future__ import annotations

import hashlib

import torch

from experiments.measured.contracts import option_text
from experiments.followup50 import run as g50


FEATURE_CANONICALIZATION_QUANTUM = 1e-4
FEATURE_PROBE_TEXTS = (
    "GraphQL capability reproducibility probe: author identifier and author name.",
    "GraphQL capability reproducibility probe: created timestamp and updated timestamp.",
)


def canonicalize_embeddings(matrix: torch.Tensor) -> torch.Tensor:
    """Project frozen normalized embeddings onto the recorded fixed decimal grid.

    Hosted CPU kernels can differ by a few float32 ulps even with deterministic
    algorithms and a single thread. Those differences are irrelevant to the frozen
    encoder's intended semantic representation but were large enough to change
    downstream checkpoint selection. Canonicalization occurs exactly once at the
    frozen-feature boundary and is therefore an explicit measured model contract,
    not a post-hoc metric repair.
    """
    quantum = FEATURE_CANONICALIZATION_QUANTUM
    return torch.round(matrix / quantum) * quantum


def tensor_sha256(matrix: torch.Tensor) -> str:
    data = matrix.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


class ReproducibleEncoder(g50.Encoder):
    """Frozen encoder with explicit CPU and feature reproducibility metadata."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta = dict(self.meta)
        self.meta["cpu_reproducibility"] = {
            "intra_op_threads": 1,
            "interop_threads": 1,
            "mkldnn_enabled": False,
            "mkl_cbwr": "COMPATIBLE",
            "pythonhashseed": "0",
        }
        self.meta["feature_canonicalization"] = {
            "kind": "post-normalization-fixed-decimal-grid",
            "quantum": FEATURE_CANONICALIZATION_QUANTUM,
            "applies_to": "all frozen encoder outputs before cache/training/calibration/evaluation/timing",
            "architecture_change": True,
            "reason": "absorb hosted-CPU frozen-encoder numerical drift before learned feature-head training",
        }

        # Preserve both hashes so the next exact rerun can distinguish remaining
        # frozen-encoder drift from failure of the canonical feature boundary.
        raw_probe = super().encode(FEATURE_PROBE_TEXTS, cached=False)
        canonical_probe = canonicalize_embeddings(raw_probe)
        self.meta["reproducibility_probe"] = {
            "texts_sha256": hashlib.sha256("\n".join(FEATURE_PROBE_TEXTS).encode()).hexdigest(),
            "raw_feature_sha256": tensor_sha256(raw_probe),
            "canonical_feature_sha256": tensor_sha256(canonical_probe),
            "rows": len(FEATURE_PROBE_TEXTS),
            "dim": int(canonical_probe.shape[-1]),
        }

    def encode(self, texts, cached=True, batch_size=8):
        # Base Encoder normalizes embeddings before returning them. Canonicalize that
        # normalized feature representation, then replace any newly populated cache
        # entries so every subsequent path observes the same canonical feature.
        texts = list(texts)
        matrix = super().encode(texts, cached=cached, batch_size=batch_size)
        matrix = canonicalize_embeddings(matrix)
        if cached:
            for text, vector in zip(texts, matrix):
                self.cache[text] = vector.detach()
        if not torch.isfinite(matrix).all():
            raise FloatingPointError("Nonfinite canonical frozen embedding")
        return matrix


def features_with_none_cached(encoder, item, clause, online=False):
    """Freshly encode only the online request clause; keep catalog/NONE cached."""
    real = item["catalog"]["options"]
    q = encoder.query(clause, cached=not online)
    texts = [option_text(o, True) for o in real] + [g50.NONE_TEXT]
    c = encoder.encode(texts, cached=True)
    qx = q.expand_as(c)
    centered = c - c.mean(0, keepdim=True)
    state = []
    for o in real:
        state.append([
            len(o["path"]) / 5.0,
            float(item["task"] == "schema"),
            len(o.get("coordinates", [])) / 5.0,
            float(o["path"][-1] in {"name", "id"}),
        ])
    state.append([0.0, float(item["task"] == "schema"), 0.0, 0.0])
    x = torch.cat([
        qx,
        c,
        qx * c,
        (qx - c).abs(),
        centered,
        torch.tensor(state, dtype=torch.float32),
    ], dim=-1)
    opts = real + [{
        "id": g50.NONE_ID,
        "path": [],
        "coordinates": [],
        "types": [],
        "gloss": g50.NONE_TEXT,
    }]
    return x, opts, c @ q


def configure_reproducibility():
    """Configure deterministic CPU execution before any encoder/model work."""
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Import-side/unit-test callers may have already started parallel work.
        # The GitHub Actions entrypoint invokes this before model construction.
        if torch.get_num_interop_threads() != 1:
            raise


def main():
    configure_reproducibility()
    g50.features_with_none = features_with_none_cached
    g50.Encoder = ReproducibleEncoder

    # ``g50.main`` historically requests two intra-op threads and also repeats the
    # inter-op setter. The reproducibility contract has already been established
    # above. Mask both historical setters during the call so PyTorch is not asked to
    # mutate inter-op state after parallel work has begun.
    original_set_num_threads = torch.set_num_threads
    original_set_num_interop_threads = torch.set_num_interop_threads

    def deterministic_set_num_threads(_requested):
        original_set_num_threads(1)

    def deterministic_set_num_interop_threads(_requested):
        if torch.get_num_interop_threads() != 1:
            raise RuntimeError("GDM50 reproducibility contract requires one inter-op thread")
        return None

    torch.set_num_threads = deterministic_set_num_threads
    torch.set_num_interop_threads = deterministic_set_num_interop_threads
    try:
        g50.main()
    finally:
        torch.set_num_threads = original_set_num_threads
        torch.set_num_interop_threads = original_set_num_interop_threads


if __name__ == "__main__":
    main()
