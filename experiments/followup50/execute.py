"""Canonical GDM50 entrypoint with corrected timing and cross-run CPU reproducibility.

GDM50's original explicit-NONE path called ``features_with_none(..., online=True)``
for timing. The implementation used ``cached=not online`` for both the request and
catalog option embeddings, so timed requests re-encoded every catalog option even
though the recorded latency scope claimed cached catalog vectors.

The first cache-corrected rerun exposed a second measurement problem: identical
seeds, model revision, packages, dataset hashes and initial checkpoint hashes could
produce materially different selected checkpoints on the multi-thread CPU path.
Forcing one PyTorch intra-op thread made operation runs bit-reproducible, but an
Actions-only same-seed rerun still exposed tiny frozen-encoder differences in schema
runs on different hosted CPU runners. This entrypoint therefore also disables oneDNN
and records the reproducibility controls; the workflow pins OpenMP/MKL to one thread
and requests MKL's compatible code path before PyTorch starts.

The strengthened repair originally configured inter-op threading here and then
called the historical GDM50 entrypoint, which tried to configure inter-op threading
a second time. PyTorch rejects that second call once parallel work has started. The
canonical wrapper now treats the first configuration as authoritative and masks both
historical thread setters while ``g50.main`` runs, preserving the one-thread contract
without changing the model or experiment semantics.
"""
from __future__ import annotations

import torch

from experiments.measured.contracts import option_text
from experiments.followup50 import run as g50


class ReproducibleEncoder(g50.Encoder):
    """Frozen encoder with explicit CPU reproducibility metadata.

    Numerical behavior is controlled before model construction by ``main`` and the
    workflow environment. This subclass does not alter embeddings; it makes the
    execution contract visible in every result artifact.
    """

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
