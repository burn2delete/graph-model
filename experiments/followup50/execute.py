"""Canonical GDM50 entrypoint with corrected online timing and deterministic CPU training.

GDM50's original explicit-NONE path called ``features_with_none(..., online=True)``
for timing. The implementation used ``cached=not online`` for both the request and
catalog option embeddings, so timed requests re-encoded every catalog option even
though the recorded latency scope claimed cached catalog vectors.

The first cache-corrected rerun then exposed a second measurement problem: identical
seeds, model revision, packages, dataset hashes and initial checkpoint hashes could
produce materially different selected checkpoints on the two-thread CPU path. Tiny
floating-point differences accumulated during training and changed checkpoint and
threshold selection. This canonical entrypoint therefore fixes the cache boundary and
forces single-thread deterministic PyTorch execution before delegating to GDM50.
"""
from __future__ import annotations

import torch

from experiments.measured.contracts import option_text
from experiments.followup50 import run as g50


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


def main():
    g50.features_with_none = features_with_none_cached
    torch.use_deterministic_algorithms(True)

    # ``g50.main`` historically requests two intra-op threads. Multithreaded CPU
    # reductions were the only remaining changing execution condition between
    # identical seeded runs. Preserve the public entrypoint while forcing one
    # intra-op thread for this repaired execution path.
    original_set_num_threads = torch.set_num_threads
    def deterministic_set_num_threads(_requested):
        original_set_num_threads(1)
    torch.set_num_threads = deterministic_set_num_threads
    try:
        g50.main()
    finally:
        torch.set_num_threads = original_set_num_threads


if __name__ == "__main__":
    main()
