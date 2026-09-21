"""Canonical GDM50 entrypoint with corrected online timing semantics.

GDM50's original explicit-NONE path called ``features_with_none(..., online=True)``
for timing.  The implementation used ``cached=not online`` for both the request
and the catalog option embeddings, so timed requests re-encoded every catalog
option (and the fixed NONE text) even though the recorded latency scope claimed
cached catalog vectors.  That made the explicit-NONE latency incomparable with
the listwise control and violated experiments/MEASUREMENT_POLICY.md.

This entrypoint changes only that cache boundary: online requests always freshly
encode the request clause, while catalog/NONE embeddings remain cached.  Model
training, checkpoints, predictions, GraphQL execution, Rover composition and all
correctness metrics continue to use the existing measured GDM50 implementation.
"""
from __future__ import annotations

import torch

from experiments.measured.contracts import option_text
from experiments.followup50 import run as g50


def features_with_none_cached(encoder, item, clause, online=False):
    """Build GDM50 features with fresh online query encoding and cached catalog.

    ``online=True`` means only the request clause is freshly encoded.  Catalog
    capability vectors and the fixed NONE vector are immutable for a catalog and
    therefore belong to the cached-index side of the latency contract.
    """
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
    x = torch.cat(
        [
            qx,
            c,
            qx * c,
            (qx - c).abs(),
            centered,
            torch.tensor(state, dtype=torch.float32),
        ],
        dim=-1,
    )
    opts = real + [{
        "id": g50.NONE_ID,
        "path": [],
        "coordinates": [],
        "types": [],
        "gloss": g50.NONE_TEXT,
    }]
    return x, opts, c @ q


def main():
    # GDM50 helpers resolve ``features_with_none`` from their module globals at
    # call time, so this repairs training/evaluation/timing consistently without
    # rewriting historical source.  Only the cache policy differs from run 1.
    g50.features_with_none = features_with_none_cached
    g50.main()


if __name__ == "__main__":
    main()
