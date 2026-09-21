"""Canonical GDM50 entrypoint with corrected timing and cross-run CPU reproducibility.

GDM50's original explicit-NONE path called ``features_with_none(..., online=True)``
for timing. The implementation used ``cached=not online`` for both the request and
catalog option embeddings, so timed requests re-encoded every catalog option even
though the recorded latency scope claimed cached catalog vectors.

The cache-corrected reruns exposed a second measurement problem: identical seeds,
model revision, packages, dataset hashes and initial checkpoint hashes could produce
materially different selected checkpoints across GitHub-hosted CPU runners. One-thread
execution, deterministic PyTorch and disabled oneDNN were insufficient. A 1e-4
post-normalization feature grid made the fixed probe reproducible, but an exact rerun
still changed every downstream GDM50 configuration, proving that a two-text probe was
not sufficient to establish the whole frozen-feature boundary.

This repair keeps the transformer frozen and preserves the downstream model family.
It adds two controls without silently changing the existing 1e-4 feature quantum:
PyTorch ATen CPU dispatch is pinned to its oldest supported/default vector codepath,
and every first-observed frozen feature is included in a deterministic text-sorted
raw/canonical corpus hash. PyTorch 2.8 reports that x86 default dispatch as ``NO AVX``;
we normalize that reporting label to the logical ``DEFAULT`` contract while preserving
the raw runtime label in evidence. The corpus evidence distinguishes CPU-kernel drift
from a failure of the recorded feature canonicalization over the actual experiment texts.

The wrapper also disables oneDNN, fixes PyTorch intra-op/inter-op execution to one
thread and records OpenMP/MKL/ATen controls. Historical thread setters are masked
while ``g50.main`` runs so they cannot mutate the established contract.
"""
from __future__ import annotations

import hashlib
import os

import torch

from experiments.measured.contracts import option_text
from experiments.followup50 import run as g50


FEATURE_CANONICALIZATION_QUANTUM = 1e-4
FEATURE_PROBE_TEXTS = (
    "GraphQL capability reproducibility probe: author identifier and author name.",
    "GraphQL capability reproducibility probe: created timestamp and updated timestamp.",
)


def canonicalize_embeddings(matrix: torch.Tensor) -> torch.Tensor:
    """Project frozen normalized embeddings onto the recorded fixed decimal grid."""
    quantum = FEATURE_CANONICALIZATION_QUANTUM
    return torch.round(matrix / quantum) * quantum


def tensor_sha256(matrix: torch.Tensor) -> str:
    data = matrix.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def feature_corpus_sha256(per_text_hashes: dict[str, str]) -> str:
    """Hash first-observed text/vector hashes independent of encounter order."""
    h = hashlib.sha256()
    for text in sorted(per_text_hashes):
        encoded = text.encode("utf-8")
        h.update(len(encoded).to_bytes(8, "big"))
        h.update(encoded)
        h.update(bytes.fromhex(per_text_hashes[text]))
    return h.hexdigest()


class ReproducibleEncoder(g50.Encoder):
    """Frozen encoder with explicit CPU and feature reproducibility evidence."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta = dict(self.meta)
        runtime_capability_reported = torch.backends.cpu.get_cpu_capability()
        requested_capability = os.environ.get("ATEN_CPU_CAPABILITY")
        if requested_capability != "default":
            raise RuntimeError(
                "GDM50 reproducibility contract requires ATEN_CPU_CAPABILITY=default"
            )
        # On the pinned PyTorch 2.8 x86 CPU wheel, ATEN_CPU_CAPABILITY=default
        # selects the oldest/non-AVX dispatch path and get_cpu_capability() reports
        # that path as "NO AVX" rather than the logical environment label "DEFAULT".
        # Accept only those two equivalent reporting labels and preserve the raw one.
        if runtime_capability_reported not in {"DEFAULT", "NO AVX"}:
            raise RuntimeError(
                "ATEN default CPU dispatch was requested but runtime reported "
                + repr(runtime_capability_reported)
            )
        runtime_capability = "DEFAULT"
        self.meta["cpu_reproducibility"] = {
            "intra_op_threads": 1,
            "interop_threads": 1,
            "mkldnn_enabled": False,
            "mkl_cbwr": "COMPATIBLE",
            "pythonhashseed": "0",
            "omp_num_threads": "1",
            "mkl_num_threads": "1",
            "omp_dynamic": "FALSE",
            "mkl_dynamic": "FALSE",
            "aten_cpu_capability_env": requested_capability,
            "aten_cpu_capability_runtime": runtime_capability,
            "aten_cpu_capability_runtime_reported": runtime_capability_reported,
            "aten_cpu_capability_contract": "oldest-supported/default non-AVX dispatch",
        }
        self.meta["feature_canonicalization"] = {
            "kind": "post-normalization-fixed-decimal-grid",
            "quantum": FEATURE_CANONICALIZATION_QUANTUM,
            "applies_to": "all frozen encoder outputs before cache/training/calibration/evaluation/timing",
            "architecture_change": True,
            "reason": "absorb hosted-CPU frozen-encoder numerical drift before learned feature-head training",
        }

        raw_probe = super().encode(FEATURE_PROBE_TEXTS, cached=False)
        canonical_probe = canonicalize_embeddings(raw_probe)
        self.meta["reproducibility_probe"] = {
            "texts_sha256": hashlib.sha256("\n".join(FEATURE_PROBE_TEXTS).encode()).hexdigest(),
            "raw_feature_sha256": tensor_sha256(raw_probe),
            "canonical_feature_sha256": tensor_sha256(canonical_probe),
            "rows": len(FEATURE_PROBE_TEXTS),
            "dim": int(canonical_probe.shape[-1]),
        }
        self._raw_text_feature_hashes: dict[str, str] = {}
        self._canonical_text_feature_hashes: dict[str, str] = {}
        self._refresh_feature_corpus_meta()

    def _refresh_feature_corpus_meta(self) -> None:
        self.meta["feature_corpus"] = {
            "scope": "first observed frozen encoding for every unique experiment text, sorted by text before aggregate hashing",
            "unique_texts": len(self._canonical_text_feature_hashes),
            "raw_text_feature_sha256": feature_corpus_sha256(self._raw_text_feature_hashes),
            "canonical_text_feature_sha256": feature_corpus_sha256(self._canonical_text_feature_hashes),
        }

    def _record_feature(self, text: str, raw: torch.Tensor, canonical: torch.Tensor) -> None:
        raw_hash = tensor_sha256(raw.reshape(1, -1))
        canonical_hash = tensor_sha256(canonical.reshape(1, -1))
        previous = self._canonical_text_feature_hashes.get(text)
        if previous is not None and previous != canonical_hash:
            raise RuntimeError(
                "Canonical frozen feature changed within one Actions worker for text "
                + hashlib.sha256(text.encode()).hexdigest()
            )
        self._raw_text_feature_hashes.setdefault(text, raw_hash)
        self._canonical_text_feature_hashes.setdefault(text, canonical_hash)
        self._refresh_feature_corpus_meta()

    def encode(self, texts, cached=True, batch_size=8):
        # Base Encoder normalizes embeddings before returning them. Record every
        # genuinely fresh feature, canonicalize the normalized representation, then
        # replace newly populated cache entries so all downstream paths observe the
        # measured canonical feature boundary.
        texts = list(texts)
        fresh_indices = [
            i for i, text in enumerate(texts)
            if (not cached) or text not in self.cache
        ]
        raw_matrix = super().encode(texts, cached=cached, batch_size=batch_size)
        matrix = canonicalize_embeddings(raw_matrix)
        for i in fresh_indices:
            self._record_feature(texts[i], raw_matrix[i], matrix[i])
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
        if torch.get_num_interop_threads() != 1:
            raise


def main():
    configure_reproducibility()
    g50.features_with_none = features_with_none_cached
    g50.Encoder = ReproducibleEncoder

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
