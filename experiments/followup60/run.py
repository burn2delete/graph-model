"""GDM60: confidence-scaled rank-preserving request/candidate ambiguity evidence.

Canonical GDM59 established that preserving top-five candidate rank through request-
relative semantic interactions improves answerable correctness and target recall versus
a matched-capacity pooled top-five control. The strongest representation still pays a
modest publication-safety cost, leaving a narrower question: can confidence weighting
attenuate lower-ranked candidate noise while preserving the rank-specific request signal?

GDM60 changes only the ambiguity semantic representation within the canonical GDM59
TOP5 family. Every arm uses the same five ranked real candidates, the same 11+10*d
ambiguity input dimension and the same ambiguity-head parameter count. The canonical
GDM59 ranked-request representation is the within-batch control. Diagnostic factorized
arms isolate product and delta evidence; weighted arms scale each rank-specific request
interaction by capability probability or by probability relative to rank one.

Capability training, explicit NONE, exact canonical GDM56 family-balanced ambiguity
curriculum, frozen DistilBERT, the 1e-4 canonical feature boundary, canonical GDM50
numerical execution, GDM54 5pp ambiguity-rescue arbitration, and deterministic
GraphQL/Federation realization remain fixed. Pass/Logbook are calibration-only.
Certificate/Manuscript are the fresh secondary holdout and never participate in
optimizer, checkpoint, calibration or representation selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.followup50 import run as g50
from experiments.followup51 import run as g51
from experiments.followup52 import run as g52
from experiments.followup53 import run as g53
from experiments.followup54 import run as g54
from experiments.followup55 import run as g55
from experiments.followup56 import run as g56
from experiments.followup57 import run as g57
from experiments.followup58 import run as g58
from experiments.followup59 import run as g59

FORMAT = "gdm60-measured-v1"
SEEDS_DEFAULT = "6001,6002"
CANONICAL_GDM59_SOURCE = "035f44836d850b24f4bebe5cd499a572b6052752"
CANONICAL_GDM59_AUDIT_RUN = 35910892558

_BASE = {
    "family": "explicit-none-plus-ambiguity",
    "head": "mlp",
    "ambiguous_weight": 1.0,
    "hard_negative": False,
    "calibration_strategy": "ambiguity-rescue",
    "status_arbitration": "ambiguity-rescue",
    "nomatch_recall_budget_pp": 5,
    "ambiguity_curriculum": "family-balanced",
}

ARMS = {
    "ranked-request-control": {**_BASE, "structured": "ranked-request-control"},
    "ranked-product-only": {**_BASE, "structured": "ranked-product-only"},
    "ranked-delta-only": {**_BASE, "structured": "ranked-delta-only"},
    "ranked-confidence-weighted": {**_BASE, "structured": "ranked-confidence-weighted"},
    "ranked-relative-confidence": {**_BASE, "structured": "ranked-relative-confidence"},
}

REPRESENTATION_MODES = tuple(ARMS)
SEMANTIC_BLOCKS = 10
TOPK = 5
_CURRENT_ENCODER_DIM = None

CALIBRATION_LANGUAGE = [
    "reader-visible heading assigned to this item",
    "time this item was first committed to durable storage",
    "time this item was most recently changed in durable storage",
    "whole-number score attached to each review",
    "readable names of people who wrote reviews",
    "readable names of people who moderated reviews",
    "registered company name of the supplier for this item",
    "stable identifiers of people who wrote reviews",
]
CALIBRATION_UNSUPPORTED = [
    "physical records-room drawer assigned to this item",
    "justification for an urgent privileged-access override",
    "cross-border tariff category for this item",
    "flag indicating this item is frozen for evidentiary retention",
]
CALIBRATION_AMBIGUITIES = [
    ("role", "review person without choosing writer or moderator role"),
    ("lifecycle-time", "durable-storage time without choosing original commit or latest change"),
    ("representation", "review writer identity without choosing readable name or stable identifier"),
    ("object-vs-supplier", "name for the item without choosing item heading or supplying company"),
]

HOLDOUT_LANGUAGE = [
    "public-facing caption for this record",
    "timestamp when this record first entered persistent storage",
    "timestamp when this record was last modified in persistent storage",
    "integer rating recorded for every review",
    "display names of people responsible for authoring each review",
    "display names of people responsible for moderating each review",
    "official corporate name of this record's supplier",
    "persistent IDs of people responsible for authoring each review",
]
HOLDOUT_UNSUPPORTED = [
    "offsite storage crate number for this record",
    "narrative explaining an exceptional administrator bypass",
    "import-export commodity code for this record",
    "flag indicating this record is subject to forensic preservation",
]
HOLDOUT_AMBIGUITIES = [
    ("role", "review participant without choosing author or moderator role"),
    ("lifecycle-time", "persistence timestamp without choosing first storage or latest modification"),
    ("representation", "review author identity without choosing display name or persistent ID"),
    ("object-vs-supplier", "record name without choosing public caption or supplier corporation"),
]

_ORIGINAL_TRAIN_MODEL = g56._ORIGINAL_TRAIN_MODEL
_ORIGINAL_RUN_ONE = g52.run_one
_ORIGINAL_CLAUSE_STATE = g51.clause_state


def ambiguity_feature_dim(encoder_dim: int) -> int:
    return 11 + SEMANTIC_BLOCKS * int(encoder_dim)


def _ambiguity_dim(_mode) -> int:
    if _CURRENT_ENCODER_DIM is None:
        raise RuntimeError("GDM60 encoder dimension was not bound before bundle construction")
    return ambiguity_feature_dim(_CURRENT_ENCODER_DIM)


def _pad_ranked(candidates: torch.Tensor, weights: torch.Tensor, topk: int = TOPK):
    if candidates.ndim != 2 or candidates.shape[0] <= 0:
        raise ValueError("candidates must be non-empty [k,d]")
    if weights.ndim != 1 or weights.shape[0] != candidates.shape[0]:
        raise ValueError("weights must match candidate rows")
    k, dim = candidates.shape
    if k > topk:
        candidates = candidates[:topk]
        weights = weights[:topk]
        k = topk
    padded = torch.zeros((topk, dim), dtype=candidates.dtype, device=candidates.device)
    padded[:k] = candidates
    padded_weights = torch.zeros((topk,), dtype=weights.dtype, device=weights.device)
    padded_weights[:k] = weights
    mask = torch.zeros((topk,), dtype=candidates.dtype, device=candidates.device)
    mask[:k] = 1.0
    return padded, padded_weights, mask, k


def _confidence_blocks(q: torch.Tensor, candidates: torch.Tensor, weights: torch.Tensor, mode: str):
    """Return two 5d rank-preserving request-relative blocks at fixed capacity."""
    padded, probs, mask, _ = _pad_ranked(candidates, weights)
    product = padded * q[None, :] * mask[:, None]
    delta = (padded - q[None, :]).abs() * mask[:, None]
    zeros = torch.zeros_like(product)

    if mode == "ranked-request-control":
        first, second = product, delta
    elif mode == "ranked-product-only":
        first, second = product, zeros
    elif mode == "ranked-delta-only":
        first, second = delta, zeros
    elif mode == "ranked-confidence-weighted":
        scale = probs * mask
        first, second = product * scale[:, None], delta * scale[:, None]
    elif mode == "ranked-relative-confidence":
        denom = probs[0].clamp_min(torch.finfo(probs.dtype).tiny)
        scale = (probs / denom).clamp(min=0.0, max=1.0) * mask
        first, second = product * scale[:, None], delta * scale[:, None]
    else:
        raise KeyError(mode)
    return first.reshape(-1), second.reshape(-1)


def clause_state(bundle, encoder, item, clause, structured=False, online=False):
    mode = structured
    if mode not in REPRESENTATION_MODES:
        return _ORIGINAL_CLAUSE_STATE(bundle, encoder, item, clause, structured=structured, online=online)

    x, opts, raw = g50.features_with_none(encoder, item, clause, online=online)
    logits = bundle.capability(x)
    probs = torch.softmax(logits, -1)
    real_logits = logits[:-1]
    order = torch.argsort(real_logits, descending=True, stable=True)
    a = int(order[0])
    b = int(order[1]) if len(order) > 1 else a
    none_i = len(opts) - 1

    dim = encoder.dim
    q = x[a, :dim]
    c1 = x[a, dim : 2 * dim]
    c2 = x[b, dim : 2 * dim]
    cos = float(F.cosine_similarity(c1.unsqueeze(0), c2.unsqueeze(0))[0])
    p1 = opts[a]["path"]
    p2 = opts[b]["path"]
    base = [
        float(logits[none_i] - logits[a]),
        float(logits[a] - logits[b]),
        float(probs[a]),
        float(probs[b]),
        float(probs[none_i]),
        float(raw[a]),
        float(raw[b]),
        cos,
        float(p1[:-1] == p2[:-1]),
        float(len(p1) == len(p2)),
        float((p1[-1] in {"name", "id"}) == (p2[-1] in {"name", "id"})),
    ]

    selected = order[: min(TOPK, len(order))]
    candidate_rows = torch.stack([x[int(i), dim : 2 * dim] for i in selected], dim=0)
    selected_weights = torch.stack([probs[int(i)] for i in selected], dim=0)
    block1, block2 = _confidence_blocks(q, candidate_rows, selected_weights, mode)
    features = base + block1.detach().float().tolist() + block2.detach().float().tolist()
    expected = ambiguity_feature_dim(dim)
    if len(features) != expected:
        raise AssertionError((mode, len(features), expected))

    return {
        "clause": clause,
        "best_real_id": opts[a]["id"],
        "second_real_id": opts[b]["id"],
        "none_margin": float(logits[none_i] - logits[a]),
        "real_margin": float(logits[a] - logits[b]),
        "best_real_prob": float(probs[a]),
        "second_real_prob": float(probs[b]),
        "none_prob": float(probs[none_i]),
        "raw_best": float(raw[a]),
        "raw_second": float(raw[b]),
        "ambiguity_features": features,
        "ambiguity_representation_mode": mode,
        "ambiguity_feature_dim": expected,
        "ambiguity_topk_effective": len(selected),
        "ambiguity_real_candidate_count": len(order),
        "ambiguity_rank_preserving": True,
    }


def ambiguity_curriculum_examples(task, public, refs):
    return list(g56.ambiguity_curriculum_examples(task, public, refs, "family-balanced"))


def _representation_description(mode: str) -> str:
    return {
        "ranked-request-control": "canonical-gdm59 ranked slots q*c1..q*c5 + abs(q-c1)..abs(q-c5)",
        "ranked-product-only": "ranked slots q*c1..q*c5 + zero block",
        "ranked-delta-only": "ranked slots abs(q-c1)..abs(q-c5) + zero block",
        "ranked-confidence-weighted": "ranked q*ci and abs(q-ci) slots scaled by capability probability p_i",
        "ranked-relative-confidence": "ranked q*ci and abs(q-ci) slots scaled by clipped p_i/p_1",
    }[mode]


def _representation_receipt(mode: str, encoder_dim: int):
    return {
        "gdm60_family": "confidence-scaled-ranked-request-ambiguity-evidence",
        "gdm60_representation_mode": mode,
        "gdm60_requested_topk": TOPK,
        "gdm60_ambiguity_feature_dim": ambiguity_feature_dim(encoder_dim),
        "gdm60_semantic_representation": _representation_description(mode),
        "gdm60_rank_preserving": True,
        "gdm60_change_scope": (
            "ambiguity-head rank-preserving request-relative confidence weighting only; fixed top5 breadth, capacity, canonical GDM56 family-balanced curriculum, canonical numerical path and GDM54 5pp rescue arbitration"
        ),
        "canonical_gdm59_source_commit": CANONICAL_GDM59_SOURCE,
        "canonical_gdm59_audit_run": CANONICAL_GDM59_AUDIT_RUN,
    }


def train_model(bundle, encoder, arm, task, public, refs, seed, epochs, directory):
    examples = ambiguity_curriculum_examples(task, public, refs)
    old_examples = g51.ambiguity_training_examples
    g51.ambiguity_training_examples = lambda _task, _public, _refs: list(examples)
    try:
        receipt = _ORIGINAL_TRAIN_MODEL(bundle, encoder, arm, task, public, refs, seed, epochs, directory)
    finally:
        g51.ambiguity_training_examples = old_examples
    receipt.update(g56._curriculum_receipt(examples, "family-balanced"))
    receipt.update(_representation_receipt(arm["structured"], encoder.dim))
    return receipt


def calibration_set(task):
    rows, refs = g50._dataset(
        task,
        [("Pass", "pass"), ("Logbook", "logbook")],
        CALIBRATION_LANGUAGE,
        CALIBRATION_UNSUPPORTED,
        [phrase for _, phrase in CALIBRATION_AMBIGUITIES],
        "gdm60-calibration",
    )
    return g55._tag_ambiguity_families(rows, refs, CALIBRATION_AMBIGUITIES)


def fresh_holdout(task):
    rows, refs = g50._dataset(
        task,
        [("Certificate", "certificate"), ("Manuscript", "manuscript")],
        HOLDOUT_LANGUAGE,
        HOLDOUT_UNSUPPORTED,
        [phrase for _, phrase in HOLDOUT_AMBIGUITIES],
        "gdm60-holdout",
    )
    return g55._tag_ambiguity_families(rows, refs, HOLDOUT_AMBIGUITIES)


def _output_root():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", default="artifacts/results")
    args, _ = parser.parse_known_args()
    return Path(args.output)


def _run_one_with_bound_dim(arm_name, task, seed, encoder, public, refs, data_report, compositor, root, epochs):
    global _CURRENT_ENCODER_DIM
    previous = _CURRENT_ENCODER_DIM
    _CURRENT_ENCODER_DIM = int(encoder.dim)
    try:
        return _ORIGINAL_RUN_ONE(arm_name, task, seed, encoder, public, refs, data_report, compositor, root, epochs)
    finally:
        _CURRENT_ENCODER_DIM = previous


def _postprocess(root):
    for summary_path in root.rglob("summary.json"):
        directory = summary_path.parent
        config_path = directory / "config.json"
        training_path = directory / "training.json"
        config = json.loads(config_path.read_text())
        training = json.loads(training_path.read_text())
        summary = json.loads(summary_path.read_text())
        arm = ARMS[config["arm"]]
        selected = json.loads((directory / "calibration.json").read_text())["selected"]
        mode = arm["structured"]

        config["research_question"] = (
            "can confidence scaling of rank-preserving top-five request interactions improve publication safety without giving back canonical GDM59 answerable gains"
        )
        config["ambiguity_curriculum"] = "family-balanced"
        config["ambiguity_representation_mode"] = mode
        config["ambiguity_feature_dim"] = training["gdm60_ambiguity_feature_dim"]
        config["ambiguity_requested_topk"] = TOPK
        config["ambiguity_rank_preserving"] = True
        training["selected_calibration"] = selected

        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        training_path.write_text(json.dumps(training, indent=2, sort_keys=True) + "\n")
        summary["config"] = config
        summary["training"] = training
        summary["calibration_scope"] = (
            "disjoint Pass/Logbook calibration-only cases; never used for optimizer updates or checkpoint selection"
        )
        summary["secondary_holdout_scope"] = (
            "new GDM60 Certificate/Manuscript synthetic holdout; never used for optimizer/checkpoint/calibration selection or representation design"
        )
        summary["regression_scope"] = "corrected public test plus inspected GDM46-GDM59 holdouts"
        summary["secondary_holdout_ambiguity_family_metrics"] = g55._ambiguity_family_metrics(
            directory / "predictions-holdout.jsonl"
        )
        summary["file_hashes"]["config.json"] = g52.file_hash(config_path)
        summary["file_hashes"]["training.json"] = g52.file_hash(training_path)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main():
    root = _output_root()
    old = {
        "FORMAT": g52.FORMAT,
        "SEEDS_DEFAULT": g52.SEEDS_DEFAULT,
        "ARMS": g52.ARMS,
        "calibration_set": g52.calibration_set,
        "fresh_holdout": g52.fresh_holdout,
        "calibrate_model": g52.calibrate_model,
        "g51_holdout": g52.g51.fresh_holdout,
        "request_from_evidence": g51.request_from_evidence,
        "train_model": g51.train_model,
        "clause_state": g51.clause_state,
        "ambiguity_dim": g51._ambiguity_dim,
        "run_one": g52.run_one,
    }
    old_g51_holdout = g52.g51.fresh_holdout
    old_g52_holdout = g52.fresh_holdout
    old_g53_holdout = g53.fresh_holdout
    old_g54_holdout = g54.fresh_holdout
    old_g55_holdout = g55.fresh_holdout
    old_g56_holdout = g56.fresh_holdout
    old_g57_holdout = g57.fresh_holdout
    old_g58_holdout = g58.fresh_holdout
    old_g59_holdout = g59.fresh_holdout

    def regression_g51_through_g59(task):
        parts = [
            old_g51_holdout(task), old_g52_holdout(task), old_g53_holdout(task),
            old_g54_holdout(task), old_g55_holdout(task), old_g56_holdout(task),
            old_g57_holdout(task), old_g58_holdout(task), old_g59_holdout(task),
        ]
        rows = []
        refs = {}
        for part_rows, part_refs in parts:
            rows.extend(part_rows)
            refs.update(part_refs)
        return rows, refs

    g52.FORMAT = FORMAT
    g52.SEEDS_DEFAULT = SEEDS_DEFAULT
    g52.ARMS = ARMS
    g52.calibration_set = calibration_set
    g52.fresh_holdout = fresh_holdout
    g52.calibrate_model = g54.calibrate_model
    g52.g51.fresh_holdout = regression_g51_through_g59
    g51.request_from_evidence = g54.request_from_evidence
    g51.train_model = train_model
    g51.clause_state = clause_state
    g51._ambiguity_dim = _ambiguity_dim
    g52.run_one = _run_one_with_bound_dim
    try:
        g52.main()
        _postprocess(root)
    finally:
        g52.FORMAT = old["FORMAT"]
        g52.SEEDS_DEFAULT = old["SEEDS_DEFAULT"]
        g52.ARMS = old["ARMS"]
        g52.calibration_set = old["calibration_set"]
        g52.fresh_holdout = old["fresh_holdout"]
        g52.calibrate_model = old["calibrate_model"]
        g52.g51.fresh_holdout = old["g51_holdout"]
        g51.request_from_evidence = old["request_from_evidence"]
        g51.train_model = old["train_model"]
        g51.clause_state = old["clause_state"]
        g51._ambiguity_dim = old["ambiguity_dim"]
        g52.run_one = old["run_one"]


if __name__ == "__main__":
    main()
