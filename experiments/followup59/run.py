"""GDM59: rank-preserving top-k semantic ambiguity representation.

Canonical GDM58 showed that simply widening a fixed permutation-invariant pooled
candidate summary from top two to broader top-k evidence does not improve operation
answerable metrics and does not improve the schema answerable/ambiguity tradeoff.
That negative/inconclusive result leaves a narrower representation hypothesis: the
extra candidates may be useful only if the ambiguity head can preserve which candidate
occupied which rank instead of collapsing candidates into pooled statistics.

GDM59 changes only the ambiguity semantic representation. Every arm uses the same top
five ranked real candidates, the same 11+10*d ambiguity input dimension and therefore
the same ambiguity-head parameter count. The within-batch control is the canonical
GDM58 top-five pooled representation. The other arms preserve candidate rank in five
explicit semantic slots and vary only which request-conditioned relation occupies the
second five slots.

Capability training, explicit NONE, exact canonical GDM56 family-balanced ambiguity
curriculum, frozen DistilBERT, the 1e-4 canonical feature boundary, canonical GDM50
numerical execution, GDM54 5pp ambiguity-rescue arbitration, and deterministic
GraphQL/Federation realization remain fixed. Permit/Chronicle are calibration-only.
Warranty/Dossier are the fresh secondary holdout and never participate in optimizer,
checkpoint, calibration or representation selection.
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

FORMAT = "gdm59-measured-v1"
SEEDS_DEFAULT = "5901,5902"
CANONICAL_GDM58_SOURCE = "9ba9caab957c1802feb17776195141f630fd96eb"
CANONICAL_GDM58_AUDIT_RUN = 35890238431

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
    "top5-pooled-control": {**_BASE, "structured": "top5-pooled-control"},
    "ranked-candidate-only": {**_BASE, "structured": "ranked-candidate-only"},
    "ranked-request-only": {**_BASE, "structured": "ranked-request-only"},
    "ranked-candidate-request": {**_BASE, "structured": "ranked-candidate-request"},
    "ranked-candidate-delta": {**_BASE, "structured": "ranked-candidate-delta"},
}

REPRESENTATION_MODES = tuple(ARMS)
SEMANTIC_BLOCKS = 10
TOPK = 5
_CURRENT_ENCODER_DIM = None

CALIBRATION_LANGUAGE = [
    "public label displayed for this record",
    "instant this record first entered durable storage",
    "instant this record was most recently revised in durable storage",
    "integer score associated with every review",
    "display names of people who authored reviews",
    "display names of people who moderated reviews",
    "registered business name of the supplying organization",
    "persistent identifiers of review authors",
]
CALIBRATION_UNSUPPORTED = [
    "physical archive shelf coordinate assigned to this record",
    "reason a temporary privileged-access waiver was approved",
    "customs declaration category assigned to this record",
    "flag indicating this record is under evidence preservation",
]
CALIBRATION_AMBIGUITIES = [
    ("role", "review-associated person without choosing author versus moderator role"),
    ("lifecycle-time", "durable-storage instant without choosing first write versus latest revision"),
    ("representation", "review author identity without choosing display name versus persistent identifier"),
    ("object-vs-supplier", "display name without choosing record label versus supplier organization"),
]

HOLDOUT_LANGUAGE = [
    "reader-facing title for this entry",
    "moment this entry first reached durable storage",
    "moment this entry was most recently rewritten in durable storage",
    "whole-number rating attached to each review",
    "human-readable names of people who authored every review",
    "human-readable names of people who moderated every review",
    "legal company name for this entry's supplier",
    "stable identifiers for people who authored every review",
]
HOLDOUT_UNSUPPORTED = [
    "warehouse aisle and bin assigned to this entry",
    "explanation for an emergency privileged-access exception",
    "international commodity classification for this entry",
    "flag saying this entry is retained under litigation hold",
]
HOLDOUT_AMBIGUITIES = [
    ("role", "person attached to a review without choosing author or moderator role"),
    ("lifecycle-time", "storage time without choosing original persistence or most recent rewrite"),
    ("representation", "review author identity without choosing readable name or stable identifier"),
    ("object-vs-supplier", "name associated with the entry without choosing entry title or supplier company"),
]

_ORIGINAL_TRAIN_MODEL = g56._ORIGINAL_TRAIN_MODEL
_ORIGINAL_RUN_ONE = g52.run_one
_ORIGINAL_CLAUSE_STATE = g51.clause_state


def ambiguity_feature_dim(encoder_dim: int) -> int:
    """All GDM59 arms keep the canonical matched-capacity ambiguity head."""
    return 11 + SEMANTIC_BLOCKS * int(encoder_dim)


def _ambiguity_dim(_mode) -> int:
    if _CURRENT_ENCODER_DIM is None:
        raise RuntimeError("GDM59 encoder dimension was not bound before bundle construction")
    return ambiguity_feature_dim(_CURRENT_ENCODER_DIM)


def _pad_ranked_candidates(candidates: torch.Tensor, topk: int = TOPK):
    if candidates.ndim != 2 or candidates.shape[0] <= 0:
        raise ValueError("candidates must be non-empty [k,d]")
    k, dim = candidates.shape
    if k > topk:
        candidates = candidates[:topk]
        k = topk
    padded = torch.zeros((topk, dim), dtype=candidates.dtype, device=candidates.device)
    padded[:k] = candidates
    mask = torch.zeros((topk,), dtype=candidates.dtype, device=candidates.device)
    mask[:k] = 1.0
    return padded, mask, k


def _ranked_blocks(q: torch.Tensor, candidates: torch.Tensor, mode: str):
    """Return two 5d rank-preserving blocks with zero-safe padding."""
    padded, mask, _ = _pad_ranked_candidates(candidates)
    candidate_slots = padded.reshape(-1)
    product_slots = (padded * q[None, :] * mask[:, None]).reshape(-1)
    delta_slots = ((padded - q[None, :]).abs() * mask[:, None]).reshape(-1)
    zeros = torch.zeros_like(candidate_slots)

    if mode == "ranked-candidate-only":
        return candidate_slots, zeros
    if mode == "ranked-request-only":
        return product_slots, delta_slots
    if mode == "ranked-candidate-request":
        return candidate_slots, product_slots
    if mode == "ranked-candidate-delta":
        return candidate_slots, delta_slots
    raise KeyError(mode)


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

    if mode == "top5-pooled-control":
        block1, block2 = g58._pooled_blocks(q, candidate_rows, selected_weights)
        rank_preserving = False
    else:
        block1, block2 = _ranked_blocks(q, candidate_rows, mode)
        rank_preserving = True

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
        "ambiguity_rank_preserving": rank_preserving,
    }


def ambiguity_curriculum_examples(task, public, refs):
    return list(g56.ambiguity_curriculum_examples(task, public, refs, "family-balanced"))


def _representation_description(mode: str) -> str:
    return {
        "top5-pooled-control": "gdm58-top5 pooled candidate(mean,weighted-mean,max,min,std)+request(q,q*weighted-mean,abs-q-weighted-mean,mean-absdiff,max-absdiff)",
        "ranked-candidate-only": "ranked slots c1..c5 + zero block",
        "ranked-request-only": "ranked slots q*c1..q*c5 + abs(q-c1)..abs(q-c5)",
        "ranked-candidate-request": "ranked slots c1..c5 + q*c1..q*c5",
        "ranked-candidate-delta": "ranked slots c1..c5 + abs(q-c1)..abs(q-c5)",
    }[mode]


def _representation_receipt(mode: str, encoder_dim: int):
    return {
        "gdm59_family": "rank-preserving-top5-semantic-ambiguity-evidence",
        "gdm59_representation_mode": mode,
        "gdm59_requested_topk": TOPK,
        "gdm59_ambiguity_feature_dim": ambiguity_feature_dim(encoder_dim),
        "gdm59_semantic_representation": _representation_description(mode),
        "gdm59_rank_preserving": mode != "top5-pooled-control",
        "gdm59_change_scope": (
            "ambiguity-head top5 semantic representation structure only; fixed candidate breadth, capacity, canonical GDM56 family-balanced curriculum, canonical numerical path and GDM54 5pp rescue arbitration"
        ),
        "canonical_gdm58_source_commit": CANONICAL_GDM58_SOURCE,
        "canonical_gdm58_audit_run": CANONICAL_GDM58_AUDIT_RUN,
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
        [("Permit", "permit"), ("Chronicle", "chronicle")],
        CALIBRATION_LANGUAGE,
        CALIBRATION_UNSUPPORTED,
        [phrase for _, phrase in CALIBRATION_AMBIGUITIES],
        "gdm59-calibration",
    )
    return g55._tag_ambiguity_families(rows, refs, CALIBRATION_AMBIGUITIES)


def fresh_holdout(task):
    rows, refs = g50._dataset(
        task,
        [("Warranty", "warranty"), ("Dossier", "dossier")],
        HOLDOUT_LANGUAGE,
        HOLDOUT_UNSUPPORTED,
        [phrase for _, phrase in HOLDOUT_AMBIGUITIES],
        "gdm59-holdout",
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
            "does preserving top-five candidate identity/rank improve ambiguity discrimination versus the same-capacity GDM58 top-five pooled control"
        )
        config["ambiguity_curriculum"] = "family-balanced"
        config["ambiguity_representation_mode"] = mode
        config["ambiguity_feature_dim"] = training["gdm59_ambiguity_feature_dim"]
        config["ambiguity_requested_topk"] = TOPK
        config["ambiguity_rank_preserving"] = training["gdm59_rank_preserving"]
        training["selected_calibration"] = selected

        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        training_path.write_text(json.dumps(training, indent=2, sort_keys=True) + "\n")
        summary["config"] = config
        summary["training"] = training
        summary["calibration_scope"] = (
            "disjoint Permit/Chronicle calibration-only cases; never used for optimizer updates or checkpoint selection"
        )
        summary["secondary_holdout_scope"] = (
            "new GDM59 Warranty/Dossier synthetic holdout; never used for optimizer/checkpoint/calibration selection or representation design"
        )
        summary["regression_scope"] = "corrected public test plus inspected GDM46-GDM58 holdouts"
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

    def regression_g51_through_g58(task):
        parts = [
            old_g51_holdout(task), old_g52_holdout(task), old_g53_holdout(task),
            old_g54_holdout(task), old_g55_holdout(task), old_g56_holdout(task),
            old_g57_holdout(task), old_g58_holdout(task),
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
    g52.g51.fresh_holdout = regression_g51_through_g58
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
