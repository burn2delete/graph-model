"""GDM61: fixed balance between rank-preserving request/product and delta evidence.

Canonical GDM60 rejects simple capability-confidence attenuation: probability-scaled
ranked request interactions lose substantial answerable recall. It also provides a
narrower factorization signal: operation benefits most from q*c_i while schema's best
factorized arm is abs(q-c_i). GDM61 asks whether fixed, task-agnostic block balance can
retain those endpoint benefits without confidence weighting.

Every arm keeps the same top-five ranked real candidates, the same 11+10*d ambiguity
input dimension and trainable capacity, the exact GDM56 family-balanced curriculum,
frozen DistilBERT, 1e-4 canonical features, GDM50 numerical execution, GDM54 5pp
rescue arbitration, and deterministic GraphQL/Federation realization. Only fixed
relative scaling of the rank-preserving q*c_i and abs(q-c_i) semantic blocks changes.
Badge/Journal are calibration-only; Voucher/Anthology are the fresh secondary holdout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.followup51 import run as g51
from experiments.followup52 import run as g52
from experiments.followup55 import run as g55
from experiments.followup56 import run as g56
from experiments.followup59 import run as g59
from experiments.followup60 import run as g60

FORMAT = "gdm61-measured-v1"
SEEDS_DEFAULT = "6101,6102"
CANONICAL_GDM60_SOURCE = "c04784ec33983829f6fe55506bcb74e0f40442b2"
CANONICAL_GDM60_AUDIT_RUN = 35929537000
TOPK = 5
SEMANTIC_BLOCKS = 10

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
    "ranked-product-only": {**_BASE, "structured": "ranked-product-only"},
    "ranked-product-dominant": {**_BASE, "structured": "ranked-product-dominant"},
    "ranked-balanced-control": {**_BASE, "structured": "ranked-balanced-control"},
    "ranked-delta-dominant": {**_BASE, "structured": "ranked-delta-dominant"},
    "ranked-delta-only": {**_BASE, "structured": "ranked-delta-only"},
}
REPRESENTATION_MODES = tuple(ARMS)

BLOCK_SCALES = {
    "ranked-product-only": (1.0, 0.0),
    "ranked-product-dominant": (1.0, 0.5),
    "ranked-balanced-control": (1.0, 1.0),
    "ranked-delta-dominant": (0.5, 1.0),
    "ranked-delta-only": (0.0, 1.0),
}

CALIBRATION_LANGUAGE = [
    "reader-facing label assigned to this entry",
    "timestamp when this entry was first persisted",
    "timestamp when this entry was most recently revised",
    "whole-number rating attached to each critique",
    "display names of people who authored critiques",
    "display names of people who moderated critiques",
    "legal company name of the vendor for this entry",
    "stable IDs of people who authored critiques",
]
CALIBRATION_UNSUPPORTED = [
    "archive shelf code for this entry",
    "rationale for an emergency administrator escalation",
    "customs classification code for this entry",
    "flag indicating this entry is under a legal hold",
]
CALIBRATION_AMBIGUITIES = [
    ("role", "critique participant without choosing author or moderator role"),
    ("lifecycle-time", "persistence time without choosing initial save or latest revision"),
    ("representation", "critique author identity without choosing display name or stable ID"),
    ("object-vs-supplier", "entry name without choosing reader label or vendor company"),
]

HOLDOUT_LANGUAGE = [
    "public title shown for this asset",
    "instant this asset was initially saved",
    "instant this asset was most recently updated",
    "integer score recorded on each assessment",
    "human-readable names of people who wrote assessments",
    "human-readable names of people who moderated assessments",
    "registered business name of this asset's supplier",
    "persistent identifiers of people who wrote assessments",
]
HOLDOUT_UNSUPPORTED = [
    "warehouse bin locator for this asset",
    "explanation for a privileged-access exception",
    "international goods classification for this asset",
    "flag indicating this asset is preserved for litigation",
]
HOLDOUT_AMBIGUITIES = [
    ("role", "assessment participant without choosing writer or moderator role"),
    ("lifecycle-time", "saved timestamp without choosing initial save or latest update"),
    ("representation", "assessment writer identity without choosing readable name or persistent identifier"),
    ("object-vs-supplier", "asset name without choosing public title or supplier business"),
]


def ambiguity_feature_dim(encoder_dim: int) -> int:
    return 11 + SEMANTIC_BLOCKS * int(encoder_dim)


def _balance_blocks(q: torch.Tensor, candidates: torch.Tensor, weights: torch.Tensor, mode: str):
    padded, _probs, mask, _ = g60._pad_ranked(candidates, weights, TOPK)
    product = padded * q[None, :] * mask[:, None]
    delta = (padded - q[None, :]).abs() * mask[:, None]
    p_scale, d_scale = BLOCK_SCALES[mode]
    return (product * p_scale).reshape(-1), (delta * d_scale).reshape(-1)


def clause_state(bundle, encoder, item, clause, structured=False, online=False):
    mode = structured
    if mode not in REPRESENTATION_MODES:
        return g60._ORIGINAL_CLAUSE_STATE(bundle, encoder, item, clause, structured=structured, online=online)

    from experiments.followup50 import run as g50
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
    block1, block2 = _balance_blocks(q, candidate_rows, selected_weights, mode)
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
    p, d = BLOCK_SCALES[mode]
    return f"ranked top5 q*ci block scale={p:.1f}; abs(q-ci) block scale={d:.1f}"


def _representation_receipt(mode: str, encoder_dim: int):
    p, d = BLOCK_SCALES[mode]
    return {
        "gdm61_family": "fixed-ranked-request-interaction-block-balance",
        "gdm61_representation_mode": mode,
        "gdm61_requested_topk": TOPK,
        "gdm61_ambiguity_feature_dim": ambiguity_feature_dim(encoder_dim),
        "gdm61_semantic_representation": _representation_description(mode),
        "gdm61_product_scale": p,
        "gdm61_delta_scale": d,
        "gdm61_rank_preserving": True,
        "gdm61_change_scope": (
            "ambiguity-head fixed product/delta block balance only; fixed top5 breadth, rank preservation, capacity, canonical GDM56 family-balanced curriculum, canonical numerical path and GDM54 5pp rescue arbitration"
        ),
        "canonical_gdm60_source_commit": CANONICAL_GDM60_SOURCE,
        "canonical_gdm60_audit_run": CANONICAL_GDM60_AUDIT_RUN,
    }


def train_model(bundle, encoder, arm, task, public, refs, seed, epochs, directory):
    examples = ambiguity_curriculum_examples(task, public, refs)
    old_examples = g51.ambiguity_training_examples
    g51.ambiguity_training_examples = lambda _task, _public, _refs: list(examples)
    try:
        receipt = g60._ORIGINAL_TRAIN_MODEL(bundle, encoder, arm, task, public, refs, seed, epochs, directory)
    finally:
        g51.ambiguity_training_examples = old_examples
    receipt.update(g56._curriculum_receipt(examples, "family-balanced"))
    receipt.update(_representation_receipt(arm["structured"], encoder.dim))
    return receipt


def calibration_set(task):
    from experiments.followup50 import run as g50
    rows, refs = g50._dataset(
        task,
        [("Badge", "badge"), ("Journal", "journal")],
        CALIBRATION_LANGUAGE,
        CALIBRATION_UNSUPPORTED,
        [phrase for _, phrase in CALIBRATION_AMBIGUITIES],
        "gdm61-calibration",
    )
    return g55._tag_ambiguity_families(rows, refs, CALIBRATION_AMBIGUITIES)


def fresh_holdout(task):
    from experiments.followup50 import run as g50
    rows, refs = g50._dataset(
        task,
        [("Voucher", "voucher"), ("Anthology", "anthology")],
        HOLDOUT_LANGUAGE,
        HOLDOUT_UNSUPPORTED,
        [phrase for _, phrase in HOLDOUT_AMBIGUITIES],
        "gdm61-holdout",
    )
    return g55._tag_ambiguity_families(rows, refs, HOLDOUT_AMBIGUITIES)


def _output_root():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", default="artifacts/results")
    args, _ = parser.parse_known_args()
    return Path(args.output)


def _postprocess(root):
    for summary_path in root.rglob("summary.json"):
        directory = summary_path.parent
        config_path = directory / "config.json"
        training_path = directory / "training.json"
        config = json.loads(config_path.read_text())
        training = json.loads(training_path.read_text())
        summary = json.loads(summary_path.read_text())
        mode = ARMS[config["arm"]]["structured"]
        selected = json.loads((directory / "calibration.json").read_text())["selected"]
        config["research_question"] = (
            "can fixed balance between rank-preserving request-product and absolute-delta blocks improve the answerable/safety frontier after GDM60"
        )
        config["ambiguity_curriculum"] = "family-balanced"
        config["ambiguity_representation_mode"] = mode
        config["ambiguity_feature_dim"] = training["gdm61_ambiguity_feature_dim"]
        config["ambiguity_requested_topk"] = TOPK
        config["ambiguity_rank_preserving"] = True
        training["selected_calibration"] = selected
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        training_path.write_text(json.dumps(training, indent=2, sort_keys=True) + "\n")
        summary["config"] = config
        summary["training"] = training
        summary["calibration_scope"] = (
            "disjoint Badge/Journal calibration-only cases; never used for optimizer updates or checkpoint selection"
        )
        summary["secondary_holdout_scope"] = (
            "new GDM61 Voucher/Anthology synthetic holdout; never used for optimizer/checkpoint/calibration selection or representation design"
        )
        summary["regression_scope"] = "corrected public test plus inspected GDM46-GDM60 holdouts"
        summary["secondary_holdout_ambiguity_family_metrics"] = g55._ambiguity_family_metrics(
            directory / "predictions-holdout.jsonl"
        )
        summary["file_hashes"]["config.json"] = g52.file_hash(config_path)
        summary["file_hashes"]["training.json"] = g52.file_hash(training_path)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main():
    original = {
        "FORMAT": g60.FORMAT,
        "SEEDS_DEFAULT": g60.SEEDS_DEFAULT,
        "ARMS": g60.ARMS,
        "REPRESENTATION_MODES": g60.REPRESENTATION_MODES,
        "calibration_set": g60.calibration_set,
        "fresh_holdout": g60.fresh_holdout,
        "train_model": g60.train_model,
        "clause_state": g60.clause_state,
        "postprocess": g60._postprocess,
        "g59_holdout": g59.fresh_holdout,
    }
    prior_g59 = g59.fresh_holdout
    prior_g60 = g60.fresh_holdout

    def regression_g59_and_g60(task):
        rows59, refs59 = prior_g59(task)
        rows60, refs60 = prior_g60(task)
        return list(rows59) + list(rows60), {**refs59, **refs60}

    g60.FORMAT = FORMAT
    g60.SEEDS_DEFAULT = SEEDS_DEFAULT
    g60.ARMS = ARMS
    g60.REPRESENTATION_MODES = REPRESENTATION_MODES
    g60.calibration_set = calibration_set
    g60.fresh_holdout = fresh_holdout
    g60.train_model = train_model
    g60.clause_state = clause_state
    g60._postprocess = _postprocess
    g59.fresh_holdout = regression_g59_and_g60
    try:
        g60.main()
    finally:
        g60.FORMAT = original["FORMAT"]
        g60.SEEDS_DEFAULT = original["SEEDS_DEFAULT"]
        g60.ARMS = original["ARMS"]
        g60.REPRESENTATION_MODES = original["REPRESENTATION_MODES"]
        g60.calibration_set = original["calibration_set"]
        g60.fresh_holdout = original["fresh_holdout"]
        g60.train_model = original["train_model"]
        g60.clause_state = original["clause_state"]
        g60._postprocess = original["postprocess"]
        g59.fresh_holdout = original["g59_holdout"]


if __name__ == "__main__":
    main()
