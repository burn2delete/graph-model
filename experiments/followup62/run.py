"""GDM62: explicit rank-preserving candidate-to-candidate semantic contrast.

GDM61 found no reliable answerable/safety improvement from fixed scaling of the
already-present request-relative product and absolute-delta blocks. GDM62 therefore
adds genuinely candidate-relational information: nonlinear relations between the
rank-1 candidate and each ranked candidate, paired with one request-relative block.

Every arm keeps the same top-five ranked real candidates, 11+10*d ambiguity input
dimension/trainable capacity, exact GDM56 family-balanced curriculum, frozen
DistilBERT, 1e-4 canonical features, GDM50 numerical execution, GDM54 5pp rescue
arbitration, and deterministic GraphQL/Federation realization. Only the two 5*d
semantic blocks supplied to the ambiguity head change. Medallion/Gazette are
calibration-only; Ticket/Compendium are the fresh secondary holdout.
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
from experiments.followup61 import run as g61

FORMAT = "gdm62-measured-v1"
SEEDS_DEFAULT = "6201,6202"
CANONICAL_GDM61_SOURCE = "f22fc36caf7327ed3d0d1f7dd2898ea31d837f3a"
CANONICAL_GDM61_AUDIT_RUN = 35948010054
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
    "ranked-request-control": {**_BASE, "structured": "ranked-request-control"},
    "ranked-product-candidate-delta": {**_BASE, "structured": "ranked-product-candidate-delta"},
    "ranked-delta-candidate-delta": {**_BASE, "structured": "ranked-delta-candidate-delta"},
    "ranked-product-candidate-product": {**_BASE, "structured": "ranked-product-candidate-product"},
    "ranked-delta-candidate-product": {**_BASE, "structured": "ranked-delta-candidate-product"},
}
REPRESENTATION_MODES = tuple(ARMS)

BLOCK_KINDS = {
    "ranked-request-control": ("request-product", "request-delta"),
    "ranked-product-candidate-delta": ("request-product", "candidate-delta"),
    "ranked-delta-candidate-delta": ("request-delta", "candidate-delta"),
    "ranked-product-candidate-product": ("request-product", "candidate-product"),
    "ranked-delta-candidate-product": ("request-delta", "candidate-product"),
}

CALIBRATION_LANGUAGE = [
    "public caption displayed for this record",
    "moment this record was first entered",
    "moment this record was most recently amended",
    "whole-number score stored on each evaluation",
    "readable names of people who wrote evaluations",
    "readable names of people who moderated evaluations",
    "registered organization name of this record's provider",
    "persistent identifiers of people who wrote evaluations",
]
CALIBRATION_UNSUPPORTED = [
    "vault compartment locator for this record",
    "reason for an elevated-access waiver",
    "cross-border tariff classification for this record",
    "flag indicating this record is frozen by a preservation order",
]
CALIBRATION_AMBIGUITIES = [
    ("role", "evaluation participant without choosing writer or moderator role"),
    ("lifecycle-time", "record time without choosing first entry or latest amendment"),
    ("representation", "evaluation writer identity without choosing readable name or persistent identifier"),
    ("object-vs-supplier", "record name without choosing public caption or provider organization"),
]

HOLDOUT_LANGUAGE = [
    "visible heading presented for this item",
    "time this item was originally recorded",
    "time this item was last changed",
    "integer grade attached to each appraisal",
    "display names of people who created appraisals",
    "display names of people who moderated appraisals",
    "legal business name of this item's source",
    "durable identifiers of people who created appraisals",
]
HOLDOUT_UNSUPPORTED = [
    "storage rack coordinate for this item",
    "justification for a privileged-operator override",
    "harmonized trade category for this item",
    "flag indicating this item is retained under litigation policy",
]
HOLDOUT_AMBIGUITIES = [
    ("role", "appraisal participant without choosing creator or moderator role"),
    ("lifecycle-time", "item timestamp without choosing original record or latest change"),
    ("representation", "appraisal creator identity without choosing display name or durable identifier"),
    ("object-vs-supplier", "item name without choosing visible heading or source business"),
]


def ambiguity_feature_dim(encoder_dim: int) -> int:
    return 11 + SEMANTIC_BLOCKS * int(encoder_dim)


def _relation_blocks(q: torch.Tensor, candidates: torch.Tensor, weights: torch.Tensor, mode: str):
    """Return two fixed 5*d blocks; only their semantic relation differs by arm."""
    padded, _probs, mask, _ = g60._pad_ranked(candidates, weights, TOPK)
    m = mask[:, None]
    request_product = padded * q[None, :] * m
    request_delta = (padded - q[None, :]).abs() * m
    anchor = padded[0] * mask[0]
    candidate_delta = (padded - anchor[None, :]).abs() * m
    candidate_product = padded * anchor[None, :] * m

    first_kind, second_kind = BLOCK_KINDS[mode]
    blocks = {
        "request-product": request_product,
        "request-delta": request_delta,
        "candidate-delta": candidate_delta,
        "candidate-product": candidate_product,
    }
    return blocks[first_kind].reshape(-1), blocks[second_kind].reshape(-1)


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
    block1, block2 = _relation_blocks(q, candidate_rows, selected_weights, mode)
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
    first, second = BLOCK_KINDS[mode]
    return f"ranked top5 {first} block + {second} block; candidate relations anchored at rank1"


def _representation_receipt(mode: str, encoder_dim: int):
    first, second = BLOCK_KINDS[mode]
    return {
        "gdm62_family": "ranked-candidate-relational-ambiguity-evidence",
        "gdm62_representation_mode": mode,
        "gdm62_requested_topk": TOPK,
        "gdm62_ambiguity_feature_dim": ambiguity_feature_dim(encoder_dim),
        "gdm62_semantic_representation": _representation_description(mode),
        "gdm62_first_block": first,
        "gdm62_second_block": second,
        "gdm62_candidate_anchor": "rank1-real-candidate",
        "gdm62_rank_preserving": True,
        "gdm62_change_scope": (
            "ambiguity-head candidate-relational semantic evidence only; fixed top5 breadth, rank preservation, capacity, canonical GDM56 family-balanced curriculum, canonical numerical path and GDM54 5pp rescue arbitration"
        ),
        "canonical_gdm61_source_commit": CANONICAL_GDM61_SOURCE,
        "canonical_gdm61_audit_run": CANONICAL_GDM61_AUDIT_RUN,
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
        [("Medallion", "medallion"), ("Gazette", "gazette")],
        CALIBRATION_LANGUAGE,
        CALIBRATION_UNSUPPORTED,
        [phrase for _, phrase in CALIBRATION_AMBIGUITIES],
        "gdm62-calibration",
    )
    return g55._tag_ambiguity_families(rows, refs, CALIBRATION_AMBIGUITIES)


def fresh_holdout(task):
    from experiments.followup50 import run as g50
    rows, refs = g50._dataset(
        task,
        [("Ticket", "ticket"), ("Compendium", "compendium")],
        HOLDOUT_LANGUAGE,
        HOLDOUT_UNSUPPORTED,
        [phrase for _, phrase in HOLDOUT_AMBIGUITIES],
        "gdm62-holdout",
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
            "does explicit rank-preserving candidate-to-candidate semantic contrast improve ambiguity decisions after fixed product/delta scaling failed in GDM61"
        )
        config["ambiguity_curriculum"] = "family-balanced"
        config["ambiguity_representation_mode"] = mode
        config["ambiguity_feature_dim"] = training["gdm62_ambiguity_feature_dim"]
        config["ambiguity_requested_topk"] = TOPK
        config["ambiguity_rank_preserving"] = True
        training["selected_calibration"] = selected
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        training_path.write_text(json.dumps(training, indent=2, sort_keys=True) + "\n")
        summary["config"] = config
        summary["training"] = training
        summary["calibration_scope"] = (
            "disjoint Medallion/Gazette calibration-only cases; never used for optimizer updates or checkpoint selection"
        )
        summary["secondary_holdout_scope"] = (
            "new GDM62 Ticket/Compendium synthetic holdout; never used for optimizer/checkpoint/calibration selection or representation design"
        )
        summary["regression_scope"] = "corrected public test plus inspected GDM46-GDM61 holdouts"
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
    prior_g61 = g61.fresh_holdout

    def regression_g59_through_g61(task):
        parts = [prior_g59(task), prior_g60(task), prior_g61(task)]
        rows = []
        refs = {}
        for part_rows, part_refs in parts:
            rows.extend(part_rows)
            refs.update(part_refs)
        return rows, refs

    g60.FORMAT = FORMAT
    g60.SEEDS_DEFAULT = SEEDS_DEFAULT
    g60.ARMS = ARMS
    g60.REPRESENTATION_MODES = REPRESENTATION_MODES
    g60.calibration_set = calibration_set
    g60.fresh_holdout = fresh_holdout
    g60.train_model = train_model
    g60.clause_state = clause_state
    g60._postprocess = _postprocess
    g59.fresh_holdout = regression_g59_through_g61
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
