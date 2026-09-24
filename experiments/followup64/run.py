"""GDM64: schema-coordinate semantic factorization for operation ambiguity.

Canonical GDM63 found that a small learned rank-preserving cross-candidate gate does
not improve answerable correctness, target precision/recall, multi-clause exactness,
or the role/representation ambiguity families over a matched-capacity rank-local
control. GDM64 changes the information factorization instead of adding more candidate
interaction: every arm keeps TOP5 rank identity and exactly two d-dimensional semantic
blocks per rank, but factorized arms embed deterministic parent/role and leaf/
representation views derived only from each provided GraphQL schema coordinate.

Forward scope is operation generation only. Schema generation is suspended/frozen.
Registry/Logbook are calibration-only. Folio/Casebook are the fresh secondary holdout.
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
from experiments.followup55 import run as g55
from experiments.followup56 import run as g56
from experiments.followup59 import run as g59
from experiments.followup60 import run as g60
from experiments.followup61 import run as g61
from experiments.followup62 import run as g62
from experiments.followup63 import run as g63

FORMAT = "gdm64-measured-v1"
SEEDS_DEFAULT = "6401,6402"
CANONICAL_GDM63_SOURCE = "43a6d16652a3fd63c155f46e99b42c757c7601a1"
CANONICAL_GDM63_AUDIT_RUN = 35988393672
CANONICAL_GDM63_PROMOTION = "4efc41f161fadd59f563f491f251ab6e058fcb71"
TOPK = 5
SEMANTIC_BLOCKS = 10
BASE_FEATURES = 11

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

REPRESENTATION_MODES = (
    "whole-request-control",
    "parent-request-factor",
    "leaf-request-factor",
    "split-parent-leaf-product",
    "split-parent-leaf-delta",
)
ARMS = {
    mode: {**_BASE, "structured": mode}
    for mode in REPRESENTATION_MODES
}

BLOCK_KINDS = {
    "whole-request-control": ("whole-product", "whole-delta"),
    "parent-request-factor": ("parent-product", "parent-delta"),
    "leaf-request-factor": ("leaf-product", "leaf-delta"),
    "split-parent-leaf-product": ("parent-product", "leaf-product"),
    "split-parent-leaf-delta": ("parent-delta", "leaf-delta"),
}

CALIBRATION_LANGUAGE = [
    "reader-facing title for this registry entry",
    "time this registry entry was first stored",
    "time this registry entry was most recently revised",
    "whole-number score attached to each review",
    "display names of people who authored reviews",
    "display names of people who moderated reviews",
    "registered company name of this entry's supplier",
    "stable identifiers of people who authored reviews",
]
CALIBRATION_UNSUPPORTED = [
    "physical archive shelf assigned to this registry entry",
    "reason an emergency privileged-access exception was approved",
    "international tariff category assigned to this registry entry",
    "flag indicating this registry entry is held for discovery",
]
CALIBRATION_AMBIGUITIES = [
    ("role", "review participant without choosing author or moderator role"),
    ("lifecycle-time", "registry time without choosing first storage or latest revision"),
    ("representation", "review author identity without choosing display name or stable identifier"),
    ("object-vs-supplier", "registry entry name without choosing entry title or supplier company"),
]

HOLDOUT_LANGUAGE = [
    "public title printed for this folio entry",
    "instant this folio entry was originally recorded",
    "instant this folio entry was last amended",
    "integer rating stored on every critique",
    "human-readable names of people who wrote critiques",
    "human-readable names of people who moderated critiques",
    "legal company name of this folio entry's supplier",
    "persistent identifiers of people who wrote critiques",
]
HOLDOUT_UNSUPPORTED = [
    "warehouse cabinet coordinate for this folio entry",
    "justification for an exceptional administrator bypass",
    "customs commodity category for this folio entry",
    "flag indicating this folio entry is retained for litigation",
]
HOLDOUT_AMBIGUITIES = [
    ("role", "critique participant without choosing writer or moderator role"),
    ("lifecycle-time", "folio record time without choosing original recording or latest amendment"),
    ("representation", "critique writer identity without choosing readable name or persistent identifier"),
    ("object-vs-supplier", "folio entry name without choosing public title or supplier company"),
]


def ambiguity_feature_dim(encoder_dim: int) -> int:
    return BASE_FEATURES + SEMANTIC_BLOCKS * int(encoder_dim)


def _coordinate_texts(option):
    """Deterministic catalog-only parent/role and leaf/representation views."""
    path = list(option.get("path") or [])
    if not path:
        raise ValueError("GDM64 coordinate factorization requires a real GraphQL path")
    parent = ".".join(str(x) for x in path[1:-1]) or "<root>"
    leaf = str(path[-1])
    return f"GraphQL parent path: {parent}", f"GraphQL leaf field: {leaf}"


def _pad_rows(rows: torch.Tensor, topk: int = TOPK):
    if rows.ndim != 2 or rows.shape[0] <= 0:
        raise ValueError("semantic rows must be non-empty [k,d]")
    k, dim = rows.shape
    k = min(k, topk)
    padded = torch.zeros((topk, dim), dtype=rows.dtype, device=rows.device)
    padded[:k] = rows[:k]
    mask = torch.zeros((topk,), dtype=rows.dtype, device=rows.device)
    mask[:k] = 1.0
    return padded, mask, k


def _factor_blocks(q: torch.Tensor, whole: torch.Tensor, parent: torch.Tensor, leaf: torch.Tensor, mode: str):
    whole_pad, mask, k = _pad_rows(whole)
    parent_pad, parent_mask, pk = _pad_rows(parent)
    leaf_pad, leaf_mask, lk = _pad_rows(leaf)
    if pk != k or lk != k or not torch.equal(mask, parent_mask) or not torch.equal(mask, leaf_mask):
        raise AssertionError("GDM64 whole/parent/leaf rank masks diverged")
    m = mask[:, None]
    blocks = {
        "whole-product": whole_pad * q[None, :] * m,
        "whole-delta": (whole_pad - q[None, :]).abs() * m,
        "parent-product": parent_pad * q[None, :] * m,
        "parent-delta": (parent_pad - q[None, :]).abs() * m,
        "leaf-product": leaf_pad * q[None, :] * m,
        "leaf-delta": (leaf_pad - q[None, :]).abs() * m,
    }
    first_kind, second_kind = BLOCK_KINDS[mode]
    return blocks[first_kind].reshape(-1), blocks[second_kind].reshape(-1), k


def clause_state(bundle, encoder, item, clause, structured=False, online=False):
    mode = structured
    if mode not in REPRESENTATION_MODES:
        return g60._ORIGINAL_CLAUSE_STATE(bundle, encoder, item, clause, structured=structured, online=online)

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
    selected_indices = [int(i) for i in selected]
    whole_rows = torch.stack([x[i, dim : 2 * dim] for i in selected_indices], dim=0)
    parent_texts = []
    leaf_texts = []
    for i in selected_indices:
        parent_text, leaf_text = _coordinate_texts(opts[i])
        parent_texts.append(parent_text)
        leaf_texts.append(leaf_text)
    # These are schema/catalog-static views. Keep them cached even for online timing;
    # only the request clause is allowed to be freshly encoded during timed generation.
    parent_rows = encoder.encode(parent_texts, cached=True)
    leaf_rows = encoder.encode(leaf_texts, cached=True)

    block1, block2, effective = _factor_blocks(q, whole_rows, parent_rows, leaf_rows, mode)
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
        "ambiguity_topk_effective": effective,
        "ambiguity_real_candidate_count": len(order),
        "ambiguity_rank_preserving": True,
        "ambiguity_catalog_semantic_views_cached": True,
    }


def ambiguity_curriculum_examples(task, public, refs):
    if task != "operation":
        raise RuntimeError("GDM64 is operation-only; schema generation is suspended")
    return list(g56.ambiguity_curriculum_examples(task, public, refs, "family-balanced"))


def _representation_description(mode: str) -> str:
    first, second = BLOCK_KINDS[mode]
    return f"ranked top5 {first} block + {second} block from provided schema coordinates"


def _representation_receipt(bundle, mode: str, encoder_dim: int):
    first, second = BLOCK_KINDS[mode]
    return {
        "gdm64_family": "schema-coordinate-semantic-factorization",
        "gdm64_representation_mode": mode,
        "gdm64_requested_topk": TOPK,
        "gdm64_ambiguity_feature_dim": ambiguity_feature_dim(encoder_dim),
        "gdm64_semantic_representation": _representation_description(mode),
        "gdm64_first_block": first,
        "gdm64_second_block": second,
        "gdm64_parent_text_contract": "GraphQL parent path: + path[1:-1] joined by dot; <root> if empty",
        "gdm64_leaf_text_contract": "GraphQL leaf field: + path[-1]",
        "gdm64_catalog_semantic_views_cached": True,
        "gdm64_rank_preserving": True,
        "gdm64_ambiguity_parameter_count": sum(p.numel() for p in bundle.ambiguity.parameters()),
        "gdm64_change_scope": (
            "operation-only schema-coordinate semantic factorization inside fixed top5 ranked ambiguity slots; deterministic parent/role and leaf/representation views derived only from provided GraphQL paths; fixed dimensions, ambiguity-head capacity, capability training, GDM56 family-balanced curriculum, canonical numerical path and GDM54 5pp rescue arbitration"
        ),
        "canonical_gdm63_source_commit": CANONICAL_GDM63_SOURCE,
        "canonical_gdm63_audit_run": CANONICAL_GDM63_AUDIT_RUN,
        "canonical_gdm63_promotion_commit": CANONICAL_GDM63_PROMOTION,
    }


def train_model(bundle, encoder, arm, task, public, refs, seed, epochs, directory):
    if task != "operation":
        raise RuntimeError("GDM64 is operation-only; schema generation is suspended")
    examples = ambiguity_curriculum_examples(task, public, refs)
    old_examples = g51.ambiguity_training_examples
    g51.ambiguity_training_examples = lambda _task, _public, _refs: list(examples)
    try:
        receipt = g60._ORIGINAL_TRAIN_MODEL(bundle, encoder, arm, task, public, refs, seed, epochs, directory)
    finally:
        g51.ambiguity_training_examples = old_examples
    receipt.update(g56._curriculum_receipt(examples, "family-balanced"))
    receipt.update(_representation_receipt(bundle, arm["structured"], encoder.dim))
    return receipt


def calibration_set(task):
    if task != "operation":
        raise RuntimeError("GDM64 is operation-only")
    rows, refs = g50._dataset(
        task,
        [("Registry", "registry"), ("Logbook", "logbook")],
        CALIBRATION_LANGUAGE,
        CALIBRATION_UNSUPPORTED,
        [phrase for _, phrase in CALIBRATION_AMBIGUITIES],
        "gdm64-calibration",
    )
    return g55._tag_ambiguity_families(rows, refs, CALIBRATION_AMBIGUITIES)


def fresh_holdout(task):
    if task != "operation":
        raise RuntimeError("GDM64 is operation-only")
    rows, refs = g50._dataset(
        task,
        [("Folio", "folio"), ("Casebook", "casebook")],
        HOLDOUT_LANGUAGE,
        HOLDOUT_UNSUPPORTED,
        [phrase for _, phrase in HOLDOUT_AMBIGUITIES],
        "gdm64-holdout",
    )
    return g55._tag_ambiguity_families(rows, refs, HOLDOUT_AMBIGUITIES)


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
            "does deterministic schema-coordinate parent/role and leaf/representation factorization expose operation ambiguity distinctions that whole-candidate embeddings and GDM63 candidate-set gates missed"
        )
        config["forward_scope"] = "operation-generation-only"
        config["schema_generation"] = "suspended-frozen"
        config["ambiguity_curriculum"] = "family-balanced"
        config["ambiguity_representation_mode"] = mode
        config["ambiguity_feature_dim"] = training["gdm64_ambiguity_feature_dim"]
        config["ambiguity_requested_topk"] = TOPK
        config["ambiguity_rank_preserving"] = True
        config["ambiguity_catalog_semantic_views_cached"] = True
        training["selected_calibration"] = selected
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        training_path.write_text(json.dumps(training, indent=2, sort_keys=True) + "\n")
        summary["config"] = config
        summary["training"] = training
        summary["calibration_scope"] = (
            "disjoint Registry/Logbook calibration-only operation cases; never used for optimizer updates or checkpoint selection"
        )
        summary["secondary_holdout_scope"] = (
            "new GDM64 Folio/Casebook operation-only synthetic holdout; never used for optimizer/checkpoint/calibration selection or representation design"
        )
        summary["regression_scope"] = "corrected public operation test plus inspected GDM46-GDM63 operation holdouts"
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
    prior_g62 = g62.fresh_holdout
    prior_g63 = g63.fresh_holdout

    def regression_g59_through_g63(task):
        if task != "operation":
            raise RuntimeError("GDM64 regression is operation-only")
        rows = []
        refs = {}
        for fn in (prior_g59, prior_g60, prior_g61, prior_g62, prior_g63):
            part_rows, part_refs = fn(task)
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
    g59.fresh_holdout = regression_g59_through_g63
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
