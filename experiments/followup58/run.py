"""GDM58: broader top-k semantic ambiguity evidence under fixed canonical contracts.

Canonical GDM57 proved that request-conditioned top-two semantics contain substantial
ambiguity signal, but the gain trades against NO_MATCH/publication safety and remains
limited to the top two real capability candidates. GDM58 therefore changes only HOW
MANY ranked candidate semantics are pooled into the ambiguity representation.

The capability objective, explicit NONE outcome, exact canonical GDM56 family-balanced
ambiguity curriculum, frozen DistilBERT encoder, 1e-4 canonical feature boundary,
canonical GDM50 numerical execution, GDM54 5pp ambiguity-rescue arbitration, and
deterministic GraphQL/Federation realization remain fixed.

All five arms use the same fixed-dimensional, permutation-invariant semantic pooling
operator and therefore the same ambiguity-head input dimension/parameter count. Arms
differ only in the number of top-ranked real candidates included in that pool:
  * top2-pooled-control
  * top3-pooled
  * top4-pooled
  * top5-pooled
  * all-pooled

Fresh Notebook/Dispatch cases are calibration-only. Fresh Entitlement/Workbook cases
are the secondary holdout and never participate in optimizer updates, checkpoint
selection, representation design, or calibration selection. No transformer fine-tuning
occurs. Schema generation remains catalog projection plus deterministic SDL/Federation
realization. All compilation, tests, training, validation and benchmarking belong in
GitHub Actions.
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

FORMAT = "gdm58-measured-v1"
SEEDS_DEFAULT = "5801,5802"
CANONICAL_GDM57_SOURCE = "3f8b9a968285110ca6c744590724ddd0cb2bd5aa"

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
    "top2-pooled-control": {**_BASE, "structured": "top2-pooled-control"},
    "top3-pooled": {**_BASE, "structured": "top3-pooled"},
    "top4-pooled": {**_BASE, "structured": "top4-pooled"},
    "top5-pooled": {**_BASE, "structured": "top5-pooled"},
    "all-pooled": {**_BASE, "structured": "all-pooled"},
}

TOPK_MODES = tuple(ARMS)
SEMANTIC_BLOCKS = 10
_CURRENT_ENCODER_DIM = None

TOPK_LIMITS = {
    "top2-pooled-control": 2,
    "top3-pooled": 3,
    "top4-pooled": 4,
    "top5-pooled": 5,
    "all-pooled": None,
}

CALIBRATION_LANGUAGE = [
    "public heading displayed for this item",
    "instant this item first entered durable storage",
    "instant this item was most recently changed in durable storage",
    "integer score associated with every review",
    "display names of people who authored reviews",
    "display names of people who moderated reviews",
    "registered business name of the supplying organization",
    "persistent identifiers of review authors",
]
CALIBRATION_UNSUPPORTED = [
    "physical shelf coordinate assigned to this item",
    "reason a temporary compliance waiver was approved",
    "customs declaration category assigned to this item",
    "flag indicating this item is under evidence preservation",
]
CALIBRATION_AMBIGUITIES = [
    ("role", "review-associated person without choosing author versus moderator role"),
    ("lifecycle-time", "durable-storage instant without choosing first write versus latest rewrite"),
    ("representation", "review author identity without choosing display name versus persistent identifier"),
    ("object-vs-supplier", "display name without choosing item heading versus supplier organization"),
]

HOLDOUT_LANGUAGE = [
    "customer-visible heading for this thing",
    "time this thing was first committed to durable storage",
    "time this thing was last modified in durable storage",
    "whole-number rating attached to each review",
    "readable names of people who authored every review",
    "readable names of people who moderated every review",
    "legal organization name for this thing's supplier",
    "stable identifiers for people who authored every review",
]
HOLDOUT_UNSUPPORTED = [
    "warehouse aisle and bin assigned to this thing",
    "explanation for an emergency privileged-access waiver",
    "international commodity classification for this thing",
    "flag saying this thing is retained under evidence hold",
]
HOLDOUT_AMBIGUITIES = [
    ("role", "person attached to a review without choosing author or moderator role"),
    ("lifecycle-time", "storage time without choosing original persistence or most recent modification"),
    ("representation", "review author identity without choosing readable name or stable identifier"),
    ("object-vs-supplier", "name associated with the thing without choosing thing heading or supplier company"),
]

_ORIGINAL_TRAIN_MODEL = g56._ORIGINAL_TRAIN_MODEL
_ORIGINAL_RUN_ONE = g52.run_one
_ORIGINAL_CLAUSE_STATE = g51.clause_state


def ambiguity_feature_dim(encoder_dim: int) -> int:
    """Every GDM58 arm has the same ambiguity-head input and parameter count."""
    return 11 + SEMANTIC_BLOCKS * int(encoder_dim)


def _ambiguity_dim(_mode) -> int:
    if _CURRENT_ENCODER_DIM is None:
        raise RuntimeError("GDM58 encoder dimension was not bound before bundle construction")
    return ambiguity_feature_dim(_CURRENT_ENCODER_DIM)


def _topk_limit(mode: str, candidate_count: int) -> int:
    requested = TOPK_LIMITS[mode]
    if candidate_count <= 0:
        raise ValueError("GDM58 requires at least one real candidate")
    return candidate_count if requested is None else min(int(requested), candidate_count)


def _pooled_blocks(q: torch.Tensor, candidates: torch.Tensor, weights: torch.Tensor):
    """Return two 5d, fixed-capacity permutation-invariant top-k semantic blocks."""
    if candidates.ndim != 2 or candidates.shape[0] <= 0:
        raise ValueError("candidates must be non-empty [k,d]")
    if weights.ndim != 1 or weights.shape[0] != candidates.shape[0]:
        raise ValueError("weights must align with candidates")
    normalized = weights / weights.sum().clamp_min(1e-12)
    weighted_mean = (candidates * normalized[:, None]).sum(dim=0)
    mean = candidates.mean(dim=0)
    maximum = candidates.max(dim=0).values
    minimum = candidates.min(dim=0).values
    std = candidates.std(dim=0, unbiased=False)
    candidate_block = torch.cat([mean, weighted_mean, maximum, minimum, std], dim=0)

    absdiff = (candidates - q[None, :]).abs()
    request_block = torch.cat(
        [
            q,
            q * weighted_mean,
            (q - weighted_mean).abs(),
            absdiff.mean(dim=0),
            absdiff.max(dim=0).values,
        ],
        dim=0,
    )
    return candidate_block, request_block


def clause_state(bundle, encoder, item, clause, structured=False, online=False):
    mode = structured
    if mode not in TOPK_MODES:
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

    k = _topk_limit(mode, len(order))
    selected = order[:k]
    candidate_rows = torch.stack([x[int(i), dim : 2 * dim] for i in selected], dim=0)
    selected_weights = torch.stack([probs[int(i)] for i in selected], dim=0)
    candidate_block, request_block = _pooled_blocks(q, candidate_rows, selected_weights)
    features = base + candidate_block.detach().float().tolist() + request_block.detach().float().tolist()
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
        "ambiguity_topk_effective": k,
        "ambiguity_real_candidate_count": len(order),
    }


def ambiguity_curriculum_examples(task, public, refs):
    return list(g56.ambiguity_curriculum_examples(task, public, refs, "family-balanced"))


def _representation_receipt(mode, encoder_dim):
    requested = TOPK_LIMITS[mode]
    return {
        "gdm58_family": "top-k-pooled-semantic-ambiguity-evidence",
        "gdm58_representation_mode": mode,
        "gdm58_requested_topk": "all" if requested is None else int(requested),
        "gdm58_ambiguity_feature_dim": ambiguity_feature_dim(encoder_dim),
        "gdm58_semantic_pooling": "candidate(mean,weighted-mean,max,min,std)+request(q,q*weighted-mean,abs-q-weighted-mean,mean-absdiff,max-absdiff)",
        "gdm58_pooling_order_invariant": True,
        "gdm58_change_scope": (
            "ambiguity-head top-k semantic evidence breadth only; fixed canonical GDM56 family-balanced curriculum, canonical numerical path and GDM54 5pp rescue arbitration"
        ),
        "canonical_gdm57_source_commit": CANONICAL_GDM57_SOURCE,
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
        [("Notebook", "notebook"), ("Dispatch", "dispatch")],
        CALIBRATION_LANGUAGE,
        CALIBRATION_UNSUPPORTED,
        [phrase for _, phrase in CALIBRATION_AMBIGUITIES],
        "gdm58-calibration",
    )
    return g55._tag_ambiguity_families(rows, refs, CALIBRATION_AMBIGUITIES)


def fresh_holdout(task):
    rows, refs = g50._dataset(
        task,
        [("Entitlement", "entitlement"), ("Workbook", "workbook")],
        HOLDOUT_LANGUAGE,
        HOLDOUT_UNSUPPORTED,
        [phrase for _, phrase in HOLDOUT_AMBIGUITIES],
        "gdm58-holdout",
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
            "does exposing more than the top two ranked candidate semantics improve ambiguity discrimination without worsening NO_MATCH/publication safety"
        )
        config["ambiguity_curriculum"] = "family-balanced"
        config["ambiguity_representation_mode"] = mode
        config["ambiguity_feature_dim"] = training["gdm58_ambiguity_feature_dim"]
        config["ambiguity_requested_topk"] = training["gdm58_requested_topk"]
        training["selected_calibration"] = selected

        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        training_path.write_text(json.dumps(training, indent=2, sort_keys=True) + "\n")
        summary["config"] = config
        summary["training"] = training
        summary["calibration_scope"] = (
            "disjoint Notebook/Dispatch calibration-only cases; never used for optimizer updates or checkpoint selection"
        )
        summary["secondary_holdout_scope"] = (
            "new GDM58 Entitlement/Workbook synthetic holdout; never used for optimizer/checkpoint/calibration selection or representation design"
        )
        summary["regression_scope"] = "corrected public test plus inspected GDM46-GDM57 holdouts"
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

    def regression_g51_through_g57(task):
        parts = [
            old_g51_holdout(task),
            old_g52_holdout(task),
            old_g53_holdout(task),
            old_g54_holdout(task),
            old_g55_holdout(task),
            old_g56_holdout(task),
            old_g57_holdout(task),
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
    g52.g51.fresh_holdout = regression_g51_through_g57
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
