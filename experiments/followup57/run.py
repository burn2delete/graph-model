"""GDM57: richer top-two semantic evidence under fixed canonical curriculum.

Canonical GDM56 showed that relation-family balancing can stabilize operation
AMBIGUOUS recovery, but representation (name-vs-ID) and object-vs-supplier ambiguity
remain essentially unresolved and schema ambiguity remains weak. Counterfactual
curriculum factors made ambiguity worse. GDM57 therefore changes the ambiguity-head
INPUT REPRESENTATION while keeping the capability objective, canonical GDM56
family-balanced curriculum, frozen DistilBERT encoder, 1e-4 feature boundary,
canonical GDM50 numerical execution, GDM54 5pp ambiguity-rescue arbitration, and
deterministic GraphQL/Federation realization fixed.

All five arms retain the same top-two ambiguity decision boundary and the same
ambiguity-head parameter count. Arms differ only in which frozen semantic feature
blocks are exposed to the ambiguity head; unused blocks are zero-filled:
  * scalar-control: canonical 11 scalar/structural features only;
  * candidate-directed: add ordered top-two candidate semantic blocks;
  * request-conditioned: add request/candidate interaction blocks;
  * candidate-request: expose both directed candidate and request blocks;
  * symmetric-request: expose order-invariant candidate semantics plus request blocks.

Fresh Credential/Snapshot cases are calibration-only. Fresh Reservation/Artifact cases
are the secondary holdout and never participate in optimizer updates, checkpoint
selection, feature design, or calibration selection. No transformer fine-tuning occurs.
Schema generation remains catalog projection plus deterministic SDL/Federation
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

FORMAT = "gdm57-measured-v1"
SEEDS_DEFAULT = "5701,5702"
CANONICAL_GDM56_SOURCE = "168317f6495b05e04597ae084b75be6890fe0621"

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
    "scalar-control": {**_BASE, "structured": "scalar-control"},
    "candidate-directed": {**_BASE, "structured": "candidate-directed"},
    "request-conditioned": {**_BASE, "structured": "request-conditioned"},
    "candidate-request": {**_BASE, "structured": "candidate-request"},
    "symmetric-request": {**_BASE, "structured": "symmetric-request"},
}

REPRESENTATION_MODES = tuple(ARMS)
SEMANTIC_BLOCKS = 10
_CURRENT_ENCODER_DIM = None

CALIBRATION_LANGUAGE = [
    "public-facing heading attached to this record",
    "instant the record was first committed to persistent storage",
    "instant the record was most recently rewritten in persistent storage",
    "integer score recorded on each review",
    "display names of people who authored reviews",
    "display names of people who moderated reviews",
    "registered company name of the supplying organization",
    "stable identifiers of review authors",
]
CALIBRATION_UNSUPPORTED = [
    "physical rack coordinate assigned to this record",
    "reason an emergency policy exemption was granted",
    "cross-border customs class assigned to this record",
    "flag saying this record is subject to preservation hold",
]
CALIBRATION_AMBIGUITIES = [
    ("role", "review-associated person without choosing author versus moderator role"),
    ("lifecycle-time", "persistence instant without choosing original commit versus latest rewrite"),
    ("representation", "review author identity without choosing display name versus stable identifier"),
    ("object-vs-supplier", "display name without choosing record heading versus supplier organization"),
]

HOLDOUT_LANGUAGE = [
    "heading a customer sees for this object",
    "moment this object was first durably stored",
    "moment this object was most recently durably changed",
    "whole-number rating on every review",
    "names of people who authored each review",
    "names of people who moderated each review",
    "official organization name for this object's supplier",
    "persistent identifiers of people who authored each review",
]
HOLDOUT_UNSUPPORTED = [
    "warehouse slot assigned to this object",
    "justification for a temporary privileged-access exception",
    "international tariff grouping for this object",
    "flag saying this object is retained for evidence preservation",
]
HOLDOUT_AMBIGUITIES = [
    ("role", "person connected to a review without choosing author or moderator role"),
    ("lifecycle-time", "stored-at instant without choosing first persistence or latest rewrite"),
    ("representation", "review author identity without choosing readable name or persistent ID"),
    ("object-vs-supplier", "name associated with the object without choosing object heading or supplier business"),
]

_ORIGINAL_TRAIN_MODEL = g56._ORIGINAL_TRAIN_MODEL
_ORIGINAL_RUN_ONE = g52.run_one
_ORIGINAL_CLAUSE_STATE = g51.clause_state
_ORIGINAL_AMBIGUITY_DIM = g51._ambiguity_dim


def ambiguity_feature_dim(encoder_dim: int) -> int:
    """All GDM57 arms use exactly the same ambiguity-head input/parameter count."""
    return 11 + SEMANTIC_BLOCKS * int(encoder_dim)


def _ambiguity_dim(_mode) -> int:
    if _CURRENT_ENCODER_DIM is None:
        raise RuntimeError("GDM57 encoder dimension was not bound before bundle construction")
    return ambiguity_feature_dim(_CURRENT_ENCODER_DIM)


def _representation_blocks(q, c1, c2, mode):
    """Return two 5d semantic blocks with arm-specific masking.

    Candidate-directed preserves top-1/top-2 order. Symmetric-request removes that
    arbitrary ordering from the candidate block while retaining request conditioning.
    All arms return the same dimensionality so parameter count is controlled.
    """
    z = torch.zeros_like(q)
    directed = torch.cat([c1, c2, c1 - c2, (c1 - c2).abs(), c1 * c2], dim=0)
    midpoint = 0.5 * (c1 + c2)
    symmetric = torch.cat(
        [midpoint, (c1 - c2).abs(), c1 * c2, (c1 - c2).square(), torch.maximum(c1, c2)],
        dim=0,
    )
    request = torch.cat([q, q * c1, q * c2, (q - c1).abs(), (q - c2).abs()], dim=0)
    zero5 = torch.cat([z, z, z, z, z], dim=0)

    if mode == "scalar-control":
        return zero5, zero5
    if mode == "candidate-directed":
        return directed, zero5
    if mode == "request-conditioned":
        return zero5, request
    if mode == "candidate-request":
        return directed, request
    if mode == "symmetric-request":
        return symmetric, request
    raise ValueError(mode)


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
    candidate_block, request_block = _representation_blocks(q, c1, c2, mode)
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
    }


def ambiguity_curriculum_examples(task, public, refs):
    return list(g56.ambiguity_curriculum_examples(task, public, refs, "family-balanced"))


def _representation_receipt(mode, encoder_dim):
    return {
        "gdm57_family": "top-two-semantic-ambiguity-representation",
        "gdm57_representation_mode": mode,
        "gdm57_ambiguity_feature_dim": ambiguity_feature_dim(encoder_dim),
        "gdm57_candidate_semantic_block": mode in {"candidate-directed", "candidate-request", "symmetric-request"},
        "gdm57_request_semantic_block": mode in {"request-conditioned", "candidate-request", "symmetric-request"},
        "gdm57_order_invariant_candidate_block": mode == "symmetric-request",
        "gdm57_change_scope": (
            "ambiguity-head top-two semantic input representation only; fixed canonical GDM56 family-balanced curriculum, numerical path and GDM54 5pp rescue arbitration"
        ),
        "canonical_gdm56_source_commit": CANONICAL_GDM56_SOURCE,
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
        [("Credential", "credential"), ("Snapshot", "snapshot")],
        CALIBRATION_LANGUAGE,
        CALIBRATION_UNSUPPORTED,
        [phrase for _, phrase in CALIBRATION_AMBIGUITIES],
        "gdm57-calibration",
    )
    return g55._tag_ambiguity_families(rows, refs, CALIBRATION_AMBIGUITIES)


def fresh_holdout(task):
    rows, refs = g50._dataset(
        task,
        [("Reservation", "reservation"), ("Artifact", "artifact")],
        HOLDOUT_LANGUAGE,
        HOLDOUT_UNSUPPORTED,
        [phrase for _, phrase in HOLDOUT_AMBIGUITIES],
        "gdm57-holdout",
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
            "does richer top-two frozen semantic evidence let the ambiguity head represent relation distinctions that curriculum balancing alone did not solve"
        )
        config["ambiguity_curriculum"] = "family-balanced"
        config["ambiguity_representation_mode"] = mode
        config["ambiguity_feature_dim"] = training["gdm57_ambiguity_feature_dim"]
        training["selected_calibration"] = selected

        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        training_path.write_text(json.dumps(training, indent=2, sort_keys=True) + "\n")
        summary["config"] = config
        summary["training"] = training
        summary["calibration_scope"] = (
            "disjoint Credential/Snapshot calibration-only cases; never used for optimizer updates or checkpoint selection"
        )
        summary["secondary_holdout_scope"] = (
            "new GDM57 Reservation/Artifact synthetic holdout; never used for optimizer/checkpoint/calibration selection"
        )
        summary["regression_scope"] = "corrected public test plus inspected GDM46-GDM56 holdouts"
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

    def regression_g51_through_g56(task):
        parts = [
            old_g51_holdout(task),
            old_g52_holdout(task),
            old_g53_holdout(task),
            old_g54_holdout(task),
            old_g55_holdout(task),
            old_g56_holdout(task),
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
    g52.g51.fresh_holdout = regression_g51_through_g56
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
