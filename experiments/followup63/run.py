"""GDM63: learned rank-preserving cross-candidate interaction for operation ambiguity.

GDM62 ruled out fixed rank-1 candidate products/deltas as a broad improvement over
the canonical ranked request-relative representation. GDM63 keeps that canonical raw
representation exactly fixed and adds a small learned interaction adapter inside the
ambiguity head. All five arms have identical inputs, architecture, parameter count,
optimizer treatment, curriculum, calibration and downstream MLP; only the deterministic
relation used to turn projected rank slots into per-rank gates differs.

Forward scope is operation generation only. Schema generation is suspended/frozen.
Ledger/Index are calibration-only. Docket/Portfolio are the fresh secondary holdout.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from experiments.followup51 import run as g51
from experiments.followup52 import run as g52
from experiments.followup55 import run as g55
from experiments.followup56 import run as g56
from experiments.followup59 import run as g59
from experiments.followup60 import run as g60
from experiments.followup61 import run as g61
from experiments.followup62 import run as g62

FORMAT = "gdm63-measured-v1"
SEEDS_DEFAULT = "6301,6302"
CANONICAL_GDM62_SOURCE = "a3cfa70a570f931e2c78553ca9d72aeb5ed65e79"
CANONICAL_GDM62_AUDIT_RUN = 35965921793
CANONICAL_GDM62_PROMOTION = "7d925c87f80ecb69bf2ea38bfac9960faf08f9cd"
TOPK = 5
SEMANTIC_BLOCKS = 10
BASE_FEATURES = 11
LATENT_WIDTH = 4

_BASE = {
    "family": "explicit-none-plus-learned-candidate-interaction",
    "ambiguous_weight": 1.0,
    "hard_negative": False,
    "calibration_strategy": "ambiguity-rescue",
    "status_arbitration": "ambiguity-rescue",
    "nomatch_recall_budget_pp": 5,
    "ambiguity_curriculum": "family-balanced",
    "structured": "ranked-request-control",
}

INTERACTION_MODES = (
    "learned-local-control",
    "learned-top1-cross",
    "learned-neighbor-cross",
    "learned-allpairs-cross",
    "learned-competitive-cross",
)
ARMS = {
    mode: {**_BASE, "head": mode, "interaction_mode": mode}
    for mode in INTERACTION_MODES
}

CALIBRATION_LANGUAGE = [
    "reader-visible title for this entry",
    "time this entry was first committed",
    "time this entry was most recently revised",
    "integer score stored for each critique",
    "display names of people who authored critiques",
    "display names of people who moderated critiques",
    "registered company name of this entry's provider",
    "stable identifiers of people who authored critiques",
]
CALIBRATION_UNSUPPORTED = [
    "physical cabinet coordinate for this entry",
    "justification for an emergency access exemption",
    "international customs category for this entry",
    "flag indicating this entry is retained for legal discovery",
]
CALIBRATION_AMBIGUITIES = [
    ("role", "critique participant without choosing author or moderator role"),
    ("lifecycle-time", "entry time without choosing initial commit or latest revision"),
    ("representation", "critique author identity without choosing display name or stable identifier"),
    ("object-vs-supplier", "entry name without choosing visible title or provider company"),
]

HOLDOUT_LANGUAGE = [
    "public heading shown for this record",
    "instant this record was originally filed",
    "instant this record was most recently amended",
    "whole-number grade attached to each assessment",
    "human-readable names of people who wrote assessments",
    "human-readable names of people who moderated assessments",
    "legal business name of this record's provider",
    "persistent identifiers of people who wrote assessments",
]
HOLDOUT_UNSUPPORTED = [
    "archive drawer coordinate for this record",
    "reason for a privileged access exception",
    "cross-border commodity category for this record",
    "flag indicating this record is held for litigation discovery",
]
HOLDOUT_AMBIGUITIES = [
    ("role", "assessment participant without choosing writer or moderator role"),
    ("lifecycle-time", "record filing time without choosing original filing or latest amendment"),
    ("representation", "assessment writer identity without choosing readable name or persistent identifier"),
    ("object-vs-supplier", "record name without choosing public heading or provider business"),
]

_CANONICAL_CLAUSE_STATE = g60.clause_state


def ambiguity_feature_dim(encoder_dim: int) -> int:
    return BASE_FEATURES + SEMANTIC_BLOCKS * int(encoder_dim)


def interaction_parameter_count(encoder_dim: int) -> int:
    # Linear(2*d -> LATENT_WIDTH, bias=True) + rank gates + three scalar controls.
    return (2 * int(encoder_dim) * LATENT_WIDTH) + LATENT_WIDTH + TOPK + 3


class LearnedInteractionAmbiguityHead(nn.Module):
    """Matched-capacity ambiguity head with a small learned rank interaction adapter."""

    def __init__(self, dim: int, kind: str):
        super().__init__()
        if kind not in INTERACTION_MODES:
            raise ValueError(kind)
        semantic = int(dim) - BASE_FEATURES
        if semantic <= 0 or semantic % SEMANTIC_BLOCKS:
            raise ValueError((dim, semantic))
        self.kind = kind
        self.encoder_dim = semantic // SEMANTIC_BLOCKS
        self.proj = nn.Linear(2 * self.encoder_dim, LATENT_WIDTH, bias=True)
        self.rank_gate = nn.Parameter(torch.zeros(TOPK, dtype=torch.float32))
        self.log_temperature = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.residual_scale = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self.interaction_bias = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, 1))

    def _slots(self, x: torch.Tensor):
        d = self.encoder_dim
        first = x[:, BASE_FEATURES : BASE_FEATURES + TOPK * d].reshape(-1, TOPK, d)
        second = x[:, BASE_FEATURES + TOPK * d :].reshape(-1, TOPK, d)
        slots = torch.cat([first, second], dim=-1)
        active = (slots.abs().sum(dim=-1) > 0).to(dtype=x.dtype)
        return first, second, slots, active

    def _interaction_scores(self, z: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        # z: [batch, rank, latent]. All operations preserve rank identity.
        pair = torch.einsum("bir,bjr->bij", z, z) / math.sqrt(float(LATENT_WIDTH))
        if self.kind == "learned-local-control":
            return z.square().mean(dim=-1) * active
        if self.kind == "learned-top1-cross":
            return pair[:, :, 0] * active * active[:, 0:1]
        if self.kind == "learned-neighbor-cross":
            idx = torch.arange(TOPK, device=z.device)
            prev = torch.clamp(idx - 1, min=0)
            scores = pair[:, idx, prev]
            prev_active = active[:, prev]
            return scores * active * prev_active

        eye = torch.eye(TOPK, dtype=torch.bool, device=z.device).unsqueeze(0)
        valid = (active[:, :, None] * active[:, None, :]).bool() & (~eye)
        denom = valid.sum(dim=-1).clamp_min(1).to(dtype=z.dtype)
        mean_other = (pair * valid.to(dtype=z.dtype)).sum(dim=-1) / denom
        if self.kind == "learned-allpairs-cross":
            return mean_other * active
        if self.kind == "learned-competitive-cross":
            masked = pair.masked_fill(~valid, torch.finfo(pair.dtype).min)
            max_other = masked.max(dim=-1).values
            any_other = valid.any(dim=-1)
            max_other = torch.where(any_other, max_other, torch.zeros_like(max_other))
            return (max_other - mean_other) * active
        raise KeyError(self.kind)

    def forward(self, x: torch.Tensor):
        first, second, slots, active = self._slots(x)
        z = F.gelu(self.proj(slots)) * active[:, :, None]
        scores = self._interaction_scores(z, active)
        temperature = F.softplus(self.log_temperature) + 0.25
        rank_strength = torch.sigmoid(self.rank_gate)[None, :]
        alpha = 0.5 * torch.tanh(self.residual_scale)
        gate = 1.0 + alpha * rank_strength * torch.tanh(scores / temperature + self.interaction_bias)
        gate = torch.where(active.bool(), gate, torch.ones_like(gate))
        transformed = torch.cat(
            [
                x[:, :BASE_FEATURES],
                (first * gate[:, :, None]).reshape(x.shape[0], -1),
                (second * gate[:, :, None]).reshape(x.shape[0], -1),
            ],
            dim=-1,
        )
        return self.net(transformed).squeeze(-1)


class Bundle(nn.Module):
    def __init__(self, pair_dim: int, ambiguity_dim: int, kind: str):
        super().__init__()
        self.capability = g51.Head(pair_dim, adapter=True)
        self.ambiguity = LearnedInteractionAmbiguityHead(ambiguity_dim, kind)


def clause_state(bundle, encoder, item, clause, structured=False, online=False):
    # Every arm receives the exact canonical GDM59/GDM62 request-relative top-five raw vector.
    st = _CANONICAL_CLAUSE_STATE(
        bundle, encoder, item, clause, structured="ranked-request-control", online=online
    )
    st["ambiguity_representation_mode"] = "ranked-request-control"
    st["ambiguity_feature_dim"] = ambiguity_feature_dim(encoder.dim)
    st["ambiguity_topk_effective"] = min(TOPK, int(st.get("ambiguity_real_candidate_count", TOPK)))
    st["ambiguity_rank_preserving"] = True
    return st


def ambiguity_curriculum_examples(task, public, refs):
    return list(g56.ambiguity_curriculum_examples(task, public, refs, "family-balanced"))


def _interaction_description(mode: str) -> str:
    return {
        "learned-local-control": "canonical q*ci+abs(q-ci) slots; learned rank-local self gate only",
        "learned-top1-cross": "canonical q*ci+abs(q-ci) slots; learned rank1-to-rank gate",
        "learned-neighbor-cross": "canonical q*ci+abs(q-ci) slots; learned previous-rank neighbor gate",
        "learned-allpairs-cross": "canonical q*ci+abs(q-ci) slots; learned mean all-other-ranks gate",
        "learned-competitive-cross": "canonical q*ci+abs(q-ci) slots; learned strongest-other minus mean-other gate",
    }[mode]


def _interaction_state_hash(state: dict) -> str:
    keys = [
        k for k in state
        if k.startswith("ambiguity.proj.")
        or k.startswith("ambiguity.rank_gate")
        or k.startswith("ambiguity.log_temperature")
        or k.startswith("ambiguity.residual_scale")
        or k.startswith("ambiguity.interaction_bias")
    ]
    if not keys:
        raise AssertionError("missing learned interaction state")
    return g51.state_hash({k: state[k] for k in sorted(keys)})


def _representation_receipt(bundle, mode: str, encoder_dim: int, directory: Path):
    initial = torch.load(directory / "initial.pt", map_location="cpu", weights_only=True)["state"]
    selected = torch.load(directory / "selected.pt", map_location="cpu", weights_only=True)["state"]
    ih = _interaction_state_hash(initial)
    sh = _interaction_state_hash(selected)
    changed = [
        k for k in initial
        if k in selected
        and (
            k.startswith("ambiguity.proj.")
            or k.startswith("ambiguity.rank_gate")
            or k.startswith("ambiguity.log_temperature")
            or k.startswith("ambiguity.residual_scale")
            or k.startswith("ambiguity.interaction_bias")
        )
        and not torch.equal(initial[k], selected[k])
    ]
    if ih == sh or "ambiguity.proj.weight" not in changed:
        raise RuntimeError("GDM63 learned interaction adapter did not update")
    total = sum(p.numel() for p in bundle.ambiguity.parameters())
    adapter = sum(p.numel() for name, p in bundle.ambiguity.named_parameters() if not name.startswith("net."))
    expected_adapter = interaction_parameter_count(encoder_dim)
    if adapter != expected_adapter:
        raise AssertionError((adapter, expected_adapter))
    return {
        "gdm63_family": "learned-rank-preserving-cross-candidate-interaction",
        "gdm63_interaction_mode": mode,
        "gdm63_requested_topk": TOPK,
        "gdm63_ambiguity_feature_dim": ambiguity_feature_dim(encoder_dim),
        "gdm63_raw_representation": "canonical ranked request-product q*ci plus request-delta abs(q-ci)",
        "gdm63_interaction_description": _interaction_description(mode),
        "gdm63_latent_width": LATENT_WIDTH,
        "gdm63_interaction_parameter_count": adapter,
        "gdm63_ambiguity_parameter_count": total,
        "gdm63_initial_interaction_hash": ih,
        "gdm63_selected_interaction_hash": sh,
        "gdm63_changed_interaction_parameter_tensors": sorted(changed),
        "gdm63_rank_preserving": True,
        "gdm63_change_scope": (
            "small learned ambiguity-head interaction over fixed canonical top5 request-relative slots only; operation task only; fixed capability training, GDM56 family-balanced curriculum, canonical numerical path and GDM54 5pp rescue arbitration"
        ),
        "canonical_gdm62_source_commit": CANONICAL_GDM62_SOURCE,
        "canonical_gdm62_audit_run": CANONICAL_GDM62_AUDIT_RUN,
        "canonical_gdm62_promotion_commit": CANONICAL_GDM62_PROMOTION,
    }


def train_model(bundle, encoder, arm, task, public, refs, seed, epochs, directory):
    if task != "operation":
        raise RuntimeError("GDM63 is operation-only; schema generation is suspended")
    examples = ambiguity_curriculum_examples(task, public, refs)
    old_examples = g51.ambiguity_training_examples
    g51.ambiguity_training_examples = lambda _task, _public, _refs: list(examples)
    try:
        receipt = g60._ORIGINAL_TRAIN_MODEL(
            bundle, encoder, arm, task, public, refs, seed, epochs, directory
        )
    finally:
        g51.ambiguity_training_examples = old_examples
    receipt.update(g56._curriculum_receipt(examples, "family-balanced"))
    receipt.update(_representation_receipt(bundle, arm["interaction_mode"], encoder.dim, directory))
    return receipt


def calibration_set(task):
    if task != "operation":
        raise RuntimeError("GDM63 is operation-only")
    from experiments.followup50 import run as g50
    rows, refs = g50._dataset(
        task,
        [("Ledger", "ledger"), ("Index", "index")],
        CALIBRATION_LANGUAGE,
        CALIBRATION_UNSUPPORTED,
        [phrase for _, phrase in CALIBRATION_AMBIGUITIES],
        "gdm63-calibration",
    )
    return g55._tag_ambiguity_families(rows, refs, CALIBRATION_AMBIGUITIES)


def fresh_holdout(task):
    if task != "operation":
        raise RuntimeError("GDM63 is operation-only")
    from experiments.followup50 import run as g50
    rows, refs = g50._dataset(
        task,
        [("Docket", "docket"), ("Portfolio", "portfolio")],
        HOLDOUT_LANGUAGE,
        HOLDOUT_UNSUPPORTED,
        [phrase for _, phrase in HOLDOUT_AMBIGUITIES],
        "gdm63-holdout",
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
        mode = config["arm"]
        selected = json.loads((directory / "calibration.json").read_text())["selected"]
        config["research_question"] = (
            "does a small learned rank-preserving cross-candidate interaction improve operation ambiguity decisions beyond a matched-capacity rank-local control while canonical request-relative slots stay fixed"
        )
        config["forward_scope"] = "operation-generation-only"
        config["schema_generation"] = "suspended-frozen"
        config["ambiguity_curriculum"] = "family-balanced"
        config["ambiguity_representation_mode"] = "ranked-request-control"
        config["ambiguity_interaction_mode"] = mode
        config["ambiguity_feature_dim"] = training["gdm63_ambiguity_feature_dim"]
        config["ambiguity_requested_topk"] = TOPK
        config["ambiguity_rank_preserving"] = True
        training["selected_calibration"] = selected
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        training_path.write_text(json.dumps(training, indent=2, sort_keys=True) + "\n")
        summary["config"] = config
        summary["training"] = training
        summary["calibration_scope"] = (
            "disjoint Ledger/Index calibration-only operation cases; never used for optimizer updates or checkpoint selection"
        )
        summary["secondary_holdout_scope"] = (
            "new GDM63 Docket/Portfolio operation-only synthetic holdout; never used for optimizer/checkpoint/calibration selection or representation design"
        )
        summary["regression_scope"] = "corrected public operation test plus inspected GDM46-GDM62 operation holdouts"
        summary["secondary_holdout_ambiguity_family_metrics"] = g55._ambiguity_family_metrics(
            directory / "predictions-holdout.jsonl"
        )
        summary["file_hashes"]["config.json"] = g52.file_hash(config_path)
        summary["file_hashes"]["training.json"] = g52.file_hash(training_path)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main():
    # GDM63 reuses the hardened GDM60 orchestration but replaces only operation ambiguity head semantics.
    original = {
        "FORMAT": g60.FORMAT,
        "SEEDS_DEFAULT": g60.SEEDS_DEFAULT,
        "ARMS": g60.ARMS,
        "calibration_set": g60.calibration_set,
        "fresh_holdout": g60.fresh_holdout,
        "train_model": g60.train_model,
        "clause_state": g60.clause_state,
        "postprocess": g60._postprocess,
        "g51_bundle": g51.Bundle,
        "g59_holdout": g59.fresh_holdout,
    }
    prior_g59 = g59.fresh_holdout
    prior_g60 = g60.fresh_holdout
    prior_g61 = g61.fresh_holdout
    prior_g62 = g62.fresh_holdout

    def regression_g59_through_g62(task):
        if task != "operation":
            raise RuntimeError("GDM63 regression is operation-only")
        rows = []
        refs = {}
        for fn in (prior_g59, prior_g60, prior_g61, prior_g62):
            part_rows, part_refs = fn(task)
            rows.extend(part_rows)
            refs.update(part_refs)
        return rows, refs

    g60.FORMAT = FORMAT
    g60.SEEDS_DEFAULT = SEEDS_DEFAULT
    g60.ARMS = ARMS
    g60.calibration_set = calibration_set
    g60.fresh_holdout = fresh_holdout
    g60.train_model = train_model
    g60.clause_state = clause_state
    g60._postprocess = _postprocess
    g51.Bundle = Bundle
    g59.fresh_holdout = regression_g59_through_g62
    try:
        g60.main()
    finally:
        g60.FORMAT = original["FORMAT"]
        g60.SEEDS_DEFAULT = original["SEEDS_DEFAULT"]
        g60.ARMS = original["ARMS"]
        g60.calibration_set = original["calibration_set"]
        g60.fresh_holdout = original["fresh_holdout"]
        g60.train_model = original["train_model"]
        g60.clause_state = original["clause_state"]
        g60._postprocess = original["postprocess"]
        g51.Bundle = original["g51_bundle"]
        g59.fresh_holdout = original["g59_holdout"]


if __name__ == "__main__":
    main()
