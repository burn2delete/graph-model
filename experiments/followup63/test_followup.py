from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from experiments.followup46.run import corrected_dataset
from experiments.followup56 import run as g56
from experiments.followup59 import run as g59
from experiments.followup63 import run


class GDM63Contracts(unittest.TestCase):
    def _dataset(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        public, refs, _ = corrected_dataset(Path(temp.name) / "dataset")
        return public, refs

    def test_exact_operation_only_ten_configuration_matrix(self):
        self.assertEqual(len(run.ARMS), 5)
        self.assertEqual(len(run.ARMS) * 2, 10)
        self.assertEqual(tuple(run.ARMS), run.INTERACTION_MODES)
        self.assertEqual(run.TOPK, 5)
        self.assertEqual(run.SEMANTIC_BLOCKS, 10)
        self.assertEqual(run.LATENT_WIDTH, 4)
        for arm_name, arm in run.ARMS.items():
            self.assertEqual(arm["family"], "explicit-none-plus-learned-candidate-interaction")
            self.assertEqual(arm["head"], arm_name)
            self.assertEqual(arm["interaction_mode"], arm_name)
            self.assertEqual(arm["structured"], "ranked-request-control")
            self.assertEqual(arm["ambiguity_curriculum"], "family-balanced")
            self.assertEqual(arm["calibration_strategy"], "ambiguity-rescue")
            self.assertEqual(arm["nomatch_recall_budget_pp"], 5)

    def test_all_arms_use_exact_canonical_gdm56_balanced_curriculum(self):
        public, refs = self._dataset()
        expected = g56.ambiguity_curriculum_examples("operation", public, refs, "family-balanced")
        actual = run.ambiguity_curriculum_examples("operation", public, refs)
        self.assertEqual(actual, expected)
        receipt = g56._curriculum_receipt(actual, "family-balanced")
        counts = receipt["gdm56_positive_family_counts"]
        self.assertEqual(set(counts), {"role", "lifecycle-time", "representation", "object-vs-supplier"})
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertEqual(receipt["gdm56_counterfactual_negative_fraction"], 0.0)

    def test_fixed_raw_dimension_and_exact_matched_capacity(self):
        self.assertEqual(run.ambiguity_feature_dim(128), 1291)
        self.assertEqual(run.ambiguity_feature_dim(768), 7691)
        for dim in (128, 768):
            feature_dim = run.ambiguity_feature_dim(dim)
            counts = []
            adapter_counts = []
            for mode in run.INTERACTION_MODES:
                head = run.LearnedInteractionAmbiguityHead(feature_dim, mode)
                counts.append(sum(p.numel() for p in head.parameters()))
                adapter_counts.append(sum(p.numel() for name, p in head.named_parameters() if not name.startswith("net.")))
            self.assertEqual(len(set(counts)), 1)
            self.assertEqual(len(set(adapter_counts)), 1)
            self.assertEqual(adapter_counts[0], run.interaction_parameter_count(dim))

    def test_canonical_raw_request_relative_slots_remain_exact(self):
        q = torch.tensor([0.2, -0.4])
        candidates = torch.tensor([
            [0.5, 0.1], [0.1, -0.2], [-0.3, 0.7], [0.8, -0.5], [0.4, 0.6]
        ])
        canonical = g59._ranked_blocks(q, candidates, "ranked-request-only")
        product = (candidates * q[None, :]).reshape(-1)
        delta = (candidates - q[None, :]).abs().reshape(-1)
        torch.testing.assert_close(canonical[0], product, rtol=0.0, atol=0.0)
        torch.testing.assert_close(canonical[1], delta, rtol=0.0, atol=0.0)

    def test_local_control_has_no_cross_rank_score_dependency(self):
        z = torch.tensor([[[1.0, 0.5, -0.2, 0.3], [0.4, -0.1, 0.8, 0.2], [0.2, 0.3, 0.1, -0.5], [0.6, 0.7, -0.4, 0.1], [0.9, -0.2, 0.2, 0.4]]])
        active = torch.ones((1, 5))
        head = run.LearnedInteractionAmbiguityHead(1291, "learned-local-control")
        first = head._interaction_scores(z, active)
        altered = z.clone()
        altered[:, 1:, :] *= 7.0
        second = head._interaction_scores(altered, active)
        torch.testing.assert_close(first[:, 0], second[:, 0], rtol=0.0, atol=0.0)

    def test_cross_candidate_modes_really_depend_on_other_ranks(self):
        z = torch.tensor([[[1.0, 0.5, -0.2, 0.3], [0.4, -0.1, 0.8, 0.2], [0.2, 0.3, 0.1, -0.5], [0.6, 0.7, -0.4, 0.1], [0.9, -0.2, 0.2, 0.4]]])
        active = torch.ones((1, 5))
        altered = z.clone()
        altered[:, 4, :] *= -3.0
        changed = []
        for mode in ("learned-allpairs-cross", "learned-competitive-cross"):
            head = run.LearnedInteractionAmbiguityHead(1291, mode)
            a = head._interaction_scores(z, active)
            b = head._interaction_scores(altered, active)
            changed.append(not torch.equal(a[:, 0], b[:, 0]))
        self.assertEqual(changed, [True, True])

    def test_operation_only_calibration_and_holdout_domains_are_fresh(self):
        public, refs = self._dataset()
        examples = run.ambiguity_curriculum_examples("operation", public, refs)
        training_clauses = {item[1] for item in examples}
        holdout_phrases = {phrase for _, phrase in run.HOLDOUT_AMBIGUITIES}
        calibration_phrases = {phrase for _, phrase in run.CALIBRATION_AMBIGUITIES}
        self.assertTrue(training_clauses.isdisjoint(holdout_phrases))
        self.assertTrue(training_clauses.isdisjoint(calibration_phrases))
        calibration_rows, _ = run.calibration_set("operation")
        holdout_rows, holdout_refs = run.fresh_holdout("operation")
        roots = {row["catalog"]["root"] for row in calibration_rows + holdout_rows}
        prior_training_roots = {root for _, root in g56.TRAINING_ONLY_DOMAINS}
        self.assertTrue(prior_training_roots.isdisjoint(roots))
        ambiguous = [row for row in holdout_rows if holdout_refs[row["id"]]["status"] == "AMBIGUOUS"]
        self.assertEqual(len(ambiguous), 16)
        families = {row.get("ambiguity_family") for row in ambiguous}
        self.assertEqual(families, {"role", "lifecycle-time", "representation", "object-vs-supplier"})
        with self.assertRaises(RuntimeError):
            run.fresh_holdout("schema")

    def test_receipts_bind_canonical_gdm62(self):
        self.assertEqual(run.CANONICAL_GDM62_SOURCE, "a3cfa70a570f931e2c78553ca9d72aeb5ed65e79")
        self.assertEqual(run.CANONICAL_GDM62_AUDIT_RUN, 35965921793)
        self.assertEqual(run.CANONICAL_GDM62_PROMOTION, "7d925c87f80ecb69bf2ea38bfac9960faf08f9cd")


if __name__ == "__main__":
    unittest.main()
