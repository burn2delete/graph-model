from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from experiments.followup46.run import corrected_dataset
from experiments.followup56 import run as g56
from experiments.followup58 import run


class GDM58Contracts(unittest.TestCase):
    def _dataset(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        public, refs, _ = corrected_dataset(Path(temp.name) / "dataset")
        return public, refs

    def test_exact_twenty_configuration_matrix(self):
        self.assertEqual(len(run.ARMS), 5)
        self.assertEqual(len(run.ARMS) * 2 * 2, 20)
        self.assertEqual(set(run.ARMS), set(run.TOPK_MODES))
        self.assertEqual(
            run.TOPK_LIMITS,
            {
                "top2-pooled-control": 2,
                "top3-pooled": 3,
                "top4-pooled": 4,
                "top5-pooled": 5,
                "all-pooled": None,
            },
        )
        for arm_name, arm in run.ARMS.items():
            self.assertEqual(arm["family"], "explicit-none-plus-ambiguity")
            self.assertEqual(arm["head"], "mlp")
            self.assertEqual(arm["structured"], arm_name)
            self.assertEqual(arm["ambiguity_curriculum"], "family-balanced")
            self.assertEqual(arm["calibration_strategy"], "ambiguity-rescue")
            self.assertEqual(arm["status_arbitration"], "ambiguity-rescue")
            self.assertEqual(arm["nomatch_recall_budget_pp"], 5)

    def test_all_arms_use_exact_canonical_gdm56_balanced_curriculum(self):
        public, refs = self._dataset()
        expected = g56.ambiguity_curriculum_examples(
            "operation", public, refs, "family-balanced"
        )
        actual = run.ambiguity_curriculum_examples("operation", public, refs)
        self.assertEqual(actual, expected)
        receipt = g56._curriculum_receipt(actual, "family-balanced")
        self.assertEqual(receipt["gdm56_ambiguity_curriculum"], "family-balanced")
        counts = receipt["gdm56_positive_family_counts"]
        self.assertEqual(
            set(counts),
            {"role", "lifecycle-time", "representation", "object-vs-supplier"},
        )
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertEqual(receipt["gdm56_counterfactual_negative_fraction"], 0.0)

    def test_matched_capacity_dimension_contract(self):
        self.assertEqual(run.ambiguity_feature_dim(128), 1291)
        self.assertEqual(run.ambiguity_feature_dim(768), 7691)
        self.assertEqual(run.SEMANTIC_BLOCKS, 10)

    def test_topk_limit_contract(self):
        self.assertEqual(run._topk_limit("top2-pooled-control", 8), 2)
        self.assertEqual(run._topk_limit("top3-pooled", 8), 3)
        self.assertEqual(run._topk_limit("top4-pooled", 8), 4)
        self.assertEqual(run._topk_limit("top5-pooled", 8), 5)
        self.assertEqual(run._topk_limit("all-pooled", 8), 8)
        self.assertEqual(run._topk_limit("top5-pooled", 3), 3)

    def test_pooling_is_fixed_dimension_and_order_invariant(self):
        q = torch.tensor([0.1, -0.2, 0.3])
        candidates = torch.tensor(
            [
                [0.2, 0.4, -0.1],
                [-0.3, 0.5, 0.7],
                [0.6, -0.1, 0.2],
            ]
        )
        weights = torch.tensor([0.6, 0.3, 0.1])
        candidate, request = run._pooled_blocks(q, candidates, weights)
        self.assertEqual(candidate.numel(), 15)
        self.assertEqual(request.numel(), 15)

        perm = torch.tensor([2, 0, 1])
        candidate2, request2 = run._pooled_blocks(q, candidates[perm], weights[perm])
        # Floating-point reductions are mathematically permutation invariant but may
        # differ by a final rounding bit when the reduction order changes. The
        # architecture contract is semantic/order invariance, not bitwise equality
        # across a deliberately permuted reduction order.
        torch.testing.assert_close(candidate, candidate2, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(request, request2, rtol=1e-6, atol=1e-7)

    def test_more_candidates_can_change_pooled_evidence_without_changing_dimension(self):
        q = torch.tensor([0.1, -0.2])
        candidates = torch.tensor([[0.2, 0.4], [-0.3, 0.5], [0.9, -0.8]])
        weights = torch.tensor([0.6, 0.3, 0.1])
        c2, r2 = run._pooled_blocks(q, candidates[:2], weights[:2])
        c3, r3 = run._pooled_blocks(q, candidates, weights)
        self.assertEqual(c2.numel(), c3.numel())
        self.assertEqual(r2.numel(), r3.numel())
        self.assertFalse(torch.equal(c2, c3))
        self.assertFalse(torch.equal(r2, r3))

    def test_fresh_holdout_has_four_labeled_ambiguity_families(self):
        rows, refs = run.fresh_holdout("schema")
        ambiguous = [row for row in rows if refs[row["id"]]["status"] == "AMBIGUOUS"]
        self.assertEqual(len(ambiguous), 16)
        families = {row.get("ambiguity_family") for row in ambiguous}
        self.assertEqual(
            families,
            {"role", "lifecycle-time", "representation", "object-vs-supplier"},
        )
        counts = {family: 0 for family in families}
        for row in ambiguous:
            counts[row["ambiguity_family"]] += 1
        self.assertEqual(set(counts.values()), {4})

    def test_evaluation_phrases_and_domains_never_enter_fixed_curriculum(self):
        public, refs = self._dataset()
        examples = run.ambiguity_curriculum_examples("operation", public, refs)
        training_clauses = {item[1] for item in examples}
        holdout_phrases = {phrase for _, phrase in run.HOLDOUT_AMBIGUITIES}
        calibration_phrases = {phrase for _, phrase in run.CALIBRATION_AMBIGUITIES}
        self.assertTrue(training_clauses.isdisjoint(holdout_phrases))
        self.assertTrue(training_clauses.isdisjoint(calibration_phrases))

        calibration_rows, _ = run.calibration_set("operation")
        holdout_rows, _ = run.fresh_holdout("operation")
        evaluation_roots = {
            row["catalog"]["root"] for row in calibration_rows + holdout_rows
        }
        prior_training_roots = {root for _, root in g56.TRAINING_ONLY_DOMAINS}
        self.assertTrue(prior_training_roots.isdisjoint(evaluation_roots))


if __name__ == "__main__":
    unittest.main()
