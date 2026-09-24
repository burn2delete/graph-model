from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from experiments.followup46.run import corrected_dataset
from experiments.followup56 import run as g56
from experiments.followup59 import run as g59
from experiments.followup62 import run


class GDM62Contracts(unittest.TestCase):
    def _dataset(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        public, refs, _ = corrected_dataset(Path(temp.name) / "dataset")
        return public, refs

    def _vectors(self):
        q = torch.tensor([0.2, -0.4])
        candidates = torch.tensor([
            [0.5, 0.1], [0.1, -0.2], [-0.3, 0.7], [0.8, -0.5], [0.4, 0.6]
        ])
        weights = torch.tensor([0.50, 0.25, 0.15, 0.07, 0.03])
        return q, candidates, weights

    def test_exact_twenty_configuration_matrix(self):
        self.assertEqual(len(run.ARMS), 5)
        self.assertEqual(len(run.ARMS) * 2 * 2, 20)
        self.assertEqual(set(run.ARMS), set(run.REPRESENTATION_MODES))
        self.assertEqual(run.TOPK, 5)
        self.assertEqual(run.SEMANTIC_BLOCKS, 10)
        self.assertEqual(
            run.BLOCK_KINDS,
            {
                "ranked-request-control": ("request-product", "request-delta"),
                "ranked-product-candidate-delta": ("request-product", "candidate-delta"),
                "ranked-delta-candidate-delta": ("request-delta", "candidate-delta"),
                "ranked-product-candidate-product": ("request-product", "candidate-product"),
                "ranked-delta-candidate-product": ("request-delta", "candidate-product"),
            },
        )
        for arm_name, arm in run.ARMS.items():
            self.assertEqual(arm["family"], "explicit-none-plus-ambiguity")
            self.assertEqual(arm["head"], "mlp")
            self.assertEqual(arm["structured"], arm_name)
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

    def test_matched_capacity_dimension_contract(self):
        self.assertEqual(run.ambiguity_feature_dim(128), 1291)
        self.assertEqual(run.ambiguity_feature_dim(768), 7691)

    def test_request_control_exactly_matches_canonical_gdm59_blocks(self):
        q, candidates, weights = self._vectors()
        left = run._relation_blocks(q, candidates, weights, "ranked-request-control")
        right = g59._ranked_blocks(q, candidates, "ranked-request-only")
        torch.testing.assert_close(left[0], right[0], rtol=0.0, atol=0.0)
        torch.testing.assert_close(left[1], right[1], rtol=0.0, atol=0.0)

    def test_candidate_delta_is_rank1_anchored_nonlinear_relation(self):
        q, candidates, weights = self._vectors()
        _, candidate_delta = run._relation_blocks(q, candidates, weights, "ranked-product-candidate-delta")
        expected = (candidates - candidates[0:1]).abs().reshape(-1)
        torch.testing.assert_close(candidate_delta, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(candidate_delta[: q.numel()], torch.zeros_like(q), rtol=0.0, atol=0.0)
        request_delta = (candidates - q[None, :]).abs().reshape(-1)
        self.assertFalse(torch.equal(candidate_delta, request_delta))

    def test_candidate_product_is_rank1_anchored_nonlinear_relation(self):
        q, candidates, weights = self._vectors()
        _, candidate_product = run._relation_blocks(q, candidates, weights, "ranked-product-candidate-product")
        expected = (candidates * candidates[0:1]).reshape(-1)
        torch.testing.assert_close(candidate_product, expected, rtol=0.0, atol=0.0)
        self.assertFalse(torch.equal(candidate_product, (candidates * q[None, :]).reshape(-1)))

    def test_pairing_arms_change_only_declared_block_semantics(self):
        q, candidates, weights = self._vectors()
        product_cd = run._relation_blocks(q, candidates, weights, "ranked-product-candidate-delta")
        delta_cd = run._relation_blocks(q, candidates, weights, "ranked-delta-candidate-delta")
        product_cp = run._relation_blocks(q, candidates, weights, "ranked-product-candidate-product")
        delta_cp = run._relation_blocks(q, candidates, weights, "ranked-delta-candidate-product")
        torch.testing.assert_close(product_cd[1], delta_cd[1], rtol=0.0, atol=0.0)
        torch.testing.assert_close(product_cp[1], delta_cp[1], rtol=0.0, atol=0.0)
        torch.testing.assert_close(product_cd[0], product_cp[0], rtol=0.0, atol=0.0)
        torch.testing.assert_close(delta_cd[0], delta_cp[0], rtol=0.0, atol=0.0)
        self.assertEqual(product_cd[0].numel() + product_cd[1].numel(), 20)

    def test_receipts_bind_canonical_gdm61_and_fixed_top5(self):
        for mode in run.REPRESENTATION_MODES:
            receipt = run._representation_receipt(mode, 128)
            self.assertEqual(receipt["gdm62_requested_topk"], 5)
            self.assertEqual(receipt["gdm62_ambiguity_feature_dim"], 1291)
            self.assertTrue(receipt["gdm62_rank_preserving"])
            self.assertEqual(receipt["gdm62_candidate_anchor"], "rank1-real-candidate")
            self.assertEqual(receipt["canonical_gdm61_source_commit"], "f22fc36caf7327ed3d0d1f7dd2898ea31d837f3a")
            self.assertEqual(receipt["canonical_gdm61_audit_run"], 35948010054)
            self.assertEqual(
                (receipt["gdm62_first_block"], receipt["gdm62_second_block"]),
                run.BLOCK_KINDS[mode],
            )

    def test_fresh_holdout_has_four_labeled_ambiguity_families(self):
        rows, refs = run.fresh_holdout("schema")
        ambiguous = [row for row in rows if refs[row["id"]]["status"] == "AMBIGUOUS"]
        self.assertEqual(len(ambiguous), 16)
        families = {row.get("ambiguity_family") for row in ambiguous}
        self.assertEqual(families, {"role", "lifecycle-time", "representation", "object-vs-supplier"})
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
        evaluation_roots = {row["catalog"]["root"] for row in calibration_rows + holdout_rows}
        prior_training_roots = {root for _, root in g56.TRAINING_ONLY_DOMAINS}
        self.assertTrue(prior_training_roots.isdisjoint(evaluation_roots))


if __name__ == "__main__":
    unittest.main()
