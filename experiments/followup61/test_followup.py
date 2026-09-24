from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from experiments.followup46.run import corrected_dataset
from experiments.followup56 import run as g56
from experiments.followup59 import run as g59
from experiments.followup60 import run as g60
from experiments.followup61 import run


class GDM61Contracts(unittest.TestCase):
    def _dataset(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        public, refs, _ = corrected_dataset(Path(temp.name) / "dataset")
        return public, refs

    def test_exact_twenty_configuration_matrix(self):
        self.assertEqual(len(run.ARMS), 5)
        self.assertEqual(len(run.ARMS) * 2 * 2, 20)
        self.assertEqual(set(run.ARMS), set(run.REPRESENTATION_MODES))
        self.assertEqual(run.TOPK, 5)
        self.assertEqual(
            run.BLOCK_SCALES,
            {
                "ranked-product-only": (1.0, 0.0),
                "ranked-product-dominant": (1.0, 0.5),
                "ranked-balanced-control": (1.0, 1.0),
                "ranked-delta-dominant": (0.5, 1.0),
                "ranked-delta-only": (0.0, 1.0),
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
        self.assertEqual(run.SEMANTIC_BLOCKS, 10)

    def test_balanced_control_exactly_matches_canonical_gdm59_request_blocks(self):
        q = torch.tensor([0.1, -0.2, 0.3])
        candidates = torch.tensor([
            [0.2, 0.4, -0.1], [-0.3, 0.5, 0.7], [0.6, -0.1, 0.2],
            [0.9, 0.3, -0.2], [-0.4, 0.8, 0.1],
        ])
        weights = torch.tensor([0.40, 0.25, 0.15, 0.12, 0.08])
        first, second = run._balance_blocks(q, candidates, weights, "ranked-balanced-control")
        expected_first, expected_second = g59._ranked_blocks(q, candidates, "ranked-request-only")
        torch.testing.assert_close(first, expected_first, rtol=0.0, atol=0.0)
        torch.testing.assert_close(second, expected_second, rtol=0.0, atol=0.0)

    def test_endpoint_arms_preserve_gdm60_factorized_content_with_fixed_block_slots(self):
        q = torch.tensor([0.2, -0.4])
        candidates = torch.tensor([[0.5, 0.1], [0.1, -0.2], [-0.3, 0.7]])
        weights = torch.tensor([0.6, 0.25, 0.15])

        # Product-only already occupies the product slot in GDM60, so the endpoint is
        # coordinate-identical as well as semantically identical.
        product_left = run._balance_blocks(q, candidates, weights, "ranked-product-only")
        product_right = g60._confidence_blocks(q, candidates, weights, "ranked-product-only")
        torch.testing.assert_close(product_left[0], product_right[0], rtol=0.0, atol=0.0)
        torch.testing.assert_close(product_left[1], product_right[1], rtol=0.0, atol=0.0)

        # GDM60's diagnostic delta-only arm packed delta evidence into its first generic
        # slot. GDM61 deliberately keeps product and delta in fixed slots across the
        # whole interpolation family, so delta-only must preserve the exact GDM60 delta
        # values while placing them in the dedicated second (delta) slot. This is a
        # contract-test correction only; the GDM61 model representation is unchanged.
        delta_left = run._balance_blocks(q, candidates, weights, "ranked-delta-only")
        delta_right = g60._confidence_blocks(q, candidates, weights, "ranked-delta-only")
        zeros = torch.zeros_like(delta_right[0])
        torch.testing.assert_close(delta_left[0], zeros, rtol=0.0, atol=0.0)
        torch.testing.assert_close(delta_left[1], delta_right[0], rtol=0.0, atol=0.0)
        torch.testing.assert_close(delta_right[1], zeros, rtol=0.0, atol=0.0)

    def test_intermediate_arms_scale_only_declared_block(self):
        q = torch.tensor([0.2, -0.4])
        candidates = torch.tensor([[0.5, 0.1], [0.1, -0.2], [-0.3, 0.7], [0.8, -0.5], [0.4, 0.6]])
        weights = torch.tensor([0.50, 0.25, 0.15, 0.07, 0.03])
        product, delta = run._balance_blocks(q, candidates, weights, "ranked-balanced-control")
        pd_product, pd_delta = run._balance_blocks(q, candidates, weights, "ranked-product-dominant")
        dd_product, dd_delta = run._balance_blocks(q, candidates, weights, "ranked-delta-dominant")
        torch.testing.assert_close(pd_product, product, rtol=0.0, atol=0.0)
        torch.testing.assert_close(pd_delta, delta * 0.5, rtol=0.0, atol=0.0)
        torch.testing.assert_close(dd_product, product * 0.5, rtol=0.0, atol=0.0)
        torch.testing.assert_close(dd_delta, delta, rtol=0.0, atol=0.0)
        self.assertEqual(product.numel() + delta.numel(), 20)

    def test_receipts_bind_canonical_gdm60_and_fixed_top5(self):
        for mode in run.REPRESENTATION_MODES:
            receipt = run._representation_receipt(mode, 128)
            self.assertEqual(receipt["gdm61_requested_topk"], 5)
            self.assertEqual(receipt["gdm61_ambiguity_feature_dim"], 1291)
            self.assertTrue(receipt["gdm61_rank_preserving"])
            self.assertEqual(receipt["canonical_gdm60_source_commit"], "c04784ec33983829f6fe55506bcb74e0f40442b2")
            self.assertEqual(receipt["canonical_gdm60_audit_run"], 35929537000)
            self.assertEqual(
                (receipt["gdm61_product_scale"], receipt["gdm61_delta_scale"]),
                run.BLOCK_SCALES[mode],
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
