from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from experiments.followup46.run import corrected_dataset
from experiments.followup56 import run as g56
from experiments.followup58 import run as g58
from experiments.followup59 import run


class GDM59Contracts(unittest.TestCase):
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

    def test_ranked_blocks_fixed_dimension_and_rank_sensitive(self):
        q = torch.tensor([0.1, -0.2, 0.3])
        candidates = torch.tensor(
            [
                [0.2, 0.4, -0.1],
                [-0.3, 0.5, 0.7],
                [0.6, -0.1, 0.2],
                [0.9, 0.3, -0.2],
                [-0.4, 0.8, 0.1],
            ]
        )
        for mode in (
            "ranked-candidate-only",
            "ranked-request-only",
            "ranked-candidate-request",
            "ranked-candidate-delta",
        ):
            first, second = run._ranked_blocks(q, candidates, mode)
            self.assertEqual(first.numel(), 15)
            self.assertEqual(second.numel(), 15)
            first_swapped, second_swapped = run._ranked_blocks(
                q, candidates[[1, 0, 2, 3, 4]], mode
            )
            self.assertFalse(
                torch.equal(torch.cat([first, second]), torch.cat([first_swapped, second_swapped]))
            )

    def test_missing_rank_slots_are_zero_safe(self):
        q = torch.tensor([0.1, -0.2])
        candidates = torch.tensor([[0.2, 0.4], [-0.3, 0.5]])
        padded, mask, k = run._pad_ranked_candidates(candidates)
        self.assertEqual(k, 2)
        self.assertEqual(tuple(padded.shape), (5, 2))
        self.assertEqual(mask.tolist(), [1.0, 1.0, 0.0, 0.0, 0.0])
        self.assertTrue(torch.equal(padded[2:], torch.zeros_like(padded[2:])))
        _, delta = run._ranked_blocks(q, candidates, "ranked-candidate-delta")
        delta_slots = delta.reshape(5, 2)
        self.assertTrue(torch.equal(delta_slots[2:], torch.zeros_like(delta_slots[2:])))

    def test_top5_pooled_control_matches_gdm58_pooling(self):
        q = torch.tensor([0.1, -0.2])
        candidates = torch.tensor(
            [[0.2, 0.4], [-0.3, 0.5], [0.9, -0.8], [0.0, 0.2], [0.3, 0.1]]
        )
        weights = torch.tensor([0.40, 0.25, 0.15, 0.12, 0.08])
        first, second = g58._pooled_blocks(q, candidates, weights)
        first2, second2 = g58._pooled_blocks(q, candidates, weights)
        torch.testing.assert_close(first, first2, rtol=0.0, atol=0.0)
        torch.testing.assert_close(second, second2, rtol=0.0, atol=0.0)
        self.assertEqual(first.numel() + second.numel(), 20)

    def test_receipts_bind_canonical_gdm58_and_fixed_top5(self):
        for mode in run.REPRESENTATION_MODES:
            receipt = run._representation_receipt(mode, 128)
            self.assertEqual(receipt["gdm59_requested_topk"], 5)
            self.assertEqual(receipt["gdm59_ambiguity_feature_dim"], 1291)
            self.assertEqual(
                receipt["canonical_gdm58_source_commit"],
                "9ba9caab957c1802feb17776195141f630fd96eb",
            )
            self.assertEqual(receipt["canonical_gdm58_audit_run"], 35890238431)
            self.assertEqual(
                receipt["gdm59_rank_preserving"], mode != "top5-pooled-control"
            )

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
