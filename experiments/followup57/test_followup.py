from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from experiments.followup46.run import corrected_dataset
from experiments.followup56 import run as g56
from experiments.followup57 import run


class GDM57Contracts(unittest.TestCase):
    def _dataset(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        public, refs, _ = corrected_dataset(Path(temp.name) / "dataset")
        return public, refs

    def test_exact_twenty_configuration_matrix(self):
        self.assertEqual(len(run.ARMS), 5)
        self.assertEqual(len(run.ARMS) * 2 * 2, 20)
        self.assertEqual(set(run.ARMS), set(run.REPRESENTATION_MODES))
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

    def test_representation_blocks_are_factorized_and_equal_length(self):
        q = torch.tensor([0.1, -0.2, 0.3])
        c1 = torch.tensor([0.2, 0.4, -0.1])
        c2 = torch.tensor([-0.3, 0.5, 0.7])
        blocks = {}
        for mode in run.REPRESENTATION_MODES:
            candidate, request = run._representation_blocks(q, c1, c2, mode)
            self.assertEqual(candidate.numel(), 15)
            self.assertEqual(request.numel(), 15)
            blocks[mode] = (candidate, request)

        self.assertEqual(torch.count_nonzero(blocks["scalar-control"][0]).item(), 0)
        self.assertEqual(torch.count_nonzero(blocks["scalar-control"][1]).item(), 0)
        self.assertGreater(torch.count_nonzero(blocks["candidate-directed"][0]).item(), 0)
        self.assertEqual(torch.count_nonzero(blocks["candidate-directed"][1]).item(), 0)
        self.assertEqual(torch.count_nonzero(blocks["request-conditioned"][0]).item(), 0)
        self.assertGreater(torch.count_nonzero(blocks["request-conditioned"][1]).item(), 0)
        self.assertGreater(torch.count_nonzero(blocks["candidate-request"][0]).item(), 0)
        self.assertGreater(torch.count_nonzero(blocks["candidate-request"][1]).item(), 0)

    def test_symmetric_candidate_block_is_order_invariant(self):
        q = torch.tensor([0.1, -0.2, 0.3])
        c1 = torch.tensor([0.2, 0.4, -0.1])
        c2 = torch.tensor([-0.3, 0.5, 0.7])
        a, _ = run._representation_blocks(q, c1, c2, "symmetric-request")
        b, _ = run._representation_blocks(q, c2, c1, "symmetric-request")
        self.assertTrue(torch.equal(a, b))
        directed_a, _ = run._representation_blocks(q, c1, c2, "candidate-directed")
        directed_b, _ = run._representation_blocks(q, c2, c1, "candidate-directed")
        self.assertFalse(torch.equal(directed_a, directed_b))

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
