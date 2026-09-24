from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from experiments.followup46.run import corrected_dataset
from experiments.followup51 import run as g51
from experiments.followup56 import run as g56
from experiments.followup59 import run as g59
from experiments.followup64 import run


class GDM64Contracts(unittest.TestCase):
    def _dataset(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        public, refs, _ = corrected_dataset(Path(temp.name) / "dataset")
        return public, refs

    def test_exact_operation_only_ten_configuration_matrix(self):
        self.assertEqual(len(run.ARMS), 5)
        self.assertEqual(len(run.ARMS) * 2, 10)
        self.assertEqual(tuple(run.ARMS), run.REPRESENTATION_MODES)
        self.assertEqual(run.TOPK, 5)
        self.assertEqual(run.SEMANTIC_BLOCKS, 10)
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
        expected = g56.ambiguity_curriculum_examples("operation", public, refs, "family-balanced")
        actual = run.ambiguity_curriculum_examples("operation", public, refs)
        self.assertEqual(actual, expected)
        receipt = g56._curriculum_receipt(actual, "family-balanced")
        counts = receipt["gdm56_positive_family_counts"]
        self.assertEqual(set(counts), {"role", "lifecycle-time", "representation", "object-vs-supplier"})
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertEqual(receipt["gdm56_counterfactual_negative_fraction"], 0.0)
        with self.assertRaises(RuntimeError):
            run.ambiguity_curriculum_examples("schema", public, refs)

    def test_fixed_dimensions_and_identical_ambiguity_head_capacity(self):
        self.assertEqual(run.ambiguity_feature_dim(128), 1291)
        self.assertEqual(run.ambiguity_feature_dim(768), 7691)
        for dim in (128, 768):
            feature_dim = run.ambiguity_feature_dim(dim)
            counts = []
            for _mode in run.REPRESENTATION_MODES:
                head = g51.AmbiguityHead(feature_dim, "mlp")
                counts.append(sum(p.numel() for p in head.parameters()))
            self.assertEqual(len(set(counts)), 1)

    def test_whole_control_is_coordinate_identical_to_canonical_ranked_request(self):
        q = torch.tensor([0.2, -0.4])
        whole = torch.tensor([
            [0.5, 0.1], [0.1, -0.2], [-0.3, 0.7], [0.8, -0.5], [0.4, 0.6]
        ])
        parent = whole * 3.0
        leaf = whole * -2.0
        first, second, effective = run._factor_blocks(q, whole, parent, leaf, "whole-request-control")
        canonical = g59._ranked_blocks(q, whole, "ranked-request-only")
        self.assertEqual(effective, 5)
        torch.testing.assert_close(first, canonical[0], rtol=0.0, atol=0.0)
        torch.testing.assert_close(second, canonical[1], rtol=0.0, atol=0.0)

    def test_coordinate_views_are_path_only_and_factor_role_vs_leaf(self):
        author_name = {"path": ["folio", "reviews", "author", "name"]}
        moderator_name = {"path": ["casebook", "reviews", "moderator", "name"]}
        author_id = {"path": ["casebook", "reviews", "author", "id"]}
        a_parent, a_leaf = run._coordinate_texts(author_name)
        m_parent, m_leaf = run._coordinate_texts(moderator_name)
        i_parent, i_leaf = run._coordinate_texts(author_id)
        self.assertEqual(a_parent, "GraphQL parent path: reviews.author")
        self.assertEqual(m_parent, "GraphQL parent path: reviews.moderator")
        self.assertEqual(i_parent, "GraphQL parent path: reviews.author")
        self.assertNotEqual(a_parent, m_parent)
        self.assertEqual(a_leaf, m_leaf)
        self.assertNotEqual(a_leaf, i_leaf)
        # Root/domain names are intentionally excluded; equal schema coordinates give equal views.
        other_root = {"path": ["different", "reviews", "author", "name"]}
        self.assertEqual(run._coordinate_texts(other_root), (a_parent, a_leaf))
        root_field = {"path": ["folio", "createdAt"]}
        self.assertEqual(run._coordinate_texts(root_field), ("GraphQL parent path: <root>", "GraphQL leaf field: createdAt"))

    def test_factorized_blocks_preserve_rank_and_zero_padding(self):
        q = torch.tensor([0.25, -0.5])
        whole = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        parent = torch.tensor([[0.2, 0.4], [0.6, 0.8]])
        leaf = torch.tensor([[0.3, -0.1], [-0.2, 0.9]])
        for mode in run.REPRESENTATION_MODES:
            first, second, effective = run._factor_blocks(q, whole, parent, leaf, mode)
            self.assertEqual(effective, 2)
            self.assertEqual(first.numel(), 10)
            self.assertEqual(second.numel(), 10)
            self.assertTrue(torch.equal(first[4:], torch.zeros_like(first[4:])))
            self.assertTrue(torch.equal(second[4:], torch.zeros_like(second[4:])))

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
            run.calibration_set("schema")
        with self.assertRaises(RuntimeError):
            run.fresh_holdout("schema")

    def test_receipts_bind_canonical_gdm63(self):
        self.assertEqual(run.CANONICAL_GDM63_SOURCE, "43a6d16652a3fd63c155f46e99b42c757c7601a1")
        self.assertEqual(run.CANONICAL_GDM63_AUDIT_RUN, 35988393672)
        self.assertEqual(run.CANONICAL_GDM63_PROMOTION, "4efc41f161fadd59f563f491f251ab6e058fcb71")


if __name__ == "__main__":
    unittest.main()
