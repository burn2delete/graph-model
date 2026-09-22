from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.followup46.run import corrected_dataset
from experiments.followup55 import run as g55
from experiments.followup56 import run


class GDM56Contracts(unittest.TestCase):
    def _dataset(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        public, refs, _ = corrected_dataset(Path(temp.name) / "dataset")
        return public, refs

    def test_exact_twenty_configuration_matrix(self):
        self.assertEqual(len(run.ARMS), 5)
        self.assertEqual(len(run.ARMS) * 2 * 2, 20)
        for arm in run.ARMS.values():
            self.assertEqual(arm["family"], "explicit-none-plus-ambiguity")
            self.assertEqual(arm["head"], "mlp")
            self.assertTrue(arm["structured"])
            self.assertEqual(arm["calibration_strategy"], "ambiguity-rescue")
            self.assertEqual(arm["status_arbitration"], "ambiguity-rescue")
            self.assertEqual(arm["nomatch_recall_budget_pp"], 5)

    def test_relation_control_is_exact_gdm55_relation_expanded(self):
        public, refs = self._dataset()
        expected = g55.ambiguity_curriculum_examples(
            "operation", public, refs, "relation-expanded"
        )
        actual = run.ambiguity_curriculum_examples(
            "operation", public, refs, "relation-expanded-control"
        )
        self.assertEqual(actual, expected)

    def test_all_arms_match_relation_control_update_budget(self):
        public, refs = self._dataset()
        receipts = {}
        for name in arm_names():
            examples = run.ambiguity_curriculum_examples("schema", public, refs, name)
            receipts[name] = (
                len(examples),
                sum(int(item[2]) == 1 for item in examples),
                sum(int(item[2]) == 0 for item in examples),
            )
        self.assertEqual(len(set(receipts.values())), 1, receipts)

    def test_balanced_arms_equalize_four_positive_relation_families(self):
        public, refs = self._dataset()
        expected = {"role", "lifecycle-time", "representation", "object-vs-supplier"}
        for name in arm_names()[1:]:
            examples = run.ambiguity_curriculum_examples("operation", public, refs, name)
            receipt = run._curriculum_receipt(examples, name)
            counts = receipt["gdm56_positive_family_counts"]
            self.assertEqual(set(counts), expected)
            self.assertGreater(min(counts.values()), 0)
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_counterfactual_factor_is_isolated(self):
        public, refs = self._dataset()
        for name in arm_names()[1:]:
            examples = run.ambiguity_curriculum_examples("operation", public, refs, name)
            negatives = [item for item in examples if int(item[2]) == 0]
            count = sum("counterfactual" in item[4] for item in negatives)
            wants = name in {
                "family-balanced-counterfactual",
                "family-balanced-counterfactual-domain",
            }
            self.assertEqual(count > 0, wants, (name, count))
            if wants:
                self.assertEqual(count, len(negatives) // 4)

    def test_domain_factor_is_isolated(self):
        public, refs = self._dataset()
        for name in arm_names()[1:]:
            examples = run.ambiguity_curriculum_examples("schema", public, refs, name)
            domain_count = sum("domain-positive" in item[4] for item in examples)
            wants = name in {
                "family-balanced-domain",
                "family-balanced-counterfactual-domain",
            }
            self.assertEqual(domain_count > 0, wants, (name, domain_count))

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

    def test_evaluation_phrases_and_domains_never_enter_curriculum(self):
        public, refs = self._dataset()
        examples = run.ambiguity_curriculum_examples(
            "operation", public, refs, "family-balanced-counterfactual-domain"
        )
        training_clauses = {item[1] for item in examples}
        holdout_phrases = {phrase for _, phrase in run.HOLDOUT_AMBIGUITIES}
        calibration_phrases = {phrase for _, phrase in run.CALIBRATION_AMBIGUITIES}
        self.assertTrue(training_clauses.isdisjoint(holdout_phrases))
        self.assertTrue(training_clauses.isdisjoint(calibration_phrases))

        training_roots = {root for _, root in run.TRAINING_ONLY_DOMAINS}
        calibration_rows, _ = run.calibration_set("operation")
        holdout_rows, _ = run.fresh_holdout("operation")
        evaluation_roots = {
            row["catalog"]["root"] for row in calibration_rows + holdout_rows
        }
        self.assertTrue(training_roots.isdisjoint(evaluation_roots))


def arm_names():
    return [
        "relation-expanded-control",
        "family-balanced-rescue",
        "family-balanced-counterfactual",
        "family-balanced-domain",
        "family-balanced-counterfactual-domain",
    ]


if __name__ == "__main__":
    unittest.main()
