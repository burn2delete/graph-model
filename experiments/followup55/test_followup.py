from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.followup46.run import corrected_dataset
from experiments.followup55 import run


class GDM55Contracts(unittest.TestCase):
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

    def test_canonical_control_is_exact_original_curriculum(self):
        public, refs = self._dataset()
        original = run._ORIGINAL_AMBIGUITY_TRAINING_EXAMPLES("operation", public, refs)
        control = run.ambiguity_curriculum_examples(
            "operation", public, refs, "canonical"
        )
        self.assertEqual(control, original)

    def test_expanded_arms_are_update_budget_matched(self):
        public, refs = self._dataset()
        curricula = {}
        for name in (
            "volume-matched",
            "lexical-expanded",
            "relation-expanded",
            "full-curriculum",
        ):
            examples = run.ambiguity_curriculum_examples("schema", public, refs, name)
            curricula[name] = (
                len(examples),
                sum(int(item[2]) == 1 for item in examples),
                sum(int(item[2]) == 0 for item in examples),
            )
        self.assertEqual(len(set(curricula.values())), 1, curricula)

    def test_semantic_diversity_exceeds_volume_only_control(self):
        public, refs = self._dataset()
        examples = {
            name: run.ambiguity_curriculum_examples("operation", public, refs, name)
            for name in (
                "volume-matched",
                "lexical-expanded",
                "relation-expanded",
                "full-curriculum",
            )
        }
        unique_positive = {
            name: len({item[1] for item in values if int(item[2]) == 1})
            for name, values in examples.items()
        }
        self.assertGreater(
            unique_positive["lexical-expanded"], unique_positive["volume-matched"]
        )
        self.assertGreater(
            unique_positive["relation-expanded"], unique_positive["volume-matched"]
        )
        self.assertGreater(
            unique_positive["full-curriculum"], unique_positive["volume-matched"]
        )
        full_sources = {item[4] for item in examples["full-curriculum"]}
        for token in ("lexical", "relation", "counterfactual", "domain-randomized"):
            self.assertTrue(any(token in source for source in full_sources), full_sources)

    def test_fresh_holdout_has_four_labeled_ambiguity_families(self):
        rows, refs = run.fresh_holdout("schema")
        ambiguous = [row for row in rows if refs[row["id"]]["status"] == "AMBIGUOUS"]
        self.assertEqual(len(ambiguous), 8)
        families = {row.get("ambiguity_family") for row in ambiguous}
        self.assertEqual(
            families,
            {"role", "lifecycle-time", "representation", "object-vs-supplier"},
        )
        counts = {family: 0 for family in families}
        for row in ambiguous:
            counts[row["ambiguity_family"]] += 1
        self.assertEqual(set(counts.values()), {2})

    def test_holdout_phrases_and_domains_never_enter_curriculum(self):
        public, refs = self._dataset()
        examples = run.ambiguity_curriculum_examples(
            "operation", public, refs, "full-curriculum"
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

    def test_full_curriculum_contains_matched_counterfactual_negatives(self):
        public, refs = self._dataset()
        examples = run.ambiguity_curriculum_examples(
            "schema", public, refs, "full-curriculum"
        )
        negatives = [item for item in examples if int(item[2]) == 0]
        self.assertTrue(
            any("counterfactual" in item[4] for item in negatives),
            {item[4] for item in negatives},
        )
        self.assertTrue(
            all(item[3] is not None for item in negatives if "counterfactual" in item[4])
        )


if __name__ == "__main__":
    unittest.main()
