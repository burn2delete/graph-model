import os
import unittest
from unittest import mock

from experiments.followup47.run import (
    ARMS, CURRICULUM, FRESH_LANGUAGE, SEMANTIC_SUFFIXES,
    calibrate, fresh_holdout, semantic_curriculum, status_from_probs,
)
from experiments.followup46.run import corrected_dataset, fresh_holdout as gdm46_holdout


class GDM47Contracts(unittest.TestCase):
    def test_batch_is_bounded(self):
        self.assertEqual(set(ARMS), {
            "control", "calibrated", "curriculum-calibrated",
            "curriculum-scoreonly", "curriculum-nohardneg",
        })
        self.assertEqual(len(ARMS), 5)
        self.assertTrue(all(a["calibration"] in {"argmax", "thresholds"} for a in ARMS.values()))
        self.assertEqual(ARMS["curriculum-scoreonly"]["status_features"], "scores-only")

    def test_holdout_is_fresh_and_task_consistent(self):
        for task, prefix in [("operation", "Return "), ("schema", "Expose capabilities for ")]:
            rows, refs = fresh_holdout(task)
            old_rows, _ = gdm46_holdout(task)
            self.assertTrue(rows)
            self.assertTrue(all(r["request"].startswith(prefix) for r in rows))
            self.assertTrue({refs[r["id"]]["status"] for r in rows} >= {"accepted", "NO_MATCH", "AMBIGUOUS"})
            self.assertTrue(set(r["id"] for r in rows).isdisjoint(r["id"] for r in old_rows))
            # Every capability is exercised as an unsupported/missing case for each of two domains.
            self.assertEqual(sum(refs[r["id"]]["status"] == "NO_MATCH" for r in rows), 16)

    def test_training_curriculum_does_not_copy_fresh_holdout_phrasing(self):
        curriculum_text = " ".join(x for phrases in CURRICULUM.values() for x in phrases).lower()
        for phrase in FRESH_LANGUAGE:
            self.assertNotIn(phrase.lower(), curriculum_text)
        self.assertEqual(set(CURRICULUM), set(SEMANTIC_SUFFIXES))

    def test_curriculum_uses_training_catalogs_and_single_targets(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            public, refs, _ = corrected_dataset(td)
            rows = [r for r in public["train"] if r["task"] == "operation"]
            examples = semantic_curriculum("operation", rows)
            self.assertTrue(examples)
            for row, ref in examples:
                self.assertEqual(ref["status"], "accepted")
                self.assertEqual(len(ref["paths"]), 1)
                self.assertTrue(row["request"].startswith("Return "))

    def test_calibration_is_validation_only_and_non_degenerate(self):
        records = [
            {"reference": {"status": "accepted", "paths": ["x"]}, "probs": [0.45, 0.40, 0.15], "selected": ["x"]},
            {"reference": {"status": "accepted", "paths": ["x"]}, "probs": [0.42, 0.41, 0.17], "selected": ["x"]},
            {"reference": {"status": "NO_MATCH", "paths": []}, "probs": [0.10, 0.80, 0.10], "selected": ["x"]},
            {"reference": {"status": "AMBIGUOUS", "paths": []}, "probs": [0.10, 0.20, 0.70], "selected": ["x"]},
        ]
        calibration, metrics = calibrate(records, "thresholds")
        self.assertEqual(calibration["mode"], "thresholds")
        self.assertEqual(calibration["selection_split"], "validation")
        self.assertGreater(metrics["selective_hmean"], 0.0)
        self.assertEqual(status_from_probs([0.1, 0.8, 0.1], calibration), "NO_MATCH")
        self.assertEqual(status_from_probs([0.1, 0.1, 0.8], calibration), "AMBIGUOUS")

    def test_execution_guard_requires_actions(self):
        # The executable module contains an explicit GITHUB_ACTIONS guard; keep it from
        # silently becoming a local benchmark entrypoint.
        from pathlib import Path
        text = Path(__file__).with_name("run.py").read_text()
        self.assertIn('os.environ.get("GITHUB_ACTIONS") != "true"', text)
        self.assertIn("Project execution belongs in GitHub Actions", text)


if __name__ == "__main__":
    unittest.main()
