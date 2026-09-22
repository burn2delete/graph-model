from __future__ import annotations

import unittest

from experiments.followup53 import run as g53


class GDM53Contracts(unittest.TestCase):
    def test_matrix_is_bounded_and_structured(self):
        self.assertEqual(len(g53.ARMS), 5)
        self.assertEqual(g53.SEEDS_DEFAULT, "5301,5302")
        self.assertEqual(
            set(g53.ARMS),
            {
                "joint-structured-control",
                "sequential-structured-control",
                "budget-structured-5pp",
                "budget-structured-10pp",
                "budget-structured-15pp",
            },
        )
        for arm in g53.ARMS.values():
            self.assertTrue(arm["structured"])
            self.assertEqual(arm["family"], "explicit-none-plus-ambiguity")

    def test_fresh_calibration_and_holdout_are_disjoint(self):
        for task in ("operation", "schema"):
            cal_rows, cal_refs = g53.calibration_set(task)
            hold_rows, hold_refs = g53.fresh_holdout(task)
            cal_ids = {row["id"] for row in cal_rows}
            hold_ids = {row["id"] for row in hold_rows}
            self.assertTrue(cal_ids)
            self.assertTrue(hold_ids)
            self.assertTrue(cal_ids.isdisjoint(hold_ids))
            self.assertEqual(cal_ids, set(cal_refs))
            self.assertEqual(hold_ids, set(hold_refs))
            self.assertTrue(all("repository" in row["catalog"]["root"] or "deployment" in row["catalog"]["root"] for row in cal_rows))
            self.assertTrue(all("order" in row["catalog"]["root"] or "payment" in row["catalog"]["root"] for row in hold_rows))

    def test_budget_floor_is_derived_only_from_joint_validation_baseline(self):
        records = [
            {"reference": {"status": "accepted", "paths": ["a"]}, "evidence": [{"best_real_id": "a", "none_margin": -2.0, "ambiguity_probability": 0.02}]},
            {"reference": {"status": "accepted", "paths": ["b"]}, "evidence": [{"best_real_id": "b", "none_margin": -1.4, "ambiguity_probability": 0.08}]},
            {"reference": {"status": "accepted", "paths": ["c"]}, "evidence": [{"best_real_id": "c", "none_margin": -0.8, "ambiguity_probability": 0.15}]},
            {"reference": {"status": "accepted", "paths": ["d"]}, "evidence": [{"best_real_id": "d", "none_margin": -0.2, "ambiguity_probability": 0.25}]},
            {"reference": {"status": "NO_MATCH", "paths": []}, "evidence": [{"best_real_id": "x", "none_margin": 0.4, "ambiguity_probability": 0.05}]},
            {"reference": {"status": "NO_MATCH", "paths": []}, "evidence": [{"best_real_id": "y", "none_margin": 0.9, "ambiguity_probability": 0.10}]},
            {"reference": {"status": "AMBIGUOUS", "paths": []}, "evidence": [{"best_real_id": "m", "none_margin": -0.6, "ambiguity_probability": 0.72}]},
            {"reference": {"status": "AMBIGUOUS", "paths": []}, "evidence": [{"best_real_id": "n", "none_margin": -0.1, "ambiguity_probability": 0.88}]},
        ]
        for budget_pp in (5, 10, 15):
            calibration, metrics = g53.calibrate_budget(records, budget_pp)
            baseline = calibration["joint_baseline_validation_metrics"]
            expected_floor = max(0.0, baseline["accepted_status_recall"] - budget_pp / 100.0)
            self.assertAlmostEqual(calibration["accepted_status_recall_floor"], expected_floor)
            self.assertGreaterEqual(metrics["accepted_status_recall"] + 1e-12, expected_floor)
            self.assertEqual(calibration["selection_split"], "validation")
            self.assertEqual(calibration["threshold_selection"], "accepted-recall-budget")

    def test_budget_values_are_explicit_architecture_inputs(self):
        self.assertEqual(g53.ARMS["budget-structured-5pp"]["accepted_recall_budget_pp"], 5)
        self.assertEqual(g53.ARMS["budget-structured-10pp"]["accepted_recall_budget_pp"], 10)
        self.assertEqual(g53.ARMS["budget-structured-15pp"]["accepted_recall_budget_pp"], 15)
        self.assertEqual(g53.ARMS["joint-structured-control"]["accepted_recall_budget_pp"], 0)
        self.assertIsNone(g53.ARMS["sequential-structured-control"]["accepted_recall_budget_pp"])


if __name__ == "__main__":
    unittest.main()
