from __future__ import annotations

import unittest

from experiments.followup54 import run as g54


class GDM54Contracts(unittest.TestCase):
    def test_matrix_is_bounded_and_structured(self):
        self.assertEqual(len(g54.ARMS), 5)
        self.assertEqual(g54.SEEDS_DEFAULT, "5401,5402")
        self.assertEqual(
            set(g54.ARMS),
            {
                "nomatch-first-control",
                "ambiguity-first-control",
                "ambiguity-rescue-0pp",
                "ambiguity-rescue-5pp",
                "ambiguity-rescue-10pp",
            },
        )
        for arm in g54.ARMS.values():
            self.assertTrue(arm["structured"])
            self.assertEqual(arm["family"], "explicit-none-plus-ambiguity")

    def test_fresh_calibration_and_holdout_are_disjoint(self):
        for task in ("operation", "schema"):
            cal_rows, cal_refs = g54.calibration_set(task)
            hold_rows, hold_refs = g54.fresh_holdout(task)
            cal_ids = {row["id"] for row in cal_rows}
            hold_ids = {row["id"] for row in hold_rows}
            self.assertTrue(cal_ids)
            self.assertTrue(hold_ids)
            self.assertTrue(cal_ids.isdisjoint(hold_ids))
            self.assertEqual(cal_ids, set(cal_refs))
            self.assertEqual(hold_ids, set(hold_refs))
            self.assertTrue(all(row["catalog"]["root"] in {"workspace", "build"} for row in cal_rows))
            self.assertTrue(all(row["catalog"]["root"] in {"ticket", "account"} for row in hold_rows))

    def test_ambiguity_first_can_preempt_none_for_same_evidence(self):
        evidence = [{
            "best_real_id": "a",
            "none_margin": 0.8,
            "ambiguity_probability": 0.95,
        }]
        base = {
            "none_margin_threshold": 0.2,
            "ambiguity_probability_threshold": 0.8,
            "selection_split": "validation",
        }
        nomatch = g54.request_from_evidence(
            evidence,
            {**base, "status_arbitration": "nomatch-first"},
        )
        ambiguity = g54.request_from_evidence(
            evidence,
            {**base, "status_arbitration": "ambiguity-first"},
        )
        self.assertEqual(nomatch["status"], "NO_MATCH")
        self.assertEqual(ambiguity["status"], "AMBIGUOUS")

    def test_rescue_calibration_preserves_declared_validation_floors(self):
        records = [
            {"reference": {"status": "accepted", "paths": ["a"]}, "evidence": [{"best_real_id": "a", "none_margin": -2.0, "ambiguity_probability": 0.05}]},
            {"reference": {"status": "accepted", "paths": ["b"]}, "evidence": [{"best_real_id": "b", "none_margin": -1.2, "ambiguity_probability": 0.10}]},
            {"reference": {"status": "accepted", "paths": ["c"]}, "evidence": [{"best_real_id": "c", "none_margin": -0.7, "ambiguity_probability": 0.20}]},
            {"reference": {"status": "accepted", "paths": ["d"]}, "evidence": [{"best_real_id": "d", "none_margin": -0.3, "ambiguity_probability": 0.25}]},
            {"reference": {"status": "NO_MATCH", "paths": []}, "evidence": [{"best_real_id": "x", "none_margin": 0.7, "ambiguity_probability": 0.15}]},
            {"reference": {"status": "NO_MATCH", "paths": []}, "evidence": [{"best_real_id": "y", "none_margin": 1.0, "ambiguity_probability": 0.30}]},
            {"reference": {"status": "AMBIGUOUS", "paths": []}, "evidence": [{"best_real_id": "m", "none_margin": 0.6, "ambiguity_probability": 0.96}]},
            {"reference": {"status": "AMBIGUOUS", "paths": []}, "evidence": [{"best_real_id": "n", "none_margin": 0.5, "ambiguity_probability": 0.90}]},
        ]
        for budget in (0, 5, 10):
            calibration, metrics = g54._rescue_calibration(records, budget)
            baseline = calibration["joint_baseline_validation_metrics"]
            self.assertEqual(calibration["selection_split"], "validation")
            self.assertEqual(calibration["status_arbitration"], "ambiguity-rescue")
            self.assertEqual(calibration["nomatch_recall_budget_pp"], budget)
            self.assertGreaterEqual(
                metrics["nomatch_recall"] + 1e-12,
                max(0.0, baseline["nomatch_recall"] - budget / 100.0),
            )
            self.assertGreaterEqual(
                metrics["accepted_status_recall"] + 1e-12,
                max(0.0, baseline["accepted_status_recall"] - 0.05),
            )


if __name__ == "__main__":
    unittest.main()
