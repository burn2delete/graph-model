from __future__ import annotations

import unittest

from experiments.followup51 import run as g51
from experiments.followup52 import run as g52


class GDM52Contracts(unittest.TestCase):
    def test_bounded_paired_arm_matrix(self):
        self.assertEqual(
            set(g52.ARMS),
            {
                "joint-mlp-control",
                "joint-structured-control",
                "sequential-mlp",
                "sequential-structured",
                "sequential-structured-noninferior",
            },
        )
        for cfg in g52.ARMS.values():
            self.assertEqual(cfg["family"], "explicit-none-plus-ambiguity")
            self.assertEqual(cfg["head"], "mlp")
            self.assertFalse(cfg["hard_negative"])
        for left, right in (
            ("joint-mlp-control", "sequential-mlp"),
            ("joint-structured-control", "sequential-structured"),
        ):
            a = dict(g52.ARMS[left])
            b = dict(g52.ARMS[right])
            a.pop("calibration_strategy")
            b.pop("calibration_strategy")
            self.assertEqual(a, b)

    def test_fresh_calibration_and_holdout_are_disjoint(self):
        for task in ("operation", "schema"):
            cal_rows, cal_refs = g52.calibration_set(task)
            hold_rows, hold_refs = g52.fresh_holdout(task)
            old_rows, old_refs = g51.fresh_holdout(task)
            self.assertTrue(cal_rows and hold_rows and old_rows)
            self.assertTrue(set(cal_refs).isdisjoint(hold_refs))
            self.assertTrue(set(hold_refs).isdisjoint(old_refs))
            self.assertEqual({row["catalog"]["root"] for row in cal_rows}, {"portfolio", "ledger"})
            self.assertEqual({row["catalog"]["root"] for row in hold_rows}, {"shipment", "invoice"})
            self.assertTrue(
                {row["catalog"]["root"] for row in hold_rows}.isdisjoint(
                    {row["catalog"]["root"] for row in old_rows}
                )
            )
            self.assertTrue(any(ref["status"] == "accepted" for ref in hold_refs.values()))
            self.assertTrue(any(ref["status"] == "NO_MATCH" for ref in hold_refs.values()))
            self.assertTrue(any(ref["status"] == "AMBIGUOUS" for ref in hold_refs.values()))

    @staticmethod
    def _records():
        return [
            {
                "reference": {"status": "accepted", "paths": ["a"]},
                "evidence": [
                    {
                        "best_real_id": "a",
                        "none_margin": -1.0,
                        "ambiguity_probability": 0.05,
                    }
                ],
            },
            {
                "reference": {"status": "accepted", "paths": ["b"]},
                "evidence": [
                    {
                        "best_real_id": "b",
                        "none_margin": -0.8,
                        "ambiguity_probability": 0.15,
                    }
                ],
            },
            {
                "reference": {"status": "NO_MATCH", "paths": []},
                "evidence": [
                    {
                        "best_real_id": "a",
                        "none_margin": 1.2,
                        "ambiguity_probability": 0.10,
                    }
                ],
            },
            {
                "reference": {"status": "AMBIGUOUS", "paths": []},
                "evidence": [
                    {
                        "best_real_id": "a",
                        "none_margin": -0.7,
                        "ambiguity_probability": 0.95,
                    }
                ],
            },
        ]

    def test_sequential_calibration_separates_status_gates(self):
        calibration, metrics = g52.calibrate_records(
            self._records(), "sequential-status-specific"
        )
        self.assertEqual(calibration["threshold_selection"], "sequential-status-specific")
        self.assertEqual(calibration["selection_split"], "validation")
        self.assertEqual(calibration["none_gate_validation_metrics"]["nomatch_recall"], 1.0)
        self.assertEqual(calibration["ambiguity_gate_validation_metrics"]["ambiguity_recall"], 1.0)
        self.assertEqual(metrics["exact_accuracy"], 1.0)

    def test_noninferiority_strategy_preserves_joint_accepted_recall(self):
        calibration, metrics = g52.calibrate_records(
            self._records(), "sequential-status-specific-noninferior"
        )
        floor = calibration["accepted_status_recall_floor"]
        self.assertGreaterEqual(metrics["accepted_status_recall"] + 1e-12, floor)
        self.assertEqual(
            calibration["accepted_status_recall_floor_source"],
            "same-record GDM51 joint calibration",
        )
        self.assertIn("joint_baseline_validation_metrics", calibration)


if __name__ == "__main__":
    unittest.main()
