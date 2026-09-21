from __future__ import annotations

import unittest

from experiments.followup49 import run


class GDM49Contracts(unittest.TestCase):
    def test_bounded_arm_matrix(self):
        self.assertEqual(
            set(run.ARMS),
            {
                "listwise-hybrid-control",
                "binary-bce-publiccal",
                "binary-bce-expandedcal",
                "binary-hardneg-expandedcal",
                "binary-focal-expandedcal",
            },
        )
        self.assertEqual(run.ARMS["listwise-hybrid-control"]["family"], "listwise-control")
        for name, cfg in run.ARMS.items():
            if name != "listwise-hybrid-control":
                self.assertEqual(cfg["family"], "binary-matcher")

    def test_calibration_and_holdout_are_disjoint(self):
        for task in ("operation", "schema"):
            cal_rows, _ = run.calibration_set(task)
            hold_rows, _ = run.fresh_holdout(task)
            self.assertTrue(cal_rows)
            self.assertTrue(hold_rows)
            self.assertTrue(set(r["id"] for r in cal_rows).isdisjoint(r["id"] for r in hold_rows))
            cal_text = {r["request"] for r in cal_rows}
            hold_text = {r["request"] for r in hold_rows}
            self.assertTrue(cal_text.isdisjoint(hold_text))

    def test_task_prefixes_and_risk_coverage(self):
        for task, prefix in (("operation", "Return "), ("schema", "Expose capabilities for ")):
            rows, refs = run.fresh_holdout(task)
            statuses = {refs[r["id"]]["status"] for r in rows}
            self.assertEqual(statuses, {"accepted", "NO_MATCH", "AMBIGUOUS"})
            self.assertTrue(all(r["request"].startswith(prefix) for r in rows))
            self.assertTrue(any("; " in r["request"] and refs[r["id"]]["status"] == "accepted" for r in rows))
            self.assertTrue(any("; " in r["request"] and refs[r["id"]]["status"] != "accepted" for r in rows))

    def test_language_partitions_do_not_reuse_holdout_phrases(self):
        hold = set(run.HOLDOUT_LANGUAGE + run.HOLDOUT_UNSUPPORTED + run.HOLDOUT_AMBIGUOUS)
        cal = set(run.CALIBRATION_LANGUAGE + run.CALIBRATION_UNSUPPORTED + run.CALIBRATION_AMBIGUOUS)
        train = set(run.NO_MATCH_TRAINING + [p for p, _ in run.AMBIGUOUS_TRAINING])
        self.assertTrue(hold.isdisjoint(cal))
        self.assertTrue(hold.isdisjoint(train))

    def test_schema_scope_remains_catalog_projection(self):
        rows, refs = run.fresh_holdout("schema")
        accepted = [r for r in rows if refs[r["id"]]["status"] == "accepted"]
        self.assertTrue(accepted)
        for row in accepted:
            available = {o["id"] for o in row["catalog"]["options"]}
            self.assertTrue(set(refs[row["id"]]["paths"]).issubset(available))


if __name__ == "__main__":
    unittest.main()
