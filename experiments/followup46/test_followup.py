import tempfile
import unittest
from pathlib import Path

from experiments.followup46.run import (
    AMBIGUOUS_LANGUAGE, ARMS, balanced_status_rows, corrected_dataset,
    fresh_holdout, task_prefix,
)


class GDM46ContractsTest(unittest.TestCase):
    def test_every_public_risk_request_uses_task_consistent_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            public, refs, report = corrected_dataset(Path(tmp))
            self.assertIn("benchmark_correction", report)
            for split, rows in public.items():
                for row in rows:
                    expected = task_prefix(row["task"])
                    self.assertTrue(row["request"].startswith(expected), (split, row["task"], row["request"]))
                    if refs[split][row["id"]]["status"] == "AMBIGUOUS":
                        self.assertIn(AMBIGUOUS_LANGUAGE[split], row["request"])

    def test_secondary_holdout_has_no_schema_return_prefix_shortcut(self):
        for task in ["operation", "schema"]:
            rows, refs = fresh_holdout(task)
            self.assertEqual(len(rows), 42)
            expected = task_prefix(task)
            for row in rows:
                self.assertTrue(row["request"].startswith(expected))
            statuses = [refs[row["id"]]["status"] for row in rows]
            self.assertEqual(statuses.count("accepted"), 32)
            self.assertEqual(statuses.count("NO_MATCH"), 8)
            self.assertEqual(statuses.count("AMBIGUOUS"), 2)

    def test_balanced_status_schedule_is_actually_balanced(self):
        with tempfile.TemporaryDirectory() as tmp:
            public, refs, _ = corrected_dataset(Path(tmp))
            rows = [r for r in public["train"] if r["task"] == "operation"]
            schedule = balanced_status_rows(rows, refs["train"], 4601)
            counts = {"accepted": 0, "NO_MATCH": 0, "AMBIGUOUS": 0}
            for row in schedule:
                counts[refs["train"][row["id"]]["status"]] += 1
            self.assertEqual(len(set(counts.values())), 1, counts)
            self.assertGreater(counts["NO_MATCH"], 0)

    def test_batch_is_bounded_and_does_not_add_backbones(self):
        self.assertEqual(set(ARMS), {"joint-raw", "balanced-raw", "balanced-learned", "balanced-learned-hardneg"})
        self.assertEqual(len(ARMS), 4)


if __name__ == "__main__":
    unittest.main()
