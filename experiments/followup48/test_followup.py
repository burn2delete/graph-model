from __future__ import annotations

import unittest

from experiments.followup48.run import ARMS, FRESH_LANGUAGE, clause_status, fresh_holdout


class GDM48Contracts(unittest.TestCase):
    def test_arm_scope(self):
        self.assertEqual(set(ARMS), {
            "global-scoreonly", "clause-prob", "clause-logit", "clause-hybrid", "clause-hybrid-nohardneg"
        })
        self.assertEqual(ARMS["global-scoreonly"]["gate"], "global-scoreonly")
        self.assertFalse(ARMS["clause-hybrid-nohardneg"]["hardneg"])

    def test_fresh_holdout_task_prefix_and_risk_coverage(self):
        for task in ["operation", "schema"]:
            rows, refs = fresh_holdout(task)
            prefix = "Return " if task == "operation" else "Expose capabilities for "
            self.assertTrue(rows)
            self.assertTrue(all(r["request"].startswith(prefix) for r in rows))
            statuses = [refs[r["id"]]["status"] for r in rows]
            self.assertIn("accepted", statuses)
            self.assertIn("NO_MATCH", statuses)
            self.assertIn("AMBIGUOUS", statuses)
            # At least one multi-clause risk case is required to test aggregation.
            self.assertTrue(any("; " in r["request"] and refs[r["id"]]["status"] != "accepted" for r in rows))
            self.assertEqual(len(rows), len({r["id"] for r in rows}))

    def test_fresh_language_has_eight_capabilities(self):
        self.assertEqual(len(FRESH_LANGUAGE), 8)
        self.assertEqual(len(set(FRESH_LANGUAGE)), 8)

    def test_clause_status_precedence(self):
        cal = {"support_threshold": 0.4, "ambiguity_threshold": 0.2}
        self.assertEqual(clause_status({"support": 0.3, "ambiguity": 0.9}, cal), "NO_MATCH")
        self.assertEqual(clause_status({"support": 0.8, "ambiguity": 0.1}, cal), "AMBIGUOUS")
        self.assertEqual(clause_status({"support": 0.8, "ambiguity": 0.4}, cal), "accepted")


if __name__ == "__main__":
    unittest.main()
