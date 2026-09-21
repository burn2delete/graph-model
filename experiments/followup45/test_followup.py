import inspect
import unittest

from experiments.followup45 import run
from experiments.measured.contracts import dataset, judge


class FollowupContractsTest(unittest.TestCase):
    def test_atomic_clause_count_comes_from_public_request(self):
        public, refs = dataset()
        row = next(r for r in public['test'] if r['task'] == 'operation' and refs['test'][r['id']]['status'] == 'accepted' and len(refs['test'][r['id']]['paths']) == 3)
        self.assertEqual(len(run.clauses(row)), 3)

    def test_fresh_holdout_is_disjoint_from_repaired_test_ids(self):
        public, _ = dataset()
        old = {r['id'] for r in public['test']}
        for task in ['operation', 'schema']:
            rows, refs = run.fresh_holdout(task)
            self.assertTrue(rows)
            self.assertFalse(old & {r['id'] for r in rows})
            self.assertEqual({r['id'] for r in rows}, set(refs))

    def test_holdout_gold_contracts_execute(self):
        for task in ['operation', 'schema']:
            rows, refs = run.fresh_holdout(task)
            for row in rows:
                ref = refs[row['id']]
                if ref['status'] != 'accepted':
                    continue
                result = judge(row, ref, {'status': 'accepted', 'selected': ref['paths']})
                self.assertTrue(result['request_correct'], (row['id'], result))

    def test_inference_cannot_receive_reference(self):
        self.assertNotIn('reference', inspect.signature(run.infer).parameters)
        self.assertNotIn('refs', inspect.signature(run.infer).parameters)

    def test_all_followup_arms_are_measured_architectural_hypotheses(self):
        self.assertEqual(set(run.ARMS), {'whole-risk', 'clause-risk', 'clause-hardneg', 'clause-top2', 'clause-top4'})
        self.assertEqual(run.ARMS['clause-top2']['top_k'], 2)
        self.assertEqual(run.ARMS['clause-top4']['top_k'], 4)


if __name__ == '__main__':
    unittest.main()
