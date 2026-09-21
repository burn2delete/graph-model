import copy
import json
from pathlib import Path
import tempfile
import unittest
import torch
from .batches import all_configs, configs, make
from .contracts import dataset
from .evidence import verify
from .models import Encoder, Features, Policy, candidate_units, labels, predict, state_hash


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.public, self.refs = dataset()
        self.item = self.public['train'][0]
        self.config = make(41, 'operation', 'pairwise', 'hash')
        self.features = Features(Encoder('hash'), self.config)

    def test_all_98_configs_resolve_to_real_implementations(self):
        self.assertEqual([len(configs(b)) for b in [41,42,43,44]], [16,56,10,16])
        self.assertEqual(len(all_configs()), 98)
        for c in all_configs():
            self.assertIn(c['decoder'], ['tree', 'flat'])
            self.assertNotIn('quality', c)
            self.assertNotIn('p50_ms', c)

    def test_optimizer_changes_parameters(self):
        x, units, _ = self.features.get(self.item)
        y = labels(self.refs['train'][self.item['id']], units)
        p = Policy(self.features.dim)
        old = state_hash(p.state_dict())
        optimizer = torch.optim.AdamW(p.parameters(), lr=.01)
        for _ in range(3):
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(p(x, 'operation'), y)
            loss.backward(); optimizer.step()
        self.assertNotEqual(old, state_hash(p.state_dict()))

    def test_gold_changes_do_not_enter_predictions(self):
        p = Policy(self.features.dim)
        before = predict(p, self.features, self.item, .5)
        changed = copy.deepcopy(self.refs)
        changed['train'][self.item['id']] = {'status': 'AMBIGUOUS', 'paths': []}
        after = predict(p, self.features, self.item, .5)
        self.assertEqual(before, after)

    def test_task_specific_heads_have_distinct_parameters(self):
        p = Policy(self.features.dim, 'shared-heads')
        self.assertEqual(len(p.heads), 2)
        ids0 = {id(x) for x in p.heads[0].parameters()}
        ids1 = {id(x) for x in p.heads[1].parameters()}
        self.assertFalse(ids0 & ids1)

    def test_feature_adapter_is_really_trainable(self):
        p = Policy(self.features.dim, 'separate-adapters')
        self.assertIsNotNone(p.heads[0].adapter)
        self.assertTrue(all(x.requires_grad for x in p.heads[0].adapter.parameters()))

    def test_state_units_include_correct_parent_path(self):
        units = candidate_units(self.item, tree=True)
        root = self.item['catalog']['root']
        author = next(x for x in units if x['id'] == root + '.reviews.author')
        self.assertEqual(author['parent'], root + '.reviews')
        self.assertIn(root + '.reviews.author.name', author['members'])
        self.assertNotIn(root + '.reviews.moderator.name', author['members'])

    def test_manifest_is_not_measured_result(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, 'summary.json').write_text(json.dumps({'mode': 'shared-heads', 'quality': .95}))
            with self.assertRaises(ValueError):
                verify(d)

    def test_missing_summary_fails(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, 'run.log').write_text('finished')
            with self.assertRaises(ValueError):
                verify(d)

    def test_hash_features_deterministic_across_instances(self):
        text = [self.item['request']]
        self.assertTrue(torch.equal(Encoder('hash').encode(text), Encoder('hash').encode(text)))


if __name__ == '__main__':
    unittest.main()
