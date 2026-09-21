import copy
import unittest
from graphql import build_schema, graphql_sync, parse, validate
from .contracts import (PATHS, api_sdl, audit, catalog, dataset, fixture,
                        judge, make_operation, subgraph_sdls, sha)


class ContractsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.public, cls.refs = dataset()

    def item_for(self, index, task='operation'):
        for item in self.public['test']:
            if item['task'] != task:
                continue
            ref = self.refs['test'][item['id']]
            wanted = item['catalog']['root'] + '.' + '.'.join(PATHS[index])
            if ref['paths'] == [wanted]:
                return item, ref
        self.fail('Missing test reference')

    def test_all_reference_programs_execute(self):
        report = audit(self.public, self.refs)
        self.assertEqual(report['overlap'], 0)
        self.assertGreater(report['counts']['train'], 100)

    def test_author_not_moderator(self):
        item, ref = self.item_for(4)
        wrong = item['catalog']['root'] + '.reviews.moderator.name'
        r = judge(item, ref, {'status': 'accepted', 'selected': [wrong]})
        self.assertTrue(r['graphql_valid'])
        self.assertFalse(r['request_correct'])
        self.assertEqual(r['response_matches'], [False] * 3)

    def test_author_identifier_not_name(self):
        item, ref = self.item_for(4)
        wrong = item['catalog']['root'] + '.reviews.author.id'
        self.assertFalse(judge(item, ref, {'status': 'accepted', 'selected': [wrong]})['request_correct'])

    def test_schema_projection_not_gold_repair(self):
        item, ref = self.item_for(4, 'schema')
        wrong = item['catalog']['root'] + '.title'
        r = judge(item, ref, {'status': 'accepted', 'selected': [wrong]})
        self.assertFalse(r['graphql_valid'])
        self.assertFalse(r['request_correct'])
        self.assertNotIn('author:', r['schema_sdl'])

    def test_no_gold_in_public_examples(self):
        forbidden = {'target', 'expected', 'reference', 'labels', 'semantic_target'}
        def visit(obj):
            if isinstance(obj, dict):
                self.assertFalse(forbidden & set(obj))
                for value in obj.values(): visit(value)
            elif isinstance(obj, list):
                for value in obj: visit(value)
        visit(self.public)

    def test_path_preserves_role(self):
        item, _ = self.item_for(4)
        opts = item['catalog']['options']
        author = next(o for o in opts if o['path'][-3:] == ['reviews', 'author', 'name'])
        mod = next(o for o in opts if o['path'][-3:] == ['reviews', 'moderator', 'name'])
        self.assertEqual(author['coordinates'][-1], mod['coordinates'][-1])
        self.assertNotEqual(author['id'], mod['id'])

    def test_real_parser_rejects_invalid_object_selection(self):
        item, _ = self.item_for(4)
        schema = build_schema(api_sdl(item))
        query = '{ ' + item['catalog']['root'] + '(id: "x") { reviews } }'
        self.assertTrue(validate(schema, parse(query)))

    def test_real_parser_handles_variables_aliases(self):
        item, ref = self.item_for(0)
        operation = make_operation(item, ref['paths'])
        self.assertIn('$id: ID!', operation)
        self.assertIn('result:', operation)
        self.assertFalse(validate(build_schema(api_sdl(item)), parse(operation)))

    def test_rover_link_imports(self):
        item, ref = self.item_for(4, 'schema')
        sdls = subgraph_sdls(item, ref['paths'])
        self.assertIn('"@external"', sdls['feedback'])
        self.assertIn('@external', sdls['feedback'])
        self.assertIn('type Query', sdls['catalog'])

    def test_missing_is_not_answerable(self):
        for item in self.public['test']:
            ref = self.refs['test'][item['id']]
            if ref['status'] == 'NO_MATCH':
                self.assertTrue(judge(item, ref, {'status': 'NO_MATCH', 'selected': []})['request_correct'])
                wrong = item['catalog']['options'][0]['id']
                self.assertFalse(judge(item, ref, {'status': 'accepted', 'selected': [wrong]})['request_correct'])
                return
        self.fail('Missing risk case')

    def test_fixture_roles_are_distinct(self):
        f = fixture(13)
        review = f['reviews'][0]
        self.assertNotEqual(review['author']['name'], review['moderator']['name'])
        self.assertNotEqual(review['author']['id'], review['author']['name'])
        self.assertNotEqual(f['createdAt'], f['updatedAt'])

    def test_dataset_reproducible(self):
        p, r = dataset()
        self.assertEqual(sha([p, r]), sha([self.public, self.refs]))


if __name__ == '__main__':
    unittest.main()
