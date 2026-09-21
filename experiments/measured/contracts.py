"""Executable, bounded catalog-projection contracts. No model code or gold fallback.

Schema projection is deliberately not unrestricted schema invention. Domains and
paraphrase families are split before training. Shared business concepts are disclosed.
"""
from __future__ import annotations
import copy
import hashlib
import itertools
import json
import random
from pathlib import Path

VERSION = 'measured-repair-v1'
PATHS = [('title',), ('createdAt',), ('updatedAt',), ('reviews', 'rating'),
         ('reviews', 'author', 'name'), ('reviews', 'moderator', 'name'),
         ('supplier', 'name'), ('reviews', 'author', 'id')]
GLOSSES = ['public display title', 'time of initial registration', 'time of most recent revision',
           'numeric rating of each review', 'display name of each review author',
           'display name of each review moderator', 'name of the supplier business',
           'identifier of each review author']
DOMAINS = {'train': [('Product', 'product'), ('Book', 'book'), ('Project', 'project')],
           'validation': [('Ticket', 'ticket')],
           'test': [('Article', 'article'), ('Course', 'course')]}
LANGUAGE = {
 'train': [
  ['public display title', 'human-readable title'],
  ['initial registration time', 'original creation timestamp'],
  ['latest modification time', 'most recent edit timestamp'],
  ['numeric review ratings', 'scores assigned in reviews'],
  ["review authors' display names", 'names of the people who wrote reviews'],
  ["review moderators' display names", 'names of the people moderating reviews'],
  ['supplier business name', 'name of the supplying organization'],
  ['review author identifiers', 'IDs of the people who wrote reviews']],
 'validation': [
  ['title shown to readers'], ['timestamp when it was registered'],
  ['timestamp when it was last revised'], ['rating values from reviews'],
  ['display name of each feedback author'], ['display name of each feedback moderator'],
  ['business name of the supplier'], ['unique identifier of each feedback author']],
 'test': [
  ['customer-facing title'], ['when it first entered the system'],
  ['when it was changed most recently'], ['numeric evaluations in the feedback'],
  ['display names of the feedback writers'], ['display names of the feedback moderators'],
  ['name of the upstream commercial provider'], ['IDs, not names, of the feedback writers']]}
SPECIAL = {'NO_MATCH': 'No catalog capability can satisfy this request.',
           'AMBIGUOUS': 'The request does not distinguish two different relationship roles.'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def catalog(entity, root):
    review, person, supplier = entity + 'Review', entity + 'Person', entity + 'Supplier'
    fields = {
     entity: {'id': 'ID!', 'title': 'String!', 'createdAt': 'String!', 'updatedAt': 'String!',
              'supplier': supplier, 'reviews': '[' + review + '!]!'},
     review: {'id': 'ID!', 'rating': 'Int!', 'author': person + '!', 'moderator': person},
     person: {'id': 'ID!', 'name': 'String!'}, supplier: {'id': 'ID!', 'name': 'String!'}}
    options = []
    for path, gloss in zip(PATHS, GLOSSES):
        parent, coordinates, types = entity, [], []
        for field in path:
            coordinates.append(parent + '.' + field)
            raw = fields[parent][field]
            types.append(raw)
            parent = raw.translate(str.maketrans('', '', '[]!'))
        route = [root] + list(path)
        options.append({'id': '.'.join(route), 'path': route, 'coordinates': coordinates,
                        'types': types, 'gloss': gloss})
    return {'entity': entity, 'root': root, 'fields': fields, 'options': options}


def dataset():
    public, refs = {}, {}
    combos = [(i,) for i in range(8)] + [(0, 1), (1, 2), (4, 5), (3, 4),
             (0, 6), (3, 7), (0, 3, 4), (1, 2, 6)]
    for split, domains in DOMAINS.items():
        public[split], refs[split] = [], {}
        for entity, root in domains:
            for variant in range(len(LANGUAGE[split][0])):
                for combo in combos:
                    for task in ['operation', 'schema']:
                        cat = catalog(entity, root)
                        clauses = [LANGUAGE[split][i][variant] for i in combo]
                        request = ('Return ' if task == 'operation' else 'Expose capabilities for ') + '; '.join(clauses) + ' for the ' + root + '.'
                        uid = sha([split, entity, task, request])[:24]
                        opts = cat.pop('options')
                        target_paths = [opts[i]['id'] for i in combo]
                        random.Random(uid).shuffle(opts)
                        item = {'id': uid, 'task': task, 'request': request,
                                'catalog': {**cat, 'options': opts}}
                        public[split].append(item)
                        refs[split][uid] = {'status': 'accepted', 'paths': target_paths,
                                           'fixture_seeds': [13, 47, 101]}
            # Unsupported and ambiguous requirements are separate from answerable tasks.
            for task in ['operation', 'schema']:
                for concept in [1, 3, 4, 6]:
                    cat = catalog(entity, root)
                    target = cat['options'][concept]['id']
                    cat['options'] = [o for o in cat['options'] if o['id'] != target]
                    q = 'Return ' + LANGUAGE[split][concept][0] + ' for the ' + root + '.'
                    uid = sha([split, task, q, 'missing'])[:24]
                    public[split].append({'id': uid, 'task': task, 'request': q, 'catalog': cat})
                    refs[split][uid] = {'status': 'NO_MATCH', 'paths': [], 'fixture_seeds': [13, 47, 101]}
                q = {'train': 'Return the name of the person associated with each review.',
                     'validation': 'Expose each review person name.',
                     'test': 'Show the display name of the person attached to each feedback item.'}[split]
                uid = sha([split, entity, task, q, 'ambiguous'])[:24]
                public[split].append({'id': uid, 'task': task, 'request': q, 'catalog': catalog(entity, root)})
                refs[split][uid] = {'status': 'AMBIGUOUS', 'paths': [], 'fixture_seeds': [13, 47, 101]}
    return public, refs


def option_text(option, typed=True):
    if option['id'] in SPECIAL:
        return SPECIAL[option['id']]
    meaning = option['gloss'] + '; relationship path ' + ' -> '.join(option['path'])
    if not typed:
        return meaning
    return meaning + '; coordinates ' + ', '.join(option['coordinates']) + '; return signatures ' + ', '.join(option['types'])


def options(item):
    return item['catalog']['options'] + [{'id': k, 'path': [], 'gloss': v} for k, v in SPECIAL.items()]


def selection_tree(paths):
    tree = {}
    for path in paths:
        at = tree
        for field in path:
            at = at.setdefault(field, {})
    return tree


def tree_text(tree):
    return ' '.join(field + (' { ' + tree_text(child) + ' }' if child else '')
                    for field, child in sorted(tree.items()))


def make_operation(item, selected):
    lookup = {o['id']: o for o in item['catalog']['options']}
    if not selected or any(s not in lookup for s in selected):
        raise ValueError('Operation requires at least one available path')
    paths = [lookup[s]['path'][1:] for s in selected]
    # Arguments/variables are generated, not extracted from evaluator-only fixture values.
    root = item['catalog']['root']
    return 'query Generated($id: ID!) { result: ' + root + '(id: $id) { ' + tree_text(selection_tree(paths)) + ' } }'


def field_inventory(cat, selected):
    lookup = {o['id']: o for o in cat['options']}
    inventory = {cat['entity'] + '.id': 'ID!'}
    for oid in selected:
        if oid not in lookup:
            raise ValueError('Unknown capability ' + oid)
        option = lookup[oid]
        for coord, raw in zip(option['coordinates'], option['types']):
            inventory[coord] = raw
            base = raw.translate(str.maketrans('', '', '[]!'))
            if base in cat['fields']:
                inventory[base + '.id'] = 'ID!'
    return inventory


def api_sdl(item, selected=None):
    cat = item['catalog']
    selected = selected if selected is not None else [o['id'] for o in cat['options']]
    inv = field_inventory(cat, selected)
    types = {}
    for coord, raw in inv.items():
        parent, field = coord.split('.')
        types.setdefault(parent, {})[field] = raw
    chunks = ['type Query { ' + cat['root'] + '(id: ID!): ' + cat['entity'] + ' }']
    for parent, fs in sorted(types.items()):
        chunks.append('type ' + parent + ' { ' + ' '.join(f + ': ' + t for f, t in sorted(fs.items())) + ' }')
    return '\n'.join(chunks)


def subgraph_sdls(item, selected):
    """Project selected capabilities and key closure; known owner convention, not learned ownership."""
    cat = item['catalog']; entity = cat['entity']
    inv = field_inventory(cat, selected)
    owners = {'catalog': {}, 'feedback': {}}
    for coord, raw in inv.items():
        parent, field = coord.split('.')
        owner = 'feedback' if parent in {entity + 'Review', entity + 'Person'} or coord == entity + '.reviews' else 'catalog'
        owners[owner].setdefault(parent, {})[field] = raw
    result = {}
    for owner, types in owners.items():
        if not types:
            continue
        imports = ['@key'] + (['@external'] if owner == 'feedback' and entity in types else [])
        chunks = ['extend schema @link(url: "https://specs.apollo.dev/federation/v2.3", import: ' + json.dumps(imports) + ')']
        if owner == 'catalog':
            chunks.append('type Query { ' + cat['root'] + '(id: ID!): ' + entity + ' }')
        for parent, fs in sorted(types.items()):
            extension = owner == 'feedback' and parent == entity
            fs = dict(fs)
            if extension:
                fs['id'] = 'ID! @external'
            chunks.append(('extend type ' if extension else 'type ') + parent + ' @key(fields: "id") { ' +
                          ' '.join(f + ': ' + t for f, t in sorted(fs.items())) + ' }')
        result[owner] = '\n'.join(chunks)
    return result


def fixture(seed):
    return {'id': 'entity-' + str(seed), 'title': 'Title-' + str(seed),
            'createdAt': 'created-' + str(seed), 'updatedAt': 'revised-' + str(seed),
            'supplier': {'id': 's-' + str(seed), 'name': 'Supplier-' + str(seed)},
            'reviews': [{'id': 'r-' + str(seed + i), 'rating': (seed + i) % 5 + 1,
                         'author': {'id': 'a-' + str(seed + i), 'name': 'Author-' + str(seed + i)},
                         'moderator': {'id': 'm-' + str(seed + i), 'name': 'Moderator-' + str(seed + i)}}
                        for i in range(seed % 3 + 1)]}


def expected_projection(tree, obj):
    # This projector interprets the evaluator's reference paths, not generated GraphQL.
    if obj is None:
        return None
    if isinstance(obj, list):
        return [expected_projection(tree, x) for x in obj]
    return {field: expected_projection(child, obj[field]) if child else obj[field]
            for field, child in tree.items()}


def judge(item, reference, prediction):
    from graphql import build_schema, graphql_sync, parse, validate
    status, selected = prediction['status'], prediction['selected']
    result = {'request_correct': False, 'graphql_valid': False, 'response_matches': [],
              'requirement_exact': False, 'incorrect_publication': False}
    if reference['status'] != 'accepted':
        result['request_correct'] = status == reference['status']
        result['incorrect_publication'] = status == 'accepted'
        return result
    if status != 'accepted':
        return result
    result['requirement_exact'] = set(selected) == set(reference['paths'])
    try:
        emitted = api_sdl(item, selected) if item['task'] == 'schema' else api_sdl(item)
        schema = build_schema(emitted)
        operation = make_operation(item, reference['paths'] if item['task'] == 'schema' else selected)
        # On schema tasks, this is an evaluator probe issued AFTER generating the schema.
        errors = validate(schema, parse(operation))
        result['graphql_valid'] = not errors
        result['operation'] = operation
        result['schema_sdl'] = emitted
        result['responses'] = []
        if errors:
            result['validation_errors'] = [str(e) for e in errors]
        else:
            lookup = {o['id']: o['path'][1:] for o in item['catalog']['options']}
            reference_tree = selection_tree([lookup[p] for p in reference['paths']])
            for seed in reference['fixture_seeds']:
                obj = fixture(seed)
                def resolve(_info, id, obj=obj):
                    return obj if id == obj['id'] else None
                actual = graphql_sync(schema, operation,
                        root_value={item['catalog']['root']: resolve}, variable_values={'id': obj['id']})
                expected = {'result': expected_projection(reference_tree, obj)}
                match = actual.errors is None and actual.data == expected
                result['response_matches'].append(match)
                result['responses'].append({'actual': actual.data, 'expected': expected,
                                            'errors': [str(e) for e in actual.errors or []]})
        result['request_correct'] = result['requirement_exact'] and result['graphql_valid'] and all(result['response_matches']) and len(result['response_matches']) == 3
    except Exception as exc:
        result['error'] = type(exc).__name__ + ': ' + str(exc)
    result['incorrect_publication'] = not result['request_correct']
    return result


def audit(public, refs):
    hashes = {split: {sha({k: v for k, v in r.items() if k != 'id'}) for r in rows}
              for split, rows in public.items()}
    for a, b in itertools.combinations(hashes, 2):
        if hashes[a] & hashes[b]:
            raise AssertionError('Public split overlap')
    for split, rows in public.items():
        if len({r['id'] for r in rows}) != len(rows):
            raise AssertionError('Duplicate task id')
        for row in rows:
            ref = refs[split][row['id']]
            if ref['status'] == 'accepted':
                check = judge(row, ref, {'status': 'accepted', 'selected': ref['paths']})
                if not check['request_correct']:
                    raise AssertionError(('Reference contract failed', row['id'], check))
    return {'counts': {s: len(v) for s, v in public.items()},
            'unique_public': {s: len(v) for s, v in hashes.items()},
            'overlap': 0, 'dataset_sha256': sha({'public': public, 'references': refs}),
            'scope': 'synthetic, shared concepts, held-out type names and paraphrase templates; not human-authored OOD'}


def write_dataset(directory):
    public, refs = dataset(); report = audit(public, refs)
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    for split in public:
        (directory / (split + '.public.json')).write_text(json.dumps(public[split], indent=2))
        (directory / (split + '.references.json')).write_text(json.dumps(refs[split], indent=2))
    (directory / 'audit.json').write_text(json.dumps(report, indent=2))
    return public, refs, report
