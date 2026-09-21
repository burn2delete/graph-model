"""Executable repair matrix, not scores or rankings.

Aliases from the old plan are explicitly marked. Fully separate here means
independently trained task-local policy checkpoints over frozen pretrained weights;
it does not claim independent transformer fine-tuning.
"""
from itertools import product

BACKBONES = ['distilbert', 'modernbert', 'mmbert', 'neobert']


def make(batch, task, arch, backbone='mmbert', mode='single', top_k=0):
    tree = arch in ['path-state', 'state', 'action-state']
    typed = arch != 'semantic-only'
    pairwise = arch == 'pairwise'
    adapter = arch in ['task-adapter', 'adapter']
    if arch.startswith('retrieval-top'):
        top_k = int(arch.rsplit('top', 1)[1])
    if arch == 'retrieval' and not top_k:
        top_k = 2
    if arch in ['shared-head', 'shared-encoder']:
        mode = 'shared-heads'
    equivalent = {'capability-set': 'independent-multilabel', 'multilabel': 'independent-multilabel',
                  'terminal-only': 'full-path-multilabel', 'terminal-path': 'full-path-multilabel',
                  'independent': 'independent-multilabel'}.get(arch, arch)
    identifier = '-'.join([str(batch), task, arch, backbone, mode, 'k' + str(top_k)])
    return {'id': identifier, 'batch': batch, 'task': task, 'architecture': arch,
            'backbone': backbone, 'mode': mode, 'decoder': 'tree' if tree else 'flat',
            'typed': typed, 'pairwise': pairwise, 'feature_adapter': adapter,
            'top_k': top_k, 'structural_augmentation': arch == 'structural-semantic',
            'implementation_equivalence': equivalent,
            'backbone_training': 'frozen', 'seed_set': [4101, 4102],
            'scope': 'catalog-grounded projection; no query planning'}


def configs(batch):
    out = []
    if batch == 41:
        groups = {
            'operation': ['path-state', 'terminal-only', 'retrieval-top2', 'retrieval-top4',
                          'pairwise', 'independent', 'shared-encoder', 'task-adapter'],
            'schema': ['action-state', 'capability-set', 'pairwise', 'multilabel',
                       'shared-encoder', 'task-adapter', 'structural-semantic', 'semantic-only']}
        for task, architectures in groups.items():
            out += [make(batch, task, arch) for arch in architectures]
    elif batch == 42:
        for task, arch, backbone in product(['operation', 'schema'], ['state', 'retrieval', 'pairwise', 'adapter'], BACKBONES):
            out.append(make(batch, task, arch, backbone))
        for task, k, reranker in product(['operation', 'schema'], [1, 2, 4, 8], ['none', 'mmbert', 'modernbert']):
            c = make(batch, task, 'retrieval-top' + str(k), 'qwen' if reranker == 'none' else reranker, top_k=k)
            c['retrieval_only'] = reranker == 'none'
            out.append(c)
    elif batch == 43:
        for task, architectures in {
          'operation': ['path-state', 'retrieval-top2', 'terminal-path', 'task-adapter'],
          'schema': ['structural-semantic', 'capability-set', 'action-state', 'task-adapter']}.items():
            out += [make(batch, task, arch) for arch in architectures]
        out += [make(batch, 'operation', 'path-state', 'modernbert'),
                make(batch, 'schema', 'structural-semantic', 'modernbert')]
    elif batch == 44:
        for mode, backbone, k in product(['shared-policy', 'shared-heads', 'separate-adapters', 'separate-checkpoints'],
                                         ['mmbert', 'modernbert'], [0, 2]):
            out.append(make(batch, 'both', 'terminal-path', backbone, mode=mode, top_k=k))
    else:
        raise ValueError('Unsupported batch')
    if len({c['id'] for c in out}) != len(out):
        raise AssertionError('Duplicate experiment id')
    return out


def all_configs(backbone=None, batches=(41, 42, 43, 44)):
    rows = [c for batch in batches for c in configs(batch)]
    return [c for c in rows if backbone is None or c['backbone'] == backbone]
