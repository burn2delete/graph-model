"""Real repaired batch worker. Invoke only in GitHub Actions for project runs.

Example: python -m experiments.measured.run --backbone mmbert --batches 41,42,43,44
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from .batches import all_configs, make
from .contracts import (VERSION, SPECIAL, api_sdl, judge, make_operation,
                        option_text, sha, subgraph_sdls, write_dataset)
from .evidence import Compositor, aggregate, file_hash, timing, verify, write_json
from .models import Encoder, Features, Policy, labels, predict, state_hash


def exact_decision(prediction, reference):
    return prediction['status'] == reference['status'] and (
        reference['status'] != 'accepted' or set(prediction['selected']) == set(reference['paths']))


def subset(rows, task):
    return rows if task == 'both' else [r for r in rows if r['task'] == task]


def augmentation(rows, refs):
    extra, references, seen = [], {}, set()
    for row in rows:
        for opt in row['catalog']['options']:
            request = 'Select the ' + opt['types'][-1] + ' value along the graph path ' + opt['id'] + '.'
            identity = sha([row['task'], request])[:24]
            if identity in seen:
                continue
            seen.add(identity)
            item = copy.deepcopy(row); item['id'] = identity; item['request'] = request
            extra.append(item)
            references[identity] = {'status': 'accepted', 'paths': [opt['id']]}
    return extra, references


def train_policy(config, seed, features, public, refs, directory, epochs):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    shared = config['mode'] != 'single' or config['task'] == 'both'
    train_rows = subset(public['train'], 'both' if shared else config['task'])
    val_rows = subset(public['validation'], 'both' if shared else config['task'])
    train_refs = dict(refs['train'])
    if config['structural_augmentation']:
        added, added_refs = augmentation(train_rows, train_refs)
        train_rows = train_rows + added; train_refs.update(added_refs)
    policy = Policy(features.dim, config['mode'], config['feature_adapter'])
    initial = copy.deepcopy(policy.state_dict())
    torch.save({'state': initial, 'config': config}, directory / 'initial.pt')
    initial_hash = state_hash(initial)
    grouped = {}
    for row in train_rows:
        x, units, _ = features.get(row)
        y = labels(train_refs[row['id']], units)
        grouped.setdefault(row['task'], []).append((x, y))
    tensors = {task: (torch.cat([x for x, _ in groups]), torch.cat([y for _, y in groups]))
               for task, groups in grouped.items()}
    if config['mode'] == 'separate-checkpoints':
        optimizers = {task: torch.optim.AdamW(policy.heads[0 if task == 'operation' else 1].parameters(), lr=.001)
                      for task in tensors}
    else:
        opt = torch.optim.AdamW(policy.parameters(), lr=.001)
        optimizers = {task: opt for task in tensors}
    history, updates, max_norm = [], 0, 0.0
    best_score, best_state, best_thresholds, best_epoch, selected_updates = None, None, None, None, 0
    generator = torch.Generator().manual_seed(seed)
    threshold_grid = [.25, .4, .5, .6, .75]
    for epoch in range(1, epochs + 1):
        policy.train(); losses = []
        for task, (x, y) in sorted(tensors.items()):
            permutation = torch.randperm(len(y), generator=generator)
            for start in range(0, len(y), 64):
                take = permutation[start:start + 64]
                optimizer = optimizers[task]; optimizer.zero_grad(set_to_none=True)
                logits = policy(x[take], task)
                loss = F.binary_cross_entropy_with_logits(logits, y[take])
                if config['pairwise']:
                    pos, neg = logits[y[take] == 1], logits[y[take] == 0]
                    if len(pos) and len(neg):
                        loss = loss + .25 * F.softplus(neg[None, :] - pos[:, None]).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0))
                if not math.isfinite(norm):
                    raise FloatingPointError('Nonfinite gradient')
                max_norm = max(max_norm, norm); optimizer.step(); updates += 1
                losses.append(float(loss.detach()))
        policy.eval(); thresholds = {}; validation_correct = 0
        for task in tensors:
            selected_rows = [r for r in val_rows if r['task'] == task]
            scores = []
            for threshold in threshold_grid:
                count = sum(exact_decision(predict(policy, features, row, threshold), refs['validation'][row['id']])
                            for row in selected_rows)
                scores.append((count, -abs(threshold - .5), threshold))
            count, _, chosen = max(scores)
            thresholds[task] = chosen; validation_correct += count
        epoch_loss = float(np.mean(losses))
        entry = {'epoch': epoch, 'loss': epoch_loss, 'optimizer_updates': updates,
                 'validation_correct': validation_correct, 'validation_examples': len(val_rows),
                 'thresholds': thresholds}
        history.append(entry)
        criterion = (validation_correct, -epoch_loss)
        if best_score is None or criterion > best_score:
            best_score = criterion; best_state = copy.deepcopy(policy.state_dict())
            best_thresholds = dict(thresholds); best_epoch = epoch; selected_updates = updates
        print(json.dumps({'event': 'epoch', 'arm': config['id'], 'seed': seed, **entry}), flush=True)
    if best_state is None or state_hash(best_state) == initial_hash or max_norm <= 0:
        raise RuntimeError('Training failed to change the selected policy')
    policy.load_state_dict(best_state); policy.eval()
    torch.save({'state': best_state, 'config': config, 'thresholds': best_thresholds,
                'encoder': features.encoder.meta, 'seed': seed}, directory / 'selected.pt')
    if config['mode'] == 'separate-checkpoints':
        for task in tensors:
            idx = 0 if task == 'operation' else 1
            torch.save({'state': policy.heads[idx].state_dict(), 'encoder': features.encoder.meta,
                        'task': task}, directory / ('selected-' + task + '.pt'))
    write_json(directory / 'training.json', history)
    receipt = {'optimizer_updates': updates, 'selected_optimizer_updates': selected_updates,
               'selected_epoch': best_epoch, 'max_gradient_norm': max_norm,
               'initial_state_hash': initial_hash, 'selected_state_hash': state_hash(best_state),
               'train_examples': len(train_rows), 'training_rows': sum(len(y) for _, y in tensors.values()),
               'trainable_parameters': sum(p.numel() for p in policy.parameters()),
               'thresholds': best_thresholds, 'validation_examples': len(val_rows),
               'independent_backbone_finetuning': False,
               'adapter_location': 'frozen-embedding feature space' if config['feature_adapter'] or config['mode'] in ['separate-adapters', 'separate-checkpoints'] else None}
    return policy, receipt


def retrieval_prediction(encoder, item, k, threshold, online=False):
    opts = item['catalog']['options']
    vectors = encoder.encode([option_text(o, True) for o in opts])
    query = encoder.query(item['request'], cached=not online)
    scores = vectors @ query
    order = torch.argsort(scores, descending=True, stable=True)[:min(k, len(opts))].tolist()
    selected = [opts[i]['id'] for i in order if float(scores[i]) >= threshold]
    return {'status': 'accepted' if selected else 'NO_MATCH', 'selected': sorted(selected),
            'scores': {o['id']: float(s) for o, s in zip(opts, scores)},
            'retrieved': [opts[i]['id'] for i in order]}


def emission_validation(item, prediction):
    """No gold labels; measures generation and local GraphQL validation only."""
    from graphql import build_schema, parse, validate
    if prediction['status'] != 'accepted':
        return
    if item['task'] == 'schema':
        build_schema(api_sdl(item, prediction['selected']))
        subgraph_sdls(item, prediction['selected'])
    else:
        validate(build_schema(api_sdl(item)), parse(make_operation(item, prediction['selected'])))


def run_one(config, seed, encoder, retriever, public, refs, data_report, compositor, root, epochs):
    directory = root / (config['id'] + '-seed' + str(seed))
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'config.json', config)
    features = Features(encoder, config, retriever)
    start_training = time.perf_counter_ns()
    retrieval_only = bool(config.get('retrieval_only'))
    if retrieval_only:
        validation = subset(public['validation'], config['task'])
        grid = [.2, .3, .4, .5, .6, .7, .8]
        threshold = max(grid, key=lambda t: (sum(exact_decision(retrieval_prediction(encoder, r, config['top_k'], t), refs['validation'][r['id']]) for r in validation), -t))
        receipt = {'optimizer_updates': 0, 'thresholds': {config['task']: threshold},
                   'reason': 'frozen pretrained retrieval-only baseline; no head training claimed'}
        def infer(item, online=False):
            return retrieval_prediction(encoder, item, config['top_k'], threshold, online)
        evidence_kind = 'pretrained-retrieval'
        write_json(directory / 'training.json', [])
    else:
        policy, receipt = train_policy(config, seed, features, public, refs, directory, epochs)
        def infer(item, online=False):
            return predict(policy, features, item, receipt['thresholds'][item['task']], online)
        evidence_kind = 'trained-model'
    training_ms = (time.perf_counter_ns() - start_training) / 1e6
    test_rows = subset(public['test'], config['task'])
    records = []
    for item in test_rows:
        prediction = infer(item)
        reference = refs['test'][item['id']]
        judgment = judge(item, reference, prediction)
        if item['task'] == 'schema' and prediction['status'] == 'accepted':
            composition = compositor.compose(item, prediction['selected'])
            judgment['composition'] = composition
            judgment['request_correct'] = judgment['request_correct'] and composition['success'] is True
            judgment['incorrect_publication'] = not judgment['request_correct']
        records.append({'id': item['id'], 'public': item, 'reference': reference,
                        'prediction': prediction, 'judgment': judgment})
    with open(directory / 'predictions.jsonl', 'w') as handle:
        for record in records:
            handle.write(json.dumps(record, allow_nan=False) + '\n')
    # A control which ignores the request is genuinely evaluated, never assigned a score.
    controls = []
    for item in test_rows:
        pred = {'status': 'accepted', 'selected': [item['catalog']['options'][0]['id']]}
        controls.append(bool(judge(item, refs['test'][item['id']], pred)['request_correct']))
    # Raw single-request measurements. Corpus vectors may be cached; query embeddings are not.
    encoder_before = encoder.calls + (retriever.calls if retriever is not None and retriever is not encoder else 0)
    samples = {}
    for task in sorted({r['task'] for r in test_rows}):
        bench_rows = [r for r in test_rows if r['task'] == task][:12]
        for row in bench_rows[:3]:
            emission_validation(row, infer(row, online=True))
        samples[task] = []
        for _ in range(3):
            for row in bench_rows:
                begin = time.perf_counter_ns()
                pred = infer(row, online=True)
                emission_validation(row, pred)
                elapsed = (time.perf_counter_ns() - begin) / 1e6
                samples[task].append(elapsed)
                # Online mode must not change the produced program.
                cached_pred = infer(row, online=False)
                if (pred['status'], pred['selected']) != (cached_pred['status'], cached_pred['selected']):
                    raise AssertionError('Online/cached prediction mismatch')
    encoder_after = encoder.calls + (retriever.calls if retriever is not None and retriever is not encoder else 0)
    raw_timing = {'generation_ms': samples, 'warmup_requests_per_task': 3,
                  'encoder_calls_measured': encoder_after - encoder_before,
                  'scope': 'single-request query encoding + cached catalog + policy + GraphQL rendering/validation; excludes download, corpus build, Rover and fixture backend',
                  'batch_throughput': None}
    write_json(directory / 'timings.json', raw_timing)
    cpuinfo = Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else ''
    hardware = {'platform': platform.platform(), 'machine': platform.machine(),
                'cpu': next((l.split(':', 1)[1].strip() for l in cpuinfo.splitlines() if l.startswith('model name')), platform.processor()),
                'threads': torch.get_num_threads(), 'torch': str(torch.__version__),
                'peak_worker_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'memory_scope': 'process high-water mark; includes prior configs and both resident frozen encoders'}
    files = ['config.json', 'predictions.jsonl', 'training.json', 'timings.json']
    if not retrieval_only:
        files += ['initial.pt', 'selected.pt']
    summary = {'format': VERSION, 'evidence_kind': evidence_kind, 'config': config,
               'source_commit': os.environ.get('GITHUB_SHA', 'unrecorded'),
               'run_id': os.environ.get('GITHUB_RUN_ID'), 'seed': seed,
               'dataset_sha256': data_report['dataset_sha256'], 'dataset_scope': data_report['scope'],
               'encoder': encoder.meta, 'retriever': retriever.meta if retriever else None,
               'training': receipt, 'training_and_selection_ms': training_ms,
               'test_examples': len(records), 'metrics': aggregate(records),
               'generation_latency': {task: timing(s) for task, s in samples.items()},
               'latency_scope': raw_timing['scope'], 'hardware': hardware,
               'first_option_control_correct': sum(controls), 'first_option_control_total': len(controls),
               'file_hashes': {name: file_hash(directory / name) for name in files}}
    write_json(directory / 'summary.json', summary)
    verified = verify(directory)
    print(json.dumps({'event': 'verified_arm', 'id': config['id'], 'seed': seed,
                      'metrics': summary['metrics'], 'verification': verified}), flush=True)
    return {'config': config['id'], 'seed': seed, 'directory': str(directory), 'verified': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', required=True)
    parser.add_argument('--batches', default='41,42,43,44')
    parser.add_argument('--output', default='artifacts/measured')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--seeds', default='4101,4102')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if os.environ.get('GITHUB_ACTIONS') != 'true':
        raise RuntimeError('Project execution belongs in GitHub Actions')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    public, refs, report = write_dataset(root / 'dataset')
    configs = all_configs(args.backbone, tuple(int(b) for b in args.batches.split(',')))
    if args.smoke:
        c = make(41, 'both', 'pairwise', 'hash', mode='shared-heads')
        configs = [c]; args.backbone = 'hash'; args.seeds = '4101'; args.epochs = 2
    if not configs:
        raise ValueError('No implemented arms match this worker')
    write_json(root / 'planned.json', {'configurations': configs, 'seeds': args.seeds.split(',')})
    compositor = Compositor(root / 'composition')
    # Fail infrastructure first, before interpreting any model-generated failure.
    control = public['train'][0]
    all_paths = [o['id'] for o in control['catalog']['options']]
    if not compositor.compose(control, all_paths)['success']:
        raise RuntimeError('Known-valid Federation control failed composition')
    encoder = Encoder(args.backbone)
    retriever = encoder if args.backbone == 'qwen' else (Encoder('qwen') if any(c['top_k'] for c in configs) else None)
    write_json(root / 'encoders.json', {'encoder': encoder.meta, 'retriever': retriever.meta if retriever else None})
    results, errors = [], []
    for config in configs:
        for seed in map(int, args.seeds.split(',')):
            try:
                results.append(run_one(config, seed, encoder, retriever, public, refs, report,
                                       compositor, root, args.epochs))
            except Exception:
                error = {'config': config['id'], 'seed': seed, 'traceback': traceback.format_exc()}
                errors.append(error); print(json.dumps({'event': 'failed_arm', **error}), flush=True)
    write_json(root / 'worker.json', {'complete': not errors, 'results': results, 'errors': errors,
                                    'expected_runs': len(configs) * len(args.seeds.split(','))})
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
