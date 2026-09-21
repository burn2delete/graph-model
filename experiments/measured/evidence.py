"""Evidence validation; rejects the original GDM41–44 manifest-only outputs."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import torch
import yaml
from .contracts import VERSION, sha, subgraph_sdls
from .models import state_hash


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def timing(samples):
    if not samples or any(not math.isfinite(x) or x < 0 for x in samples):
        raise ValueError('Real nonnegative timing samples required')
    return {'n': len(samples), 'p50_ms': float(np.quantile(samples, .5)),
            'p95_ms': float(np.quantile(samples, .95)), 'mean_ms': float(np.mean(samples))}


class Compositor:
    def __init__(self, directory, enabled=True):
        self.root = Path(directory).resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled; self.cache = {}
        self.version = None
        if enabled:
            if not shutil.which('rover'):
                raise RuntimeError('Rover is required; installation failure is not model failure')
            p = subprocess.run(['rover', '--version'], check=True, capture_output=True, text=True, timeout=20)
            self.version = p.stdout.strip()

    def compose(self, item, selected):
        sdls = subgraph_sdls(item, selected)
        identity = sha([sdls, '2.11.0'])
        if identity in self.cache:
            return self.cache[identity]
        if not self.enabled:
            return {'success': None, 'reason': 'unit-test-only; composition not run'}
        folder = self.root / identity; folder.mkdir(exist_ok=True)
        config = {'federation_version': '=2.11.0', 'subgraphs': {}}
        for owner, sdl in sdls.items():
            (folder / (owner + '.graphql')).write_text(sdl)
            config['subgraphs'][owner] = {'routing_url': 'http://' + owner + '/graphql',
                                          'schema': {'file': owner + '.graphql'}}
        (folder / 'supergraph.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        env = dict(os.environ, APOLLO_ELV2_LICENSE='accept', APOLLO_TELEMETRY_DISABLED='true')
        start = time.perf_counter_ns()
        p = subprocess.run(['rover', 'supergraph', 'compose', '--config', 'supergraph.yaml'],
                           cwd=folder, env=env, capture_output=True, text=True, timeout=120)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        (folder / 'supergraph.graphql').write_text(p.stdout)
        (folder / 'stderr.log').write_text(p.stderr)
        result = {'success': p.returncode == 0 and bool(p.stdout.strip()),
                  'returncode': p.returncode, 'elapsed_ms': elapsed, 'rover_version': self.version,
                  'composition_version': '2.11.0', 'cache_key': identity,
                  'supergraph_sha256': hashlib.sha256(p.stdout.encode()).hexdigest()}
        write_json(folder / 'receipt.json', result)
        self.cache[identity] = result
        return result


def aggregate(records):
    by_task = {}
    for task in ['operation', 'schema']:
        rs = [r for r in records if r['public']['task'] == task]
        if not rs:
            continue
        answerable = [r for r in rs if r['reference']['status'] == 'accepted']
        risk = [r for r in rs if r['reference']['status'] != 'accepted']
        correct = sum(bool(r['judgment']['request_correct']) for r in rs)
        by_task[task] = {'examples': len(rs), 'correct': correct, 'accuracy': correct / len(rs),
             'answerable': len(answerable),
             'answerable_correct': sum(bool(r['judgment']['request_correct']) for r in answerable),
             'risk_cases': len(risk), 'risk_correct': sum(bool(r['judgment']['request_correct']) for r in risk),
             'accepted': sum(r['prediction']['status'] == 'accepted' for r in rs),
             'incorrect_publications': sum(bool(r['judgment']['incorrect_publication']) for r in rs)}
    return by_task


def verify(directory):
    directory = Path(directory)
    summary_path = directory / 'summary.json'
    if not summary_path.is_file():
        raise ValueError('Missing summary.json: a manifest is not a result')
    summary = json.loads(summary_path.read_text())
    if summary.get('format') != VERSION or summary.get('evidence_kind') not in ['trained-model', 'pretrained-retrieval']:
        raise ValueError('Not a measured experiment')
    for name, expected in summary['file_hashes'].items():
        if file_hash(directory / name) != expected:
            raise ValueError('Evidence hash mismatch: ' + name)
    raw = [json.loads(line) for line in (directory / 'predictions.jsonl').read_text().splitlines()]
    if not raw or len(raw) != summary['test_examples']:
        raise ValueError('Missing or incomplete predictions')
    if aggregate(raw) != summary['metrics']:
        raise ValueError('Summary does not reproduce from raw predictions')
    for row in raw:
        if row['public']['id'] != row['id']:
            raise ValueError('Misaligned public case')
        pred, ref, judgment = row['prediction'], row['reference'], row['judgment']
        expected = pred['status'] == ref['status'] if ref['status'] != 'accepted' else (
            pred['status'] == 'accepted' and set(pred['selected']) == set(ref['paths']) and
            judgment['graphql_valid'] and judgment['response_matches'] == [True, True, True])
        if row['public']['task'] == 'schema' and ref['status'] == 'accepted' and expected:
            expected = judgment.get('composition', {}).get('success') is True
        if bool(judgment['request_correct']) != bool(expected):
            raise ValueError('Invalid correctness label')
        for response in judgment.get('responses', []):
            if not isinstance(response, dict) or 'actual' not in response or 'expected' not in response:
                raise ValueError('Missing execution evidence')
        if judgment.get('response_matches'):
            reproduced = [r['actual'] == r['expected'] and not r['errors'] for r in judgment['responses']]
            if reproduced != judgment['response_matches']:
                raise ValueError('Response comparison mismatch')
    training = summary['training']
    if summary['evidence_kind'] == 'trained-model':
        if training['optimizer_updates'] <= 0 or training['max_gradient_norm'] <= 0:
            raise ValueError('No training evidence')
        initial = torch.load(directory / 'initial.pt', map_location='cpu', weights_only=True)
        selected = torch.load(directory / 'selected.pt', map_location='cpu', weights_only=True)
        if state_hash(initial['state']) == state_hash(selected['state']):
            raise ValueError('No parameter changes')
        if state_hash(selected['state']) != training['selected_state_hash']:
            raise ValueError('Checkpoint hash mismatch')
        history = json.loads((directory / 'training.json').read_text())
        if not history or any(not math.isfinite(x['loss']) for x in history):
            raise ValueError('Missing or nonfinite training losses')
    raw_timing = json.loads((directory / 'timings.json').read_text())
    for task, samples in raw_timing['generation_ms'].items():
        if timing(samples) != summary['generation_latency'][task]:
            raise ValueError('Timing summary mismatch')
    if raw_timing['encoder_calls_measured'] <= 0:
        raise ValueError('Timing did not execute an encoder')
    if not summary['dataset_sha256'] or not summary['encoder']['revision']:
        raise ValueError('Missing provenance')
    receipt = {'verified': True, 'format': VERSION, 'examples': len(raw),
               'selected_state_hash': training.get('selected_state_hash'),
               'summary_sha256': file_hash(summary_path)}
    write_json(directory / 'VERIFIED.json', receipt)
    return receipt


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('directory'); a = p.parse_args()
    print(json.dumps(verify(a.directory), indent=2))
