"""Collection gate: no missing arm, manifest-only job or failed worker can pass."""
import argparse
import json
from pathlib import Path
from .batches import all_configs
from .evidence import verify, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--batches', default='41,42,43,44')
    parser.add_argument('--seeds', default='4101,4102')
    args = parser.parse_args()
    root = Path(args.directory)
    expected = {(c['id'], s) for c in all_configs(batches=tuple(map(int, args.batches.split(','))))
                for s in map(int, args.seeds.split(','))}
    found, summaries, errors = set(), [], []
    for path in root.rglob('worker.json'):
        worker = json.loads(path.read_text())
        if not worker.get('complete'):
            errors.append({'worker': str(path), 'errors': worker.get('errors')})
    for path in root.rglob('summary.json'):
        try:
            verify(path.parent)
            s = json.loads(path.read_text())
            key = (s['config']['id'], s['seed'])
            if key in found:
                raise ValueError('Duplicate arm/seed')
            found.add(key)
            summaries.append({'id': key[0], 'seed': key[1], 'metrics': s['metrics'],
                              'generation_latency': s['generation_latency'], 'hardware': s['hardware'],
                              'evidence_kind': s['evidence_kind'], 'source_commit': s['source_commit']})
        except Exception as exc:
            errors.append({'path': str(path), 'error': str(exc)})
    missing = sorted(expected - found); unexpected = sorted(found - expected)
    report = {'complete': not missing and not errors and not unexpected,
              'expected': len(expected), 'verified': len(found), 'missing': missing,
              'unexpected': unexpected, 'errors': errors, 'results': summaries,
              'previous_manifest_scores_invalid': True}
    write_json(root / 'BATCH_REPORT.json', report)
    print(json.dumps({'complete': report['complete'], 'expected': len(expected),
                      'verified': len(found), 'missing_count': len(missing), 'errors': errors}, indent=2))
    if not report['complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
