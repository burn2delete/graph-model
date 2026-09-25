# GDM65 repair record

This record preserves workflow diagnostics, not model-performance evidence. GDM65 remains unpromoted until complete measured operation evidence, the required exact-source rerun, independent audit, and README promotion all pass.

## Pre-measurement failure: unterminated `find -exec`

Original measured source: [`a1e88067776c835c99cd259e41136d4bd87bf08b`](https://github.com/burn2delete/graph-model/commit/a1e88067776c835c99cd259e41136d4bd87bf08b).

Failed Actions run: [`36023296645`, attempt 1](https://github.com/burn2delete/graph-model/actions/runs/36023296645/attempts/1). Preflight job: [`107713325611`](https://github.com/burn2delete/graph-model/actions/runs/36023296645/job/107713325611).

Pinned dependency installation and Rover archive extraction completed. Both workflow installation blocks contained an unescaped shell semicolon after `find ... -exec install ...`. The preflight stopped before compilation, tests, smoke training, the measured operation worker, or batch verification.

Exact diagnostic excerpts from the original Actions job log:

```text
2026-09-24T15:52:39.2967105Z find: missing argument to `-exec'
2026-09-24T15:52:39.2985094Z ##[error]Process completed with exit code 1.
2026-09-24T15:52:39.4753516Z ##[error]No files were found with the provided path: artifacts/. No artifacts will be uploaded.
```

No preflight ZIP was uploaded: the artifacts directory was still empty. The original job/run logs are the diagnostic source; do not invent an artifact ID or regard the absence of artifacts as successful evidence. No GDM65 model results came from this failed attempt.

## In-place workflow-only repair

Repair commit: [`e2a331f405689867c28979754082716abecf3bb7`](https://github.com/burn2delete/graph-model/commit/e2a331f405689867c28979754082716abecf3bb7).

The GitHub commit diff contains exactly two changed lines in `.github/workflows/measured-followup65.yml`: the preflight and operation Rover-install commands now terminate `find -exec` with an escaped semicolon:

```bash
find /tmp/rover-install -type f -name rover -exec install -m 755 '{}' "$HOME/.local/bin/rover" \;
```

No experiment source, tests, architecture, data, curriculum, seeds, holdout, numerical execution, calibration, or evidence gate changed. The source commit remains the repair commit above even though this diagnostic document was committed afterward.

The workflow-file change automatically started exactly one repaired source run, [`36097355421`](https://github.com/burn2delete/graph-model/actions/runs/36097355421), attempt 1, at the repair SHA. It was observed in progress when this record was written. Read live Actions state for subsequent progress. No manual duplicate dispatch was requested.

All compilation, tests, smoke, training, evaluation, and independent verification remain Actions-only. The repair itself supplies no model-performance evidence. If another failure occurs, preserve it and repair GDM65 in place; do not skip arms or advance to GDM66.
