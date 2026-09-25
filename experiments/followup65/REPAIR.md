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

The workflow-file change automatically started exactly one repaired source run, [`36097355421`](https://github.com/burn2delete/graph-model/actions/runs/36097355421), attempt 1, at the repair SHA.

## Pre-measurement failure: GDM64 entrypoint recursion

The repaired workflow passed Rover installation and then passed all **106 current and inherited contract tests** in preflight job [`107952308820`](https://github.com/burn2delete/graph-model/actions/runs/36097355421/job/107952308820). The deterministic hash smoke then failed before producing a valid smoke result because `experiments.followup65.execute` rebound the real `followup64.run.main` function to `followup65.run.main`, while `followup65.run.main` delegated back through that same rebound symbol. The resulting call cycle recursed until Python raised:

```text
RecursionError: maximum recursion depth exceeded
```

The failure is strictly pre-measurement. The full DistilBERT operation worker and independent batch verifier were skipped, so there is no GDM65 model-performance evidence from this run. The failed preflight artifact is **10847918183**, SHA-256 `d5c893df4d2547ba25b5232b8a0919f311253e2cf163527fd2e13dd8c9e59287`.

## In-place source repair for execution delegation

Repair implementation commit on branch `gdm65-recursion-repair`: [`344ebb5e792c2c2ac01a673cf2e7bb4a89ed522a`](https://github.com/burn2delete/graph-model/commit/344ebb5e792c2c2ac01a673cf2e7bb4a89ed522a).

The repair changes only GDM65 execution delegation plus a contract regression test:

- `experiments/followup65/execute.py` no longer rebinds the real GDM64 run module's `main`. Instead it temporarily replaces only the `followup64.execute` runner reference with a small namespace whose `main` points to GDM65, so the canonical numerical wrapper still executes exactly once while GDM65 can safely delegate into the original GDM64 runner.
- `experiments/followup65/test_followup.py` adds a non-training contract test that reproduces the wrapper/delegation topology and requires the call sequence `GDM65 -> original GDM64` without rebinding the real GDM64 entrypoint.
- The measured workflow content is unchanged; only its Git executable-bit metadata is changed on the repair branch so that the existing push-path filter can trigger one fresh run after the repair reaches `main`.

No GDM65 architecture, relation features, trainable capacity, data, seeds, holdout, optimizer, numerical path, curriculum, calibration, GraphQL realization, or evidence gate is changed by this repair. The next accepted source must pass Actions compilation/tests and the independent structured-relation hash smoke before any measured result is considered.

All compilation, tests, smoke, training, evaluation, and independent verification remain Actions-only. If another failure occurs, preserve it and repair GDM65 in place; do not skip arms or advance to GDM66.
