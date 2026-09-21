# GDM41–44 repair

The original results are invalid. No model won GDM41, 42, 43 or 44: the original matrix jobs only wrote manifests, and their ranking scripts manufactured metrics. The old dictionaries are removed from current entrypoints. History remains intact.

The existing four entrypoints and workflows now call `experiments.measured.run`. The coordinator `.github/workflows/measured-repair.yml` executes the whole repair once, grouped by backbone to reuse frozen representation caches, with at most three model workers concurrently. The four legacy workflow names remain manual entrypoints for rerunning a specific corrected batch; they no longer emit placeholders on every push.

## What actually runs

98 declared configurations, two fixed seeds each. GDM42 includes genuine pretrained Qwen retrieval-only controls: they perform model inference and validation-only threshold selection but do not claim gradient updates. All other arms train PyTorch policies, retain initial and selected checkpoint weights, and fail evidence verification if the weights did not change.

The encoders are DistilBERT, ModernBERT, mmBERT and NeoBERT. Qwen GraphQL supplies candidate retrieval where configured. Model revisions resolve to commit hashes before loading and are saved. NeoBERT is the sole explicitly permitted pinned remote-code model. Loading/dependency failures remain failed arms, not substituted models or scores.

Current controls are frozen-encoder experiments. Task adapters are residual MLP **feature adapters**, not internal transformer adapters. Separate checkpoints train independent task-local learned policy components over a frozen pretrained initialization; independent full-backbone fine-tuning and doubled-residency memory have NOT been measured. These corrections replace the overbroad earlier labels rather than pretending the underlying hypotheses were already implemented.

The state/action arms score public path-prefix states and enforce path closure. They do not invent unrestricted schema mutations. Flat operation outputs are full paths, never ambiguous terminal coordinates alone. The `capability-set` and `multilabel` names describe equivalent implementations and are marked as aliases, not independent theories. Repeated configurations across historical batches are replication/rerun coverage, not additional independent evidence.

## Data and evaluation

The dataset has separate public input and reference files, split before training by type names and paraphrase templates. It remains synthetic and shares business concepts and topology families; do not call it enterprise OOD or broad natural-language validation. Do not compare its numbers directly with older eight-query repeated tests.

Operations are rendered as real GraphQL with a required variable and root alias, parsed and validated using graphql-core, and executed against three changing fixtures. Author, moderator, author ID, title, creation time and revision time intentionally differ. Wrong but valid paths fail response comparison.

Schema tasks select available capability paths and project a minimal Federation schema with deterministic ownership and keys. They are catalog-grounded projection, not unconstrained schema design or learned ownership. Rover composes actual projected subgraphs. The separate evaluator probes requested paths against the generated schema. References never enter `predict()` or repair a missing generated capability.

## Evidence and speed

Each run contains `initial.pt`, `selected.pt`, `training.json`, per-example `predictions.jsonl`, `timings.json`, `summary.json` and `VERIFIED.json`, plus config/model revisions and dataset hashes. The verifier checks real updates, changed weights, checkpoints, result recounts, execution comparisons and timing quantiles. A manifest, an empty log, a simulated number or a green job alone cannot pass.

Reported generation latency is measured one request at a time with fresh query encoding and cached catalog vectors, after warmup and across repeated trials. It includes policy decoding and GraphQL rendering/local validation, but excludes download, initial index build, Rover composition and backend fixture execution. Rover timing is recorded separately. Process peak RSS includes previous configurations in the same worker and is not attributed to an isolated model. Full-backbone sharing memory claims remain out of scope.

All compilation, tests, training and benchmarking run on GitHub Actions. The first preflight (35559894236) passed 12 contract tests. The full smoke training and corrected model workers must independently pass; do not infer that from the preliminary test result.
