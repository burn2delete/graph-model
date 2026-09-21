# Measurement policy — applies to every research batch

## Correction of the record

The original GDM41, GDM42, GDM43 and GDM44 jobs did **not** run the claimed model experiments. Their matrix jobs wrote configuration manifests. The scoring scripts used hand-assigned or simulated accuracy, latency and memory numbers. Those numbers are invalid as empirical evidence, and every architecture promotion based on them is withdrawn. Historical commits and Actions logs remain preserved for audit; do not delete or relabel them as measurements.

Affected initial runs: GDM41 35557767736; GDM42 35557923352; GDM43 35558183960; GDM44 35558330866. A green workflow status does not validate their results.

## Project execution

All compilation, tests, training, validation and benchmarks run in **GitHub Actions** in `burn2delete/graph-model`. Do not substitute local container results. Query planning is out of scope. Never create another manifest-only matrix to stand in for model execution.

## Required evidence

Every trained arm must save: the exact experiment configuration; dataset and split hashes; pretrained model revision and pooling configuration; initialization and selected checkpoint; optimizer-update counts, training history and changed-weight evidence; validation-only selection record; public test inputs and separately stored reference outputs; per-example predictions and generated GraphQL artifacts; evaluator output; raw timing samples and hardware/runtime details; a verifier receipt. Frozen pretrained retrieval-only controls must identify themselves explicitly and save actual embedding/retrieval evidence instead of claiming optimizer updates.

Fail the job on Python, model-loading, composition, result-schema or verifier failure. Use `bash` with `set -euo pipefail`. Always upload diagnostics after failure. Do not allow an existing log file to satisfy a missing-result check.

## Task contracts

Operations are full selection paths and executable response contracts. An author edge and an author's name leaf are different necessary actions, not globally exclusive semantic labels. Schema generation is explicitly catalog-grounded projection in this repaired screening batch, not unrestricted ontology invention. Separate model and engine evaluation; fixture values and hidden targets may never enter prediction. Empty or ambiguous predictions are not successes on answerable requests.

Use a real GraphQL parser/validator/executor. Keep requirements correctness distinct from Federation composition. Report skipped or unavailable composition as unmeasured, never as passed. A known-valid control must compose before using the composer to judge generated schemas.

## Experiment labels

An adapter over cached frozen representations is a **feature adapter**, not a fine-tuned transformer layer. Separate task checkpoints over one frozen pretrained initialization are not evidence about independent backbone fine-tuning. Record these distinctions in each artifact. Names alone do not constitute implemented ablations.

## Timing and selection

Measure single-request warm latency separately from batch throughput, head-only cached latency and offline embedding/index build time. Include warmup, raw samples, p50/p95, tokenizer/encoder calls and peak process RSS. Do not add timings from unrelated runners and call the result end-to-end latency. Never choose a winner from estimates, placeholder metrics, missing arms or test-target-dependent retraining. Report the actual number of unique semantic cases; shuffled copies are not new worlds.

## Hourly loop

Validate artifacts before promoting any result. If an experiment is broken, repair it in place. Do not open new numbered experiments to avoid completing existing ones. Do not duplicate active repaired jobs. Respect the bounded runner concurrency and do not provision paid compute without authorization.
