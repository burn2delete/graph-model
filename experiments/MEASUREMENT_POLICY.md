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

## Canonical promotion and README maintenance

**Every experiment becoming canonical MUST include an update to the root `README.md` as part of that promotion.** This applies to positive, negative, and inconclusive experiments, including calibration-only or data-only follow-ups. A canonical experiment establishes usable evidence under a declared contract; it does not automatically make an arm the preferred architecture.

The promotion sequence is: complete measured evidence, successful independent verification, the experiment's required same-source/same-seed rerun and exact-attempt Actions audit, evidence-based analysis, then README synchronization and the promotion notification. Do not announce a new canonical experiment or launch the next numbered batch while its README update is outstanding. Record a documentation blocker if the update cannot be committed; never invent results to finish it.

The README update MUST:

1. Refresh the last-synchronized date, latest canonical experiment, and separate active/unpromoted status. Read live Actions state rather than repeating stale queued/running claims.
2. Explain the implemented architecture and what the new experiment changed. Distinguish learned components from deterministic construction and validation; feature adapters from transformer fine-tuning; catalog projection from unconstrained schema generation; trained heads from inference-only controls and validation-calibrated gates. Explicitly state when architecture is unchanged and only curriculum, calibration, arbitration, or numerical execution changed.
3. Summarize the hypothesis, paired controls, supported findings, failed hypotheses, unresolved limitations, and the next unverified hypothesis. Scope conclusions to the actual data and comparisons; a negative result does not prove all possible approaches in that family impossible.
4. Report schema and operation correctness separately, including answerable and risk accuracy, incorrect publication, target precision/recall, relevant clause-count or ambiguity-family failures, regression versus fresh-holdout behavior, and measured p50/p95 and memory with their execution scope. State denominators and whether counts pool seeds; repeated seeds and reproducibility attempts are not additional unique semantic worlds. Do not conflate accepted-status recall with answerable correctness, or answerable-case target precision with publication safety.
5. Link the exact measured source commit, workflow run and attempts, actual `BATCH_REPORT.json` artifact, and passing audit/report provenance, including artifact IDs and digests for the latest canonical result. Preserve invalidated-result warnings and prior diagnostic history. A README, a run link, or expired artifact metadata cannot substitute for source evidence.
6. Keep fresh holdouts out of optimizer, checkpoint, and calibration selection. Label inspected holdouts as regression-only for subsequent work and avoid presenting unlike holdout scores as a causal cross-batch leaderboard.

A docs-only refresh must not dispatch training, tests, benchmarks, or another numbered experiment to manufacture support for prose. Any necessary execution or document-validation checks still belong in GitHub Actions, not a local container. Keep README prose and the provenance ledger readable rather than appending raw logs.

The hourly orchestrator MUST retain this README-maintenance obligation whenever its task prompt is updated. It MUST read the current README after the measurement policy and repair record, and verify that any newly canonical experiment is documented before advancing. This is a required research-process step, not a claim that a README edit itself supplies evidence or that CI currently enforces documentation completeness.
