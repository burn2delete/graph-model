# Graph Model

**A GraphQL Decision Model (GDM): learn the decisions that satisfy a request, then construct and evaluate real GraphQL.**

The goal is a model that reasons about GraphQL and generates schemas and operations. The current measured implementation is deliberately narrower: a **frozen semantic encoder plus learned capability and ambiguity heads**, followed by **deterministic GraphQL construction**. It selects from a supplied capability catalog instead of generating arbitrary GraphQL tokens.

Research is limited to **operation generation and schema generation**. **Query planning is out of scope.** This is a research system, not a production-ready general-purpose GraphQL generator.

## Current research state

_Last synchronized: 2026-09-22. Latest canonical experiment: GDM54._

| State | Meaning |
| --- | --- |
| **Canonical through GDM54** | GDM50–GDM54 each have two independently verified measured attempts and a passing exact-attempt reproducibility audit. Canonical means usable, reproducible evidence under the recorded execution contract—not that every arm is a winner. |
| **GDM55: unpromoted** | Ambiguity-curriculum generalization is the next experiment. Its initial [run 35749924051](https://github.com/burn2delete/graph-model/actions/runs/35749924051) failed in preflight before smoke or measured training. It requires an in-place repair; no GDM55 performance result is claimed here. |
| **Invalid historical evidence** | The original GDM41–44 runs produced manifests and fabricated/simulated scores, not model results. Those scores and their promotions are withdrawn. Only the [canonical measured repair](https://github.com/burn2delete/graph-model/actions/runs/35565425124), with 196/196 verified results, is the GDM41–44 baseline. |

Read the [measurement policy](experiments/MEASUREMENT_POLICY.md) and [repair record](experiments/measured/REPAIR.md) before interpreting results or changing an experiment. Historical code and logs are retained for diagnosis; their presence does not make them current architecture or valid evidence.

## Architecture

### Decision pipeline

```text
Public request + task + typed capability catalog
                    |
        Deterministic clause decomposition
                    |
          Frozen DistilBERT encoder
       normalized features -> explicit 1e-4 grid
              /                         \
    fresh request-clause vectors    cached catalog/NONE vectors
              \                         /
             Clause–candidate pair features
                    |
      Learned residual feature adapter + scorer
         listwise scores over real paths + NONE
                    |
       Top-two real candidates + NONE evidence
                    |
          Learned clause-local ambiguity head
                    |
        Validation-calibrated thresholds and
        deterministic clause/status arbitration
                    |
         Deterministic request aggregation
              /                      \
     NO_MATCH / AMBIGUOUS       accepted capability paths
      no usable generation              |
                              Typed deterministic realization
                                  /                 \
                       GraphQL operation       projected SDL/subgraphs
                                  |                 |
                       parse/validate/execute   requirements + Rover
                                  \                 /
                         per-example evidence and metrics
```

### 1. Public input and clause decomposition

Each example provides a task (`operation` or `schema`), a request, and a catalog of available capabilities. A capability contains a **full field path**, GraphQL field coordinates and return signatures, and a public semantic description. For example, `ticket.reviews.author.name`, `ticket.reviews.author.id`, and `ticket.reviews.moderator.name` are distinct capabilities; the terminal word `name` alone is not enough to identify a response contract.

The current benchmark uses task-specific request framing and semicolon-separated clauses. The [clause splitter](experiments/followup45/run.py) reads only that public syntax and falls back to the whole request when the framing does not match. **This is not a learned natural-language decomposition model.** A clause normally asks for one capability; a multi-clause request asks for a set.

Catalogs are bounded and synthetic. The common generator exposes eight capability families involving titles, lifecycle timestamps, review ratings, author/moderator names, supplier names, and author IDs; missing-capability cases remove options. Fresh domain names do not imply new enterprise schemas or arbitrary graph topology.

### 2. Frozen semantic representations

The active line uses `distilbert/distilbert-base-uncased` with the exact model revision recorded in each result. Token embeddings are mean-pooled, normalized, and projected onto an explicit `1e-4` feature grid. The encoder is frozen: **its transformer weights do not receive optimizer updates**.

Catalog descriptions include paths, coordinates, and type signatures. Their vectors, including the fixed NONE description, are cached. Timed online requests freshly encode request clauses; they do not reuse request embeddings or re-encode the catalog. The encoder refuses silent input truncation above its declared token limit.

The feature grid is an explicitly recorded **numerical architecture change**, not an invisible implementation detail. Raw and canonical probe/corpus hashes make its effects auditable. See [the encoder](experiments/measured/models.py) and [the canonical feature/cache boundary](experiments/followup50/execute.py).

### 3. Learned capability selection, including NONE

For clause vector `q` and candidate vector `c`, the scorer consumes:

```text
[q, c, q*c, abs(q-c), c-mean(catalog vectors), structural features]
```

The structural features include path depth, task, coordinate count, and leaf-kind information. With embedding dimension `d`, the pair representation has `5d + 4` dimensions.

The [capability head](experiments/measured/models.py) applies a residual MLP feature adapter with a 32-unit bottleneck, then a 96-unit scoring MLP. It jointly scores all real candidates and a **NONE candidate**. NONE uses a fixed public description with a learned score; it is not an invented GraphQL field or a separately fine-tuned transformer token.

Training uses listwise supervision over real paths and NONE. Unsupported clauses can assign their target mass to NONE instead of forcing some real capability to win. Multiple supervised alternatives can share target mass. Validation selects the capability checkpoint; test and secondary-holdout targets do not participate.

These are **feature adapters over frozen representations**, not internal transformer adapters or transformer fine-tuning. Operation and schema results use task-local learned checkpoints over the same frozen encoder family; this does not measure independent backbone training or production shared-backbone residency.

### 4. A separate learned ambiguity discriminator

The [GDM51 architecture](experiments/followup51/run.py), retained by the later structured-head experiments, adds a clause-local ambiguity discriminator. First the capability model is trained and selected; then its weights remain fixed while the ambiguity head is trained.

The basic ambiguity input has seven scalar features: NONE-versus-best-real and best-versus-second-real logit margins, three probabilities, and two encoder similarity scores. The structured variant adds four signals about the top-two candidates: embedding similarity, shared parent path, equal depth, and matching leaf-kind classification. Its MLP has a 32-unit hidden layer.

**The structured ambiguity head sees eleven summary features, not the entire request embedding or the full graph.** It is a learned discriminator over candidate evidence, not a general symbolic ambiguity reasoner. Curriculum diversity and information lost by this representation remain research questions.

### 5. Calibration and deterministic arbitration

Training produces scores; validation-only calibration chooses how to act on them. The reference policy uses the NONE margin for `NO_MATCH`, then ambiguity confidence for `AMBIGUOUS`, otherwise the best real path. GDM54 measures alternate clause-level ordering and a separately calibrated high-confidence ambiguity rescue threshold.

Request aggregation is explicit: any remaining `NO_MATCH` clause rejects the request as `NO_MATCH`; otherwise any `AMBIGUOUS` clause makes the request ambiguous; otherwise selected paths are deduplicated and emitted. Changing clause precedence is not the same as changing this request-level rule.

Calibration budgets constrain **validation status recall**, not guaranteed future correctness. A 5pp guardrail means five absolute percentage points on the specified validation measure. It is not a promise of at most 5pp loss on a fresh holdout.

There is no trained request-global status head in this current line. The capability and ambiguity heads are trained; threshold selection, rescue budgets, and request aggregation are not optimizer-trained status models.

### 6. Deterministic GraphQL realization

An accepted operation becomes a selection tree with required path closure, a root alias, and a variable. For illustration—not as a claimed model prediction—selecting `ticket.title` and `ticket.reviews.author.name` produces:

```graphql
query Generated($id: ID!) {
  result: ticket(id: $id) {
    reviews {
      author {
        name
      }
    }
    title
  }
}
```

The operation is parsed and validated with `graphql-core`, then executed against three changing fixtures whose author/moderator, name/ID, and creation/revision values intentionally differ. A valid query returning the wrong values is not correct.

For schemas, selected paths are projected into a minimal field inventory with required IDs, object dependencies, and Federation key closure. Subgraph ownership and SDL realization follow deterministic conventions; Rover composes the actual subgraphs, and a separate evaluator checks requested capabilities. **Successful composition is not proof that a schema satisfies the request.**

This is **catalog projection plus deterministic SDL/Federation realization**, not unconstrained ontology invention, learned ownership, or arbitrary schema design. The current operation contract also does not establish general mutation/subscription generation. See [the executable contracts and renderers](experiments/measured/contracts.py).

## What the experiments have taught us

The architecture above is an evolving research line, not a collection of universally winning components. The useful conclusions are bounded by each experiment's controls and data.

| Experiment line | Measured learning | Consequence and limit |
| --- | --- | --- |
| [GDM45](https://github.com/burn2delete/graph-model/actions/runs/35570021211) | Clause decomposition improved answerable multi-capability generation relative to whole-request scoring. | Keep clause-local decisions. Use only its accepted-case evidence: schema risk/status scores had a task-prefix shortcut. |
| [GDM46–GDM49](experiments/followup49/) | Global rejection gates and independent absolute match thresholds often improved rejection while sacrificing answerable recall, especially across several clauses. | Risk handling must be evaluated separately from capability discrimination; rejecting everything is not success. |
| [GDM50](experiments/followup50/) | Explicit NONE makes an open-set outcome possible in listwise scoring, but does not by itself solve ambiguity or multi-clause recall. Repeated runs also exposed numerical and timing defects. | Retain an explicit unsupported outcome and strict evidence/reproducibility gates; do not promote an architecture from one green run. |
| [GDM51](experiments/followup51/) | A separate clause-local ambiguity discriminator recovered substantial answerable recall in its tested configurations, but risk/publication tradeoffs remained. | Separate capability scoring from ambiguity recognition; success on one synthetic holdout is not broad generalization. |
| [GDM52–GDM53](experiments/followup53/) | Sequential risk-first calibration cut incorrect publication at a recall cost. A zero-loss validation floor returned the joint control; relaxed budgets exposed intermediate points, particularly for schemas. | Evaluate a measured precision/recall tradeoff rather than a single aggregate accuracy. A validation floor need not transfer to held-out cases. |
| [GDM54](experiments/followup54/) | Ambiguity-first ordering recovered very few ambiguous cases while losing many correct NO_MATCH decisions. Constrained rescue did not improve held-out ambiguity over NO_MATCH-first. | The tested ordering changes are insufficient. Curriculum/representation improvement is the next hypothesis—not a proven solution or a proof that every possible calibration method is exhausted. |

### Latest canonical result: GDM54

The following is a **within-batch comparison on the same Ticket/Account holdout**, not a ranking against the different holdouts used by GDM50–GDM53. Rates are means over seeds `5401,5402`; count columns pool those two seeds from one measured attempt. Each task has **76 unique holdout cases**: 36 answerable, 32 NO_MATCH, and 8 AMBIGUOUS. Thus `6/16` below is eight ambiguity cases evaluated by two models, not sixteen independent semantic worlds. The reproducibility rerun does not increase the sample size.

| Task / arbitration | Answerable accuracy | Risk accuracy | Incorrect publication | AMBIGUOUS correct | NO_MATCH correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Operation / NO_MATCH-first | 37.50% | 55.00% | 19.74% | 1/16 | 43/64 |
| Operation / ambiguity-first | 37.50% | 36.25% | 19.74% | 2/16 | 27/64 |
| Operation / rescue 0pp, 5pp, or 10pp | 37.50% | 50.00% | 19.74% | 1/16 | 39/64 |
| Schema / NO_MATCH-first | 44.44% | 61.25% | 14.47% | 6/16 | 43/64 |
| Schema / ambiguity-first | 44.44% | 27.50% | 14.47% | 8/16 | 14/64 |
| Schema / rescue 0pp, 5pp, or 10pp | 44.44% | 58.75% | 14.47% | 6/16 | 41/64 |

All operation arms have **96.77% target precision / 32.81% target recall**; all schema arms have **100% / 39.06%**. Correct answerable requests fall from **17/32 to 9/24 to 1/16** for operations and **18/32 to 10/24 to 4/16** for schemas as clause count increases from one to three. This combines status rejection and any selection error; it is not a pure false-rejection metric.

Regression request accuracy is also separate: operation NO_MATCH-first / ambiguity-first / rescue is **45.88% / 38.66% / 45.10%**; schema is **44.76% / 33.68% / 43.99%**. These regression suites contain previously inspected cases and are not fresh evidence of generalization.

The accepted operation errors include moderator-name requests selecting author-name paths. No accepted schema semantic confusion was recorded on this holdout, but that does **not** erase the low recall or incorrect publication on risk cases.

### What the metrics mean

**Answerable accuracy** is exact request correctness among reference-answerable cases, including the executable/requirements contract—not merely deciding to accept. **Risk accuracy** requires the exact reference status (`NO_MATCH` versus `AMBIGUOUS`), not just any refusal. **Incorrect publication** counts accepted-but-wrong requests over **all evaluated requests**, not only published requests.

**Target precision and recall are counted only on reference-answerable cases.** Rejection yields an empty predicted target set, reducing recall. Consequently, 100% target precision can coexist with incorrect publication on unsupported or ambiguous requests. The [collector definitions](experiments/followup49/collect.py) are authoritative; do not present these precision figures as end-to-end publication safety.

### Speed, memory, and the numerical contract

For the GDM54 NO_MATCH-first control, attempt 1 reports:

| Task | Mean per-seed p50 | Mean per-seed p95 | Mean process RSS after evaluation |
| --- | ---: | ---: | ---: |
| Operation | 47.45 ms | 90.16 ms | 652,090 KiB |
| Schema | 45.66 ms | 88.38 ms | 667,020 KiB |

These are measured single-request warm-generation summaries, **not pooled quantiles, throughput, or an end-to-end service SLA**. Their scope is fresh clause encoding, feature scoring, deterministic generation and local validation, with catalog/NONE vectors cached. Downloads, initial embedding/index construction, backend fixture execution, and Rover composition are excluded; composition is evaluated separately. RSS is process-level and affected by preceding work in the worker, not isolated model-only memory. The audit retains both attempts' latency observations; identical model outputs do not imply identical hosted-runner speed.

The canonical execution path fixes PyTorch/OMP/MKL thread settings, default CPU dispatch, deterministic algorithms, and disables oneDNN. It uses non-foreach/non-fused AdamW and non-foreach clipping. AdamW moments and stored head parameters remain float32, while the final bias-corrected parameter application is recomputed in float64 and cast back to float32. That explicitly recorded repair followed diagnostics of cross-run denominator drift. See [the implementation](experiments/followup50/execute_deterministic.py).

Exact equality has been demonstrated by the recorded Actions attempts. It is **not a general guarantee across arbitrary hardware, library upgrades, or unseen configurations**; each new batch must still pass its own audit.

## Evidence and canonical promotion

All compilation, tests, training, validation, benchmarks, and independent evidence checks execute through **GitHub Actions**. Reading source and inspecting already-produced artifacts does not substitute for executing an evidence verifier in Actions.

A trained result must include the configuration and data/split hashes, exact encoder revision/pooling, initial and selected checkpoints, changed-weight evidence, optimizer counts and histories for every trained head, validation-only selection/calibration records, per-example predictions, generated GraphQL, real evaluator/composer evidence, raw timing samples, and runtime/memory metadata. Frozen retrieval-only controls must instead be labeled **inference-only**; they must never claim training updates. Hash-backbone preflight smoke is not a measured pretrained-model result.

Promotion follows this sequence:

1. Complete every declared arm and independently verify the actual source run's `BATCH_REPORT.json`. Preserve failed-arm diagnostics and repair in place; a manifest or a green job is insufficient.
2. Rerun the same source workflow, commit, configuration, and seeds. Bind each attempt's artifacts by exact IDs, digests, and job/attempt provenance—not filenames alone.
3. Run the dedicated Actions audit. Independently reconstruct each batch report, verify checkpoint and execution evidence, and compare selected state/capability/ambiguity hashes, epochs, optimizer/training records, calibration documents, deterministic metrics/predictions, and canonical feature hashes. For the current 20-result batches, both tasks must match 10/10. Timing and memory are observational, but their measurement contracts must be valid.
4. Inspect the actual audit report: require `complete`, `evidence_verified`, `reproducible`, and `promotion_eligible` to be true, with no errors. **Then update this README as part of the same canonical-promotion step**, before announcing promotion or starting the next numbered experiment.

A README update must refresh the canonical state/provenance, explain the implemented architecture delta or explicitly state that only data/calibration changed, summarize supported and negative findings, report schema and operation metrics separately with denominators and speed/memory scope, retain limitations, and separate the next unverified hypothesis. It must not promote failed, incomplete, simulated, or unaudited results. See the durable [README maintenance policy](experiments/MEASUREMENT_POLICY.md#canonical-promotion-and-readme-maintenance).

Do not launch a duplicate of an active batch or audit. Keep Actions concurrency bounded, preserve the hourly research cadence, and do not provision paid compute. A documentation-only refresh must not dispatch a measured batch.

### Canonical provenance

| Experiment | Canonical source | Measured run | Passing reproducibility audit |
| --- | --- | --- | --- |
| GDM41–44 repair | [Repair contract](experiments/measured/REPAIR.md) | [35565425124: 196/196](https://github.com/burn2delete/graph-model/actions/runs/35565425124) | Original fabricated runs excluded |
| GDM50 | [62d2720](https://github.com/burn2delete/graph-model/commit/62d2720800f5d98e1521e0200766a80e998ebe9b) | [35680386718](https://github.com/burn2delete/graph-model/actions/runs/35680386718) | [35687452276](https://github.com/burn2delete/graph-model/actions/runs/35687452276) |
| GDM51 | [f52baf8](https://github.com/burn2delete/graph-model/commit/f52baf82222947ca00139437e90fe9b5428bd9ae) | [35687697452](https://github.com/burn2delete/graph-model/actions/runs/35687697452) | [35695965127](https://github.com/burn2delete/graph-model/actions/runs/35695965127) |
| GDM52 | [f0ca439](https://github.com/burn2delete/graph-model/commit/f0ca439c3bb567213108b55a3bd56e449425509b) | [35702079193](https://github.com/burn2delete/graph-model/actions/runs/35702079193) | [35711502477](https://github.com/burn2delete/graph-model/actions/runs/35711502477) |
| GDM53 | [97b8ceb](https://github.com/burn2delete/graph-model/commit/97b8ceb7c3820690545873e9ced2854ad1e26cee) | [35717392980](https://github.com/burn2delete/graph-model/actions/runs/35717392980) | [35728895825](https://github.com/burn2delete/graph-model/actions/runs/35728895825) |
| GDM54 | [031c7f6](https://github.com/burn2delete/graph-model/commit/031c7f6087eb356f8baeeedcb2bf25ce9fe35cb4) | [35729681954](https://github.com/burn2delete/graph-model/actions/runs/35729681954) | [35748761095](https://github.com/burn2delete/graph-model/actions/runs/35748761095) |

Latest canonical source report: [GDM54 attempt-1 artifact 10695003395](https://github.com/burn2delete/graph-model/actions/runs/35729681954/artifacts/10695003395), ZIP SHA256 `7ee5658705431bcbad90dca5c3e5cd8511b373a567c375ab7f2dbcd60ecbca39`. Its independent [audit artifact 10703898523](https://github.com/burn2delete/graph-model/actions/runs/35748761095/artifacts/10703898523) has ZIP SHA256 `c334c8a4d342b23d88cc289f7bb8a5915d69c8352c4848488db3b8e1d0427cad`. Artifact retention is finite; a link or this summary is not a substitute for retained evidence when re-verification is needed.

## Next experiment: GDM55, not a result

[GDM55](experiments/followup55/) tests whether broader ambiguity supervision helps where the tested arbitration changes did not. It holds the structured architecture, capability objective, canonical numerical path, feature quantum, schema realization, and GDM54 5pp rescue calibration policy fixed; only the ambiguity-head curriculum changes.

Its five arms distinguish the canonical curriculum, repeated canonical examples at a larger update budget, lexical expansion, relation-family expansion, and a full curriculum combining diversity with disambiguating counterfactual negatives and training-only domain randomization. Volume matching is intended to distinguish more optimizer exposure from more semantic diversity. The full curriculum is a bundled treatment, not a separate causal ablation of each added ingredient.

Environment/Pipeline is calibration-only; Campaign/Profile is the secondary holdout. Role, lifecycle-time, name-versus-ID representation, and object-versus-supplier naming are separately tracked ambiguity families. Those holdout targets must not select training, checkpoints, calibration, or curriculum changes. Initial preflight failure is an implementation/contract issue to repair, not evidence for or against the modeling hypothesis.

## Limits and reading the repository

The experiments are small, synthetic, catalog-bounded screens with shared business concepts and topology families. Domain names and phrasing are changed across batches, but that alone is not enterprise out-of-distribution evaluation; some domain labels also recur across historical roles. Use actual split provenance and overlap checks, not names alone. Once inspected, a holdout becomes regression material for subsequent research. Use **within-batch paired controls** for causal interpretation rather than comparing percentages from different fresh holdouts as a progress leaderboard.

Benchmark design must retain shortcut controls, candidate-order checks, split-integrity checks, paired requests requiring different answers from the same candidates, and separate schema/operation metrics. Candidate-position, decision-kind-only, lexical, and refusal baselines are requirements for robust interpretation—not a claim that every historical batch implemented every control. Autoregressive generation, backbone fine-tuning or training from scratch, unrestricted schema invention, and production-scale generalization remain outside the evidence established by this line.

| Location | Purpose |
| --- | --- |
| [experiments/MEASUREMENT_POLICY.md](experiments/MEASUREMENT_POLICY.md) | Execution, evidence, scope, and README/promotion requirements |
| [experiments/measured/](experiments/measured/) | Repaired baseline, frozen encoders, shared heads, GraphQL contracts and evaluation |
| [experiments/followup50/](experiments/followup50/) | Explicit NONE, cached-catalog boundary, canonical CPU/optimizer execution |
| [experiments/followup51/](experiments/followup51/) | Staged capability/ambiguity training and structured ambiguity features |
| [followup52](experiments/followup52/), [followup53](experiments/followup53/), [followup54](experiments/followup54/) | Hierarchical calibration, recall-budget sensitivity, status arbitration |
| [experiments/followup55/](experiments/followup55/) | Current unpromoted ambiguity-curriculum experiment |
| [.github/workflows/](.github/workflows/) | Actions-only preflights, measured batches, diagnostics, and exact-attempt audits |

Use the declared workflow and its execution wrapper when reproducing an experiment: calling a historical `run.py` directly can bypass the canonical numerical configuration. Inspect the live queue before dispatching anything. No local compilation, tests, training, validation, or benchmarks are part of the accepted research workflow.
