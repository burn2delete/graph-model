# Graph Model

**A GraphQL Decision Model (GDM): learn the decisions that satisfy a request, then construct and evaluate real GraphQL.**

The goal is a model that reasons about GraphQL and generates schemas and operations. The current measured implementation is deliberately narrower: a **frozen semantic encoder plus learned capability and ambiguity heads**, followed by **deterministic GraphQL construction**. It selects from a supplied capability catalog instead of generating arbitrary GraphQL tokens.

Research is limited to **operation generation and schema generation**. **Query planning is out of scope.** This is a research system, not a production-ready general-purpose GraphQL generator.

## Current research state

_Last synchronized: 2026-09-22. Latest canonical experiment: GDM55._

| State | Meaning |
| --- | --- |
| **Canonical through GDM55** | GDM50–GDM55 each have two independently verified measured attempts and a passing exact-attempt reproducibility audit. Canonical means usable, reproducible evidence under the recorded execution contract—not that every arm is a winning architecture. |
| **Next hypothesis: curriculum balance/stability, unpromoted** | GDM55 shows that broader relation-family supervision can recover some held-out ambiguity, but the effect is strongly seed/task dependent and a bundled full curriculum can make ambiguity worse. The next bounded question is how to preserve relation-family gains while stabilizing them and avoiding NO_MATCH/publication regressions. No result is claimed for that hypothesis yet. |
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

The [GDM51 architecture](experiments/followup51/run.py), retained by GDM52–GDM55, adds a clause-local ambiguity discriminator. First the capability model is trained and selected; then its weights remain fixed while the ambiguity head is trained.

The basic ambiguity input has seven scalar features: NONE-versus-best-real and best-versus-second-real logit margins, three probabilities, and two encoder similarity scores. The structured variant adds four signals about the top-two candidates: embedding similarity, shared parent path, equal depth, and matching leaf-kind classification. Its MLP has a 32-unit hidden layer.

**The structured ambiguity head sees eleven summary features, not the entire request embedding or the full graph.** It is a learned discriminator over candidate evidence, not a general symbolic ambiguity reasoner. GDM55 changes only what ambiguity examples this same head trains on; it does not change the head architecture.

### 5. Calibration and deterministic arbitration

Training produces scores; validation-only calibration chooses how to act on them. The reference policy uses the NONE margin for `NO_MATCH`, then ambiguity confidence for `AMBIGUOUS`, otherwise the best real path. GDM54 measured alternate ordering and high-confidence ambiguity rescue. GDM55 holds the GDM54 5pp high-confidence rescue policy fixed while changing only the ambiguity-head curriculum.

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
| [GDM54](experiments/followup54/) | Ambiguity-first ordering recovered very few ambiguous cases while losing many correct NO_MATCH decisions. Constrained rescue did not improve held-out ambiguity over NO_MATCH-first. | The tested ordering changes are insufficient; improve the learned ambiguity evidence before doing more threshold-only work. |
| [GDM55](experiments/followup55/) | Broader relation-family supervision can recover additional AMBIGUOUS cases, but gains are task/seed dependent. Lexical expansion is modest; volume alone does not explain relation gains; the bundled full curriculum can reduce ambiguity detection and increase incorrect publication. | Broader supervision is useful signal, not a solved recipe. The next bounded question is **balanced, stable relation-family training** that preserves NO_MATCH/publication behavior. |

## Latest canonical result: GDM55

GDM55 is an **ambiguity-curriculum-only experiment**. It keeps the canonical GDM51 structured capability/ambiguity architecture, GDM50 numerical path, explicit NONE objective, `1e-4` frozen-feature grid, deterministic schema/operation realization, and GDM54 5pp ambiguity-rescue policy fixed. Only the ambiguity-head training curriculum changes.

The five arms are:

- `canonical-rescue-control`: the canonical GDM51 ambiguity curriculum;
- `volume-matched-control`: repeats canonical semantics at the larger update budget;
- `lexical-expanded-rescue`: broader paraphrases over canonical relation families;
- `relation-expanded-rescue`: additional ambiguity relation families at the same expanded update budget;
- `full-curriculum-rescue`: lexical + relation diversity + matched disambiguating counterfactual negatives + training-only domain randomization.

The full curriculum is a bundled treatment. GDM55 does **not** identify which ingredient in that bundle causes any effect.

### Data and denominators

Environment/Pipeline is calibration-only. Campaign/Profile is the fresh GDM55 secondary holdout and is now regression-only for subsequent experiments. Each task has **84 unique holdout cases**: 36 answerable, 32 NO_MATCH, and 16 AMBIGUOUS. Rates below are means over seeds `5501,5502`; status counts pool those two seeds from one measured attempt. Therefore `17/32` means sixteen unique ambiguous cases evaluated by two independently initialized models—not thirty-two unique semantic worlds. The exact reproducibility rerun does not increase sample size.

### Operation generation

| Curriculum | Answerable accuracy | Risk accuracy | Incorrect publication | Target precision | Target recall | AMBIGUOUS | NO_MATCH |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Canonical rescue | 30.56% | 43.75% | 11.90% | 100.00% | 22.66% | 12/32 | 30/64 |
| Volume matched | 31.94% | 43.75% | 12.50% | 100.00% | 23.44% | 10/32 | 32/64 |
| Lexical expanded | 34.72% | 44.79% | 14.29% | 98.08% | 29.69% | 10/32 | 33/64 |
| Relation expanded | 33.33% | 42.71% | **11.31%** | **100.00%** | 24.22% | **17/32** | 24/64 |
| Full curriculum | **37.50%** | 38.54% | 22.62% | 92.26% | **37.50%** | 8/32 | 29/64 |

Operation relation expansion recovers **7 more AMBIGUOUS decisions than the volume-matched control (17/32 vs 10/32)** while using the same ambiguity optimizer-update budget, which is evidence that semantic relation diversity—not exposure alone—matters in this bounded setting. The gain is not free: NO_MATCH falls from 32/64 to 24/64. The full curriculum has the highest answerable accuracy and target recall but doubles incorrect publication relative to relation expansion and detects fewer ambiguities.

Multi-clause exact correctness remains weak. For the relation-expanded arm, correct answerable requests are **17/32 one-clause, 7/24 two-clause, and 0/16 three-clause**. For full curriculum they are 17/32, 9/24, and 1/16. This is a persistent request-level compounding problem, not solved by ambiguity curriculum alone.

### Schema generation

| Curriculum | Answerable accuracy | Risk accuracy | Incorrect publication | Target precision | Target recall | AMBIGUOUS | NO_MATCH |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Canonical rescue | 40.28% | 40.62% | 29.76% | 89.17% | 38.28% | 4/32 | 35/64 |
| Volume matched | 40.28% | 42.71% | 25.60% | 90.95% | 35.94% | 8/32 | 33/64 |
| Lexical expanded | 41.67% | **47.92%** | 26.19% | 89.29% | 39.06% | 10/32 | **36/64** |
| Relation expanded | 44.44% | 46.88% | **24.40%** | **92.06%** | 40.62% | **10/32** | 35/64 |
| Full curriculum | **48.61%** | 33.33% | 31.55% | 89.28% | **45.31%** | 0/32 | 32/64 |

Schema relation expansion improves answerable accuracy, target precision/recall, ambiguity recovery, and incorrect publication relative to the volume-matched control. Lexical expansion slightly wins aggregate risk accuracy because it retains one more NO_MATCH. The full curriculum again shows interference: it improves answerable recall but collapses AMBIGUOUS detection to **0/32** and has the highest incorrect-publication rate in the schema sweep.

For schema relation expansion, correct answerable requests are **19/32 one-clause, 10/24 two-clause, and 3/16 three-clause**. Full curriculum reaches 20/32, 12/24, and 3/16. Longer requests remain substantially harder.

### Ambiguity-family behavior

Counts below pool seeds; each family has eight decisions (four unique cases × two seeds).

| Task / curriculum | Lifecycle-time | Object-vs-supplier | Representation | Role |
| --- | ---: | ---: | ---: | ---: |
| Operation / canonical | 4/8 | 0/8 | 2/8 | 6/8 |
| Operation / volume | 4/8 | 0/8 | 1/8 | 5/8 |
| Operation / lexical | 4/8 | 0/8 | 2/8 | 4/8 |
| Operation / relation | 4/8 | **2/8** | **4/8** | **7/8** |
| Operation / full | 4/8 | 0/8 | 2/8 | 2/8 |
| Schema / canonical | 4/8 | 0/8 | 0/8 | 0/8 |
| Schema / volume | 5/8 | 0/8 | 2/8 | 1/8 |
| Schema / lexical | 4/8 | **2/8** | **4/8** | 0/8 |
| Schema / relation | **6/8** | **2/8** | 2/8 | 0/8 |
| Schema / full | 0/8 | 0/8 | 0/8 | 0/8 |

Relation-family expansion is the first tested curriculum in this line to recover held-out `object-vs-supplier` ambiguity in both tasks. It also improves operation role/representation ambiguity. But family behavior is **seed unstable**: for operation relation expansion, seed 5501 gets 14/16 ambiguous cases correct while seed 5502 gets 3/16; schema relation expansion moves in the opposite direction (2/16 versus 8/16). The next experiment should target stability rather than simply add more heterogeneous examples.

### Semantic errors and regressions

The operation canonical, volume, and relation-expanded arms have no accepted semantic-confusion entries in the GDM55 aggregate. Lexical expansion has one author/moderator confusion. Full curriculum has author/moderator and created/updated confusions. Schema arms continue to confuse lifecycle time with review rating in some accepted requests, and the full curriculum also introduces author-name versus author-ID errors.

Regression accuracy is not fresh evidence: operation canonical / volume / lexical / relation / full is **45.74% / 47.26% / 47.11% / 42.48% / 46.12%**; schema is **43.16% / 42.10% / 43.09% / 43.39% / 45.14%**. Those suites contain previously inspected GDM46–GDM54 holdouts.

### Speed, memory, and numerical execution

Attempt 1 shows no meaningful architectural latency/memory separation among the curriculum-only arms. Operation mean per-seed p50 is **63.23–65.34 ms**, p95 **122.96–124.43 ms**, and process RSS **646,630–660,894 KiB**. Schema p50 is **61.12–62.01 ms**, p95 **119.59–123.15 ms**, and RSS **650,648–668,300 KiB**. The exact audit also records attempt-2 hosted-runner variation, especially schema latency, while deterministic predictions/hashes remain exact.

These timings cover fresh request-clause encoding + learned feature-space scoring + deterministic generation/local validation with catalog/NONE embeddings cached. They exclude model download, initial embedding/index construction, backend fixture execution, and Rover composition. RSS is process-level and affected by earlier work in the same worker; it is not isolated model residency.

The canonical execution path fixes PyTorch/OMP/MKL thread settings, default CPU dispatch, deterministic algorithms, and disables oneDNN. It uses non-foreach/non-fused AdamW and non-foreach clipping. AdamW moments and stored head parameters remain float32, while the final bias-corrected parameter application is recomputed in float64 and cast back to float32. **This remains feature-head training over a frozen transformer, not transformer fine-tuning.**

### Canonical interpretation

GDM55 supports three bounded conclusions:

1. **More optimizer exposure alone is not enough.** The volume-matched control does not reproduce the operation ambiguity gain from relation expansion.
2. **Relation-family diversity can help**, especially operation ambiguity and schema answerable/precision tradeoffs, but the effect is unstable across seeds and costs NO_MATCH recall in operation.
3. **More heterogeneous data is not monotonically better.** The bundled full curriculum can improve answerable recall while damaging ambiguity detection and publication safety; schema AMBIGUOUS falls to 0/32.

This does **not** prove relation expansion is the final curriculum, nor that counterfactual negatives or domain randomization are intrinsically harmful. The full arm changes several curriculum ingredients together. The next bounded hypothesis is to **factor and balance relation-family training while holding the expanded update budget fixed**, with explicit seed-stability and NO_MATCH/publication constraints.

## What the metrics mean

**Answerable accuracy** is exact request correctness among reference-answerable cases, including the executable/requirements contract—not merely deciding to accept. **Risk accuracy** requires the exact reference status (`NO_MATCH` versus `AMBIGUOUS`), not just any refusal. **Incorrect publication** counts accepted-but-wrong requests over **all evaluated requests**, not only published requests.

**Target precision and recall are counted only on reference-answerable cases.** Rejection yields an empty predicted target set, reducing recall. Consequently, 100% target precision can coexist with incorrect publication on unsupported or ambiguous requests. The [collector definitions](experiments/followup49/collect.py) are authoritative; do not present these precision figures as end-to-end publication safety.

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
| **GDM55** | [db13313](https://github.com/burn2delete/graph-model/commit/db13313e7fd5021922768add76aaac617ec21208) | [35755096818, attempts 1/2](https://github.com/burn2delete/graph-model/actions/runs/35755096818) | [35762380169](https://github.com/burn2delete/graph-model/actions/runs/35762380169) |

GDM55 attempt-1 BATCH_REPORT artifact [10708637460](https://github.com/burn2delete/graph-model/actions/runs/35755096818/artifacts/10708637460) has ZIP SHA256 `54c8a2893521c18d6bc0ae527ca4c4aee187b3adcf99121e7d41b6e53823dd51`; attempt-2 report artifact [10710285394](https://github.com/burn2delete/graph-model/actions/runs/35755096818/artifacts/10710285394) has ZIP SHA256 `eacb659e4e3a63c9650ad1d71fc39e0a4a9a8cd9c885f9bae8e56abe9133c265`.

The independent GDM55 audit artifact [10710766560](https://github.com/burn2delete/graph-model/actions/runs/35762380169/artifacts/10710766560) has ZIP SHA256 `1102fb7776ee0108c68d5fe735fe1f2a02a7b6c927b7f8932093010eb9ae9d48`. Its `REPRODUCIBILITY_REPORT.json` records `complete=true`, `evidence_verified=true`, `reproducible=true`, `promotion_eligible=true`, `errors=[]`, and exact **operation 10/10 + schema 10/10** matches. All twenty comparisons have empty deterministic `differing_fields`; canonical and raw feature comparisons also have no differences in this audit.

The earlier GDM55 source commit `52ffdd8ef2aaf009d56a37bf53b38b3a3c0444e6` / run `35749924051` failed only in preflight because a test expected eight instead of the deliberately generated sixteen AMBIGUOUS holdout rows. Smoke and measured workers were skipped, so it is preserved as an implementation diagnostic—not model evidence. The in-place repair at canonical source `db13313...` changed only that test expectation.

Artifact retention is finite; a link or this README summary is not a substitute for retained evidence when re-verification is needed.

## Next bounded hypothesis, not a result

GDM55 makes a narrower next step preferable to another architecture change: **factor and balance the relation-expanded ambiguity curriculum** while preserving the GDM55 head architecture, numerical path, capability objective, schema realization, and rescue policy. The experiment should distinguish relation-family balance from counterfactual negatives and training-domain randomization, keep optimizer-update exposure matched, and make seed stability plus NO_MATCH/publication preservation explicit selection criteria.

No GDM56 evidence exists at the time of this synchronization. Campaign/Profile is now inspected and must be regression-only in later work; any next batch requires fresh calibration and secondary-holdout domains.

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
| [experiments/followup55/](experiments/followup55/) | Canonical ambiguity-curriculum generalization experiment |
| [.github/workflows/](.github/workflows/) | Actions-only preflights, measured batches, diagnostics, and exact-attempt audits |

Use the declared workflow and its execution wrapper when reproducing an experiment: calling a historical `run.py` directly can bypass the canonical numerical configuration. Inspect the live queue before dispatching anything. No local compilation, tests, training, validation, or benchmarks are part of the accepted research workflow.