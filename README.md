# Graph Model

**A GraphQL Decision Model (GDM): learn the decisions that satisfy a request, then construct and evaluate real GraphQL.**

This repository researches models that reason about GraphQL and generate **operations and schemas**. Query planning is out of scope.

The measured system is deliberately narrower than a general-purpose GraphQL generator: a **frozen semantic encoder plus learned capability and ambiguity heads** selects catalog-grounded capabilities, then deterministic code realizes those capabilities as executable GraphQL operations or Federation schemas. The transformer backbone is not fine-tuned, and schema generation does not invent arbitrary ontologies.

## Current research state

_Last synchronized: 2026-09-23. Latest canonical experiment: GDM60._

| State | Meaning |
| --- | --- |
| **Canonical through GDM60** | GDM50–GDM60 each have repeated measured evidence plus a passing exact-attempt reproducibility audit. Canonical means usable reproducible evidence under the declared contract; it does **not** mean every tested arm is a winning architecture. |
| **Latest supported result** | GDM60 shows that simple capability-confidence scaling of the GDM59 rank-preserving request-relative representation is **negative** on the fresh holdout: it improves neither task’s answerable/safety tradeoff. Factorizing the representation is task-sensitive: the request/candidate product block is much more useful for operations, while the absolute request/candidate delta block is the strongest schema factorization and slightly improves the within-batch schema control on every aggregate metric except AMBIGUOUS, which is unchanged. |
| **Remaining tradeoff** | Operation `ranked-product-only` recovers substantial answerable/recall and multi-clause correctness but gives back some risk/publication safety. Schema `ranked-delta-only` gives only a small aggregate gain. Name-vs-ID, role, and lifecycle semantic confusions remain. |
| **Next unverified hypothesis** | The next bounded question is whether a **fixed balance between the rank-preserving product and delta blocks** can retain the operation product signal and schema delta signal without the severe recall loss caused by capability-confidence weighting. This is not yet a result. |
| **Invalid historical evidence** | Original GDM41–44 jobs wrote manifests and fabricated/simulated metrics. Their promotions are withdrawn. Only measured repair run `35565425124` (196/196 verified) is the valid GDM41–44 baseline. |

Read [`experiments/MEASUREMENT_POLICY.md`](experiments/MEASUREMENT_POLICY.md) and [`experiments/measured/REPAIR.md`](experiments/measured/REPAIR.md) before interpreting results or changing experiments.

## Architecture

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
             Clause-candidate pair features
                    |
      Learned residual feature adapter + scorer
         listwise scores over real paths + NONE
                    |
        Ranked real candidates + NONE evidence
                    |
          Learned clause-local ambiguity head
        eleven scalar/structural evidence features
        + matched-capacity semantic representation
                    |
         Validation-calibrated thresholds and
         deterministic clause/status arbitration
                    |
          Deterministic request aggregation
              /                      \
     NO_MATCH / AMBIGUOUS       accepted capability paths
                                      |
                              Typed deterministic realization
                                  /                 \
                       GraphQL operation       projected SDL/subgraphs
                                  |                 |
                       parse/validate/execute   requirements + Rover
                                  \                 /
                         per-example evidence and metrics
```

### Public inputs and bounded capabilities

Each example contains a task (`operation` or `schema`), a natural-language request, and a typed catalog of available capabilities. A capability is a **full GraphQL path** with coordinates, type signatures, and a public semantic description. Paths such as `reviews.author.name`, `reviews.author.id`, and `reviews.moderator.name` are distinct response contracts even when they share terminal field names.

The benchmark is synthetic and catalog-bounded. Fresh domain names and paraphrases are screening controls, not evidence of enterprise out-of-distribution generalization or arbitrary graph-topology generation.

### Clause decomposition

Multi-capability requests are decomposed using public request syntax into clause-local decisions. This is deterministic benchmark parsing, not a learned natural-language decomposition model. The line keeps clause-local scoring because GDM45 showed whole-request capability scoring degrades sharply as requested capability count grows.

### Frozen semantic encoder and numerical boundary

The active line uses `distilbert/distilbert-base-uncased` at the exact revision recorded in each result. The transformer is frozen. Mean-pooled normalized representations cross an explicit **`1e-4` canonical feature boundary** before learned-head training. Request clauses are encoded fresh during timed generation; catalog and NONE vectors are cached.

The CPU/numerical contract fixes one-thread/default CPU dispatch, disables oneDNN, uses PyTorch 2.8 single-tensor AdamW with `foreach=false` and `fused=false`, non-foreach gradient clipping, float32 optimizer moments/state, and evaluates the final decoupled-weight-decay plus bias-corrected parameter application in float64 before storing float32 parameters. This is a reproducibility contract, not transformer fine-tuning.

### Learned capability head and explicit NONE

For a request-clause vector `q` and candidate vector `c`, the capability scorer consumes request/candidate representations, elementwise interaction, absolute difference, catalog-relative candidate information, and structural GraphQL features. A residual feature adapter and MLP score every real capability plus an explicit **NONE** outcome.

Unsupported requests can therefore train toward NONE rather than being forced onto a real field. These are learned feature-space heads over frozen representations. Separate operation/schema checkpoints do not imply separately fine-tuned transformer backbones.

### Learned ambiguity head

GDM51 introduced a separate clause-local ambiguity discriminator after capability training. GDM57 established useful semantic ambiguity signal; GDM58 showed that simply widening a fixed pooled summary is not enough; GDM59 showed that preserving top-five rank in request-relative interactions is materially better than pooling.

GDM60 keeps **top-five candidate breadth, rank preservation, and the same ambiguity-head capacity** (`11 + 10*d`, or 7,691 DistilBERT inputs) in every arm. It changes only the factorization/scaling of the two request-relative semantic blocks:

- `ranked-request-control`: canonical GDM59 `q*c_i` plus `abs(q-c_i)` blocks;
- `ranked-product-only`: `q*c_i` with a zero-filled delta block;
- `ranked-delta-only`: `abs(q-c_i)` with a zero-filled product block;
- `ranked-confidence-weighted`: both blocks scaled by candidate capability probability `p_i`;
- `ranked-relative-confidence`: both blocks scaled by clipped `p_i/p_1`, preserving rank-1 scale while attenuating lower-confidence ranks.

Missing ranks are zero-padded. Capability training/ranking, the GDM56 family-balanced ambiguity curriculum, GDM54 rescue calibration, numerical execution, and deterministic GraphQL realization are fixed. GDM60 therefore isolates **request-relative representation factorization/confidence scaling**, not candidate breadth, head size, curriculum, or backbone.

### Validation-only arbitration

The capability and ambiguity heads are optimizer-trained. Thresholds and request arbitration are not. Validation-only calibration chooses NONE and ambiguity thresholds; the line retains the GDM54 high-confidence ambiguity-rescue policy with a 5 percentage-point NO_MATCH-recall budget and accepted-status recall guardrail.

Request aggregation is deterministic: a remaining NO_MATCH clause rejects the request; otherwise any AMBIGUOUS clause makes the request ambiguous; otherwise selected paths are emitted.

### Deterministic GraphQL realization

Accepted operation capabilities are converted to a closed selection tree, parsed/validated with `graphql-core`, and executed against changing fixtures that distinguish author/moderator, name/ID, and creation/update mistakes.

Accepted schema capabilities are projected into minimal SDL/subgraphs with deterministic object/key closure and ownership conventions. Rover composes the generated subgraphs, and a separate requirements evaluator checks requested capabilities. **Composition success is not request correctness.** Schema generation remains catalog projection plus deterministic SDL/Federation realization.

## What the experiment line has taught us

| Experiment | Measured learning | Consequence / limit |
| --- | --- | --- |
| GDM45 | Clause-local decomposition improves multi-capability generation relative to whole-request scoring. | Keep clause-local decisions. Its schema risk/status path had a shortcut, so only accepted-case evidence is retained. |
| GDM46–49 | Strong rejection gates can improve risk handling while destroying answerable recall. | Risk and answerable correctness must be reported separately. |
| GDM50 | Explicit NONE supplies an open-set unsupported outcome, but does not solve ambiguity or multi-clause compounding. Numerical diagnostics exposed cross-run optimizer drift. | Keep explicit NONE and the hardened numerical contract. |
| GDM51 | A separate ambiguity discriminator improves capability-vs-risk decomposition. | Ambiguity deserves a learned component; representation matters. |
| GDM52–54 | Calibration/arbitration expose precision/recall tradeoffs; threshold ordering alone cannot repair weak ambiguity evidence. | Keep calibration effects separate from representation capability. |
| GDM55–56 | Semantic curriculum diversity and family balance matter, especially for operation ambiguity, but more curriculum is not monotonic and counterfactual-negative arms can collapse ambiguity recovery. | Curriculum alone is not the missing representation. |
| GDM57 | Request-conditioned top-two semantics sharply improve AMBIGUOUS recovery; combined candidate+request semantics improve schema risk balance. | Useful semantic signal exists, but top-two evidence still trades AMBIGUOUS against NO_MATCH/publication safety. |
| GDM58 | Wider top-k evidence through the same pooled representation produces almost no operation benefit and does not improve schema answerable/ambiguity tradeoff over top-2. | Simple pooled breadth is negative/inconclusive. |
| GDM59 | Rank-preserving top-five request interactions outperform the matched pooled control on answerable correctness/recall while retaining strong ambiguity recovery. | Rank identity helps through request-relative interaction, not raw candidate slots; publication and semantic-confusion tradeoffs remain. |
| **GDM60** | **Simple capability-confidence weighting of ranked request interactions loses substantial answerable recall. The product block is the strongest operation factorization; the delta block is the strongest schema factorization and slightly improves the within-batch schema request-control across aggregate accuracy/safety/precision/recall metrics.** | **Confidence attenuation is negative under this contract. Product-vs-delta usefulness is task-sensitive, but the schema delta gain is small and neither factorization solves semantic families or multi-clause correctness.** |

# Latest canonical result: GDM60

GDM60 changes **only the request-relative semantic factorization/confidence scaling supplied to the learned ambiguity head**. Every arm sees the same top-five candidates, preserves candidate rank, uses the same 7,691-input head capacity, and keeps explicit NONE, capability training, the GDM56 family-balanced ambiguity curriculum, frozen DistilBERT, the `1e-4` feature boundary, canonical numerical execution, GDM54 rescue arbitration, and deterministic GraphQL/Federation realization fixed.

## GDM60 data and denominators

`Pass` / `Logbook` are calibration-only. `Certificate` / `Manuscript` are the fresh GDM60 secondary holdout and are now **regression-only** for subsequent work. GDM46–GDM59 inspected holdouts also remain regression-only.

Each task has **84 unique secondary-holdout cases**: 36 answerable, 32 NO_MATCH, and 16 AMBIGUOUS. Rates are means over seeds `6001,6002`. Status counts pool those seeds: for example, `22/32` AMBIGUOUS represents sixteen unique ambiguous cases evaluated by two independently initialized models, not thirty-two independent semantic worlds. The exact same-source rerun is reproducibility evidence and does not add semantic cases.

The collector independently recomputes role, lifecycle-time, representation/name-vs-ID, and object-vs-supplier family metrics from per-example predictions, and the exact-attempt audit requires them to match between attempts. The top-level batch report does not pool family values across seeds, so no aggregate family count is invented here. Residual accepted errors directly show name-vs-ID, role, and lifecycle distinctions remain unresolved.

## Operation generation

| Representation | Answerable accuracy | Risk accuracy | Incorrect publication | Target precision | Target recall | AMBIGUOUS | NO_MATCH |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ranked request control | 23.61% | **59.38%** | **8.33%** | **100.00%** | 18.75% | **19/32** | **38/64** |
| Ranked product only | **38.89%** | 56.25% | 12.50% | 98.28% | **35.16%** | 16/32 | **38/64** |
| Ranked delta only | 19.44% | 58.33% | 10.12% | **100.00%** | 14.06% | 18/32 | **38/64** |
| Capability-confidence weighted | 19.44% | 58.33% | 10.12% | **100.00%** | 14.06% | 18/32 | **38/64** |
| Relative-confidence weighted | 31.94% | 57.29% | 11.90% | **100.00%** | 26.56% | 17/32 | **38/64** |

`ranked-product-only` is the strongest operation answerable/recall factorization: versus the within-batch request-control it gains **15.28 percentage points answerable accuracy** and **16.41 points target recall**. Exact correctness improves from **11/32, 5/24, 1/16** for one-/two-/three-clause requests to **15/32, 10/24, 3/16**. The cost is lower risk accuracy (-3.13 points), higher incorrect publication (+4.17 points), lower target precision (-1.72 points), and AMBIGUOUS recovery falling from 19/32 to 16/32. This is useful factorization evidence, not a free architecture win.

Both confidence-scaled representations are negative for the intended safety hypothesis. Absolute probability weighting collapses operation answerable accuracy to 19.44% and recall to 14.06%; relative confidence recovers some answerable cases but remains below product-only and does not improve risk/publication versus the request-control.

The product-only arm still publishes a lifecycle confusion (`updatedAt` as `createdAt`). The stronger answerable result therefore does not establish semantic correctness once a request is accepted.

## Schema generation

| Representation | Answerable accuracy | Risk accuracy | Incorrect publication | Target precision | Target recall | AMBIGUOUS | NO_MATCH |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ranked request control | 50.00% | 58.33% | 20.24% | 95.12% | 50.78% | **22/32** | 34/64 |
| Ranked product only | 51.39% | 54.17% | 23.81% | 93.10% | 51.56% | 17/32 | 35/64 |
| Ranked delta only | **51.39%** | **59.38%** | 19.64% | **95.74%** | **52.34%** | **22/32** | 35/64 |
| Capability-confidence weighted | 29.17% | 58.33% | **18.45%** | 93.10% | 29.69% | 20/32 | **36/64** |
| Relative-confidence weighted | 38.89% | 58.33% | 19.05% | 95.12% | 39.06% | 21/32 | 35/64 |

`ranked-delta-only` is the strongest schema factorization in this batch. Relative to the request-control, it gains **1.39 points answerable accuracy**, **1.04 points risk accuracy**, reduces incorrect publication by **0.60 points**, raises target precision by **0.62 points**, raises recall by **1.56 points**, and improves NO_MATCH from 34/64 to 35/64 while retaining 22/32 AMBIGUOUS. Exact correctness changes from **19/32, 11/24, 6/16** to **19/32, 12/24, 6/16**. The gain is therefore broad but small; it does not justify claiming a solved architecture.

`ranked-product-only` reaches the same 51.39% answerable accuracy but loses risk accuracy, precision, publication safety, and AMBIGUOUS recovery. Both confidence-scaled arms substantially reduce answerable recall; their modest publication-rate improvements do not compensate for the lost answerable capability under this contract.

Accepted schema errors still include `author.id` published as `author.name` and moderator requests published as author names. GDM60 therefore does not establish resolution of representation/name-vs-ID or role ambiguity, and the aggregate report is insufficient to claim any family solved.

## Regression, holdout, and interpretation limits

All previously inspected suites, including GDM59 `Warranty` / `Dossier` and now GDM60 `Certificate` / `Manuscript`, are **regression-only** for subsequent experiments. They are not a causal cross-batch leaderboard. GDM60 mean regression request accuracy spans approximately **45.27%–48.65% for operation** and **43.41%–47.68% for schema** across arms.

The very different fresh-holdout baseline of GDM60 versus GDM59 is exactly why conclusions are made **within each batch**, not by treating unlike synthetic domains as a longitudinal leaderboard. GDM60 tests only deterministic per-rank block factorization and simple capability-confidence scaling. It does not test learned attention over candidate sets, a new request-global status model, unrestricted GraphQL token generation, or arbitrary schema invention.

## Speed and memory

Attempt 1 mean-per-seed warm-generation measurements are observational:

| Task | p50 range across arms | p95 range across arms | mean RSS-after-evaluation range |
| --- | ---: | ---: | ---: |
| Operation | 70.00–70.84 ms | 138.43–139.47 ms | 1,099,514–1,101,642 KiB |
| Schema | 69.27–70.60 ms | 136.64–139.66 ms | 1,100,334–1,104,810 KiB |

Generation timing covers fresh request-clause encoding, learned-head scoring, deterministic GraphQL realization, and local validation with catalog/NONE vectors cached. Downloads, initial catalog embedding, backend fixture execution, and Rover composition are excluded from these quantiles; composition is recorded separately. RSS is process-level and affected by worker reuse, so it is **not isolated model memory**. Timing/memory are observational and need not bit-match across hosted-runner attempts; deterministic state/predictions do.

## GDM60 reproducibility and provenance

GDM60’s canonical measured source is commit [`c04784ec33983829f6fe55506bcb74e0f40442b2`](https://github.com/burn2delete/graph-model/commit/c04784ec33983829f6fe55506bcb74e0f40442b2), source run [`35918183757`](https://github.com/burn2delete/graph-model/actions/runs/35918183757), attempts 1 and 2. Each attempt independently verified **20/20** declared results with no missing, failed, or unexpected configurations. Both preflights passed 67 current/inherited tests plus the deterministic `ranked-relative-confidence` hash smoke; that smoke is contract evidence, not pretrained-model performance.

- Attempt 1 `BATCH_REPORT` artifact **10777266084**, ZIP SHA-256 `9e4db92b71060fa9adac6a0edae8e1581b76bba81ffc6c67998f857c1372483f`.
- Attempt 2 `BATCH_REPORT` artifact **10779222122**, ZIP SHA-256 `b63d54094c15e039d078f2fa84c26127eea48108a1fdd967ff616e5949ab1975`.
- Passing exact-attempt audit run [`35929537000`](https://github.com/burn2delete/graph-model/actions/runs/35929537000), audit workflow commit [`765e156ec33231d38d96552dc81e8e0fb955a9ba`](https://github.com/burn2delete/graph-model/commit/765e156ec33231d38d96552dc81e8e0fb955a9ba).
- Passing audit artifact **10781490854**, ZIP SHA-256 `ed4b27f37b6c85b27359aa40d681cc36705d3ae7c3ba7e24c187e70868155f70`.

The actual audit report is `gdm60-reproducibility-audit-v1` with `complete=true`, `evidence_verified=true`, `reproducible=true`, `promotion_eligible=true`, `errors=[]`, exact **operation 10/10 + schema 10/10**, and all 20 comparisons `exact_match=true`, `differing_fields=[]`, with `raw_feature_differences=[]`. The audit independently reconstructed both batch reports and verified attempt provenance, actual initial/selected full/capability/ambiguity checkpoint hashes, changed learned heads, optimizer/training/calibration records, matched 7,691-input top-five representation receipts, fixed curriculum/numerical/calibration contracts, deterministic regression/holdout/clause/family metrics, exact per-example predictions, and canonical fixed-probe/full-corpus hashes. The audit itself used bounded sequential/disk-backed comparison; its process-memory diagnostics are audit-runtime observations, not model-memory measurements.

Artifact retention is finite. IDs and digests record provenance but do not substitute for retained source evidence if future re-verification is required.

## Prior canonical result: GDM59

GDM59 established that preserving top-five candidate rank is useful primarily through request-relative semantic interactions rather than raw candidate slots. On its fresh holdout, `ranked-request-only` materially improved answerable correctness and target recall versus the top-five pooled control and improved all three schema clause-count buckets, but paid a modest publication/precision cost.

GDM59’s measured source is `035f44836d850b24f4bebe5cd499a572b6052752`, source run `35898256989` attempts 1/2, BATCH_REPORT artifacts **10769046397** (`b4a2237c074588a61031eb957ede60780b41bf5afc3b020d9c26768149ab5b20`) and **10770769769** (`c11e48f18b56d0e28f03720a9a7579957cbc97402642d591e1644789a656b312`), passing audit run `35910892558`, and audit artifact **10773722103** (`66dfea1e8bce33a06c77a80dbb287a29df2ddaba68d57aed289e51899f2b90a3`).

## Prior canonical result: GDM58

GDM58 changed only how many ranked candidates entered a fixed permutation-invariant pooled semantic representation. It found no operation answerable/precision/recall/publication benefit from wider pooled breadth and no schema answerable/ambiguity improvement over the top-two pooled control. The earlier GDM58 2-GiB archive-guard failure and repeated runner-shutdown audit attempts remain preserved as audit/runtime diagnostics, not source-model failures; the bounded-resource exact audit passed without weakening equality.

GDM58’s measured source is `9ba9caab957c1802feb17776195141f630fd96eb`, source run `35819165142` attempts 1/2, BATCH_REPORT artifacts **10732952674** (`490d4ddcf88bbb517186c15d4dc7f85c643b3c75d480b27409d59f4c7a5c2a1a`) and **10735225675** (`36adadbd454fe064b20242e9ce1d8bdcbabfa4537520192b666331cf5ce6aaff`), passing audit run `35890238431`, and audit artifact **10765506374** (`b3f2f2273f8eca3cf6e12ba04b93144dcc51f59d327a47d91f96dc2bcb0c7163`).

## Canonical provenance

| Experiment | Canonical measured source/run | Passing exact-attempt audit |
| --- | --- | --- |
| GDM41–44 repair | run `35565425124`, 196/196 measured repair | Original fabricated runs excluded |
| GDM50 | source `62d2720800f5d98e1521e0200766a80e998ebe9b`, run `35680386718` | `35687452276` |
| GDM51 | source `f52baf82222947ca00139437e90fe9b5428bd9ae`, run `35687697452` | `35695965127` |
| GDM52 | source `f0ca439c3bb567213108b55a3bd56e449425509b`, run `35702079193` | `35711502477` |
| GDM53 | source `97b8ceb7c3820690545873e9ced2854ad1e26cee`, run `35717392980` | `35728895825` |
| GDM54 | source `031c7f6087eb356f8baeeedcb2bf25ce9fe35cb4`, run `35729681954` | `35748761095` |
| GDM55 | source `db13313e7fd5021922768add76aaac617ec21208`, run `35755096818` | `35762380169` |
| GDM56 | source `168317f6495b05e04597ae084b75be6890fe0621`, run `35774902004` | `35798481278` |
| GDM57 | source `3f8b9a968285110ca6c744590724ddd0cb2bd5aa`, run `35803609103` | `35811475866` |
| GDM58 | source `9ba9caab957c1802feb17776195141f630fd96eb`, run `35819165142` | `35890238431` |
| GDM59 | source `035f44836d850b24f4bebe5cd499a572b6052752`, run `35898256989` | `35910892558` |
| **GDM60** | source `c04784ec33983829f6fe55506bcb74e0f40442b2`, run `35918183757` | **`35929537000`** |

## Evidence and promotion policy

All accepted compilation, tests, training, validation, benchmarks, and independent evidence checks run in **GitHub Actions**. Source/artifact inspection may analyze already-produced evidence but does not replace an Actions verifier.

A canonical promotion requires complete declared results, changed learned-head checkpoint evidence, optimizer histories, validation-only checkpoint/calibration selection, per-example predictions/generated GraphQL, real execution/composition checks, raw timing/runtime metadata, an exact same-source/same-seed rerun, and a dedicated independent exact-attempt audit. The root README must be synchronized **before** announcing promotion or launching the next numbered experiment.

Fresh holdouts never select optimizer behavior, checkpoints, thresholds, curriculum, or representation changes. Once inspected, they become regression-only. Repeated seeds and exact reruns are reproducibility evidence, not extra semantic worlds.

## Current limits and next bounded question

The measured system remains a small synthetic catalog-projection research environment. It has not established unrestricted schema invention, general mutation/subscription generation, arbitrary enterprise topology handling, transformer fine-tuning benefits, or production-scale latency/memory behavior.

GDM60 rejects the simplest safety idea after GDM59: **multiplying rank-preserving request-relative semantics by capability confidence does not recover safety without erasing answerable capability**. The useful signal is more structured. Operations benefit substantially from the product interaction block, whereas schemas get their best factorized tradeoff from the absolute-difference block. The schema effect is small, and residual semantic confusions remain, so this is evidence about representation components rather than a final task-specific architecture.

The next unverified hypothesis is a bounded **interaction-block balance** experiment: preserve top-five rank, head capacity, curriculum, encoder, numerical path, calibration, and GraphQL realization, then vary only fixed relative scaling between the product and absolute-difference blocks. A suitable matched-capacity set should include product-only, product-dominant, balanced GDM59 control, delta-dominant, and delta-only arms on fresh calibration/secondary-holdout domains. This directly tests whether the task-specific GDM60 endpoints are robust or whether an intermediate fixed balance gives a better answerable/safety frontier. It is a hypothesis, not yet a result.

## Repository map

| Location | Purpose |
| --- | --- |
| [`experiments/MEASUREMENT_POLICY.md`](experiments/MEASUREMENT_POLICY.md) | Execution, evidence, scope, and mandatory README-promotion rules |
| [`experiments/measured/`](experiments/measured/) | Repaired baseline, frozen encoders, shared heads, GraphQL contracts/evaluation |
| [`experiments/followup50/`](experiments/followup50/) | Explicit NONE, cache boundary, canonical numerical execution |
| [`experiments/followup51/`](experiments/followup51/) | Capability/ambiguity staged training and structured ambiguity features |
| [`experiments/followup52/`](experiments/followup52/), [`followup53`](experiments/followup53/), [`followup54`](experiments/followup54/) | Calibration and arbitration experiments |
| [`experiments/followup55/`](experiments/followup55/) | Ambiguity curriculum generalization |
| [`experiments/followup56/`](experiments/followup56/) | Relation-family balance/factorization |
| [`experiments/followup57/`](experiments/followup57/) | Top-two semantic ambiguity representation |
| [`experiments/followup58/`](experiments/followup58/) | Pooled top-k ambiguity evidence |
| [`experiments/followup59/`](experiments/followup59/) | Rank-preserving top-five ambiguity representation |
| [`experiments/followup60/`](experiments/followup60/) | Request-relative factorization and confidence scaling, latest canonical experiment |
| [`.github/workflows/`](.github/workflows/) | Actions-only measured batches and exact-attempt audits |

Use each experiment’s declared workflow/execution wrapper when reproducing it. Calling historical Python entrypoints directly can bypass the canonical numerical contract.
