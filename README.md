# Graph Model

**A GraphQL Decision Model (GDM): learn the decisions that satisfy a request, then construct and evaluate real GraphQL.**

This repository researches models that reason about GraphQL and generate **operations and schemas**. Query planning is out of scope.

The measured system is deliberately narrower than a general-purpose GraphQL generator: a **frozen semantic encoder plus learned capability and ambiguity heads** selects catalog-grounded capabilities, then deterministic code realizes those capabilities as executable GraphQL operations or Federation schemas. The transformer backbone is not fine-tuned, and schema generation does not invent arbitrary ontologies.

## Current research state

_Last synchronized: 2026-09-23. Latest canonical experiment: GDM59._

| State | Meaning |
| --- | --- |
| **Canonical through GDM59** | GDM50–GDM59 each have repeated measured evidence plus a passing exact-attempt reproducibility audit. Canonical means usable reproducible evidence under the declared contract; it does **not** mean every tested arm is a winning architecture. |
| **Latest supported result** | GDM59 shows that preserving candidate rank/identity in **request-conditioned top-five semantic interactions** materially improves answerable correctness and target recall versus a matched-capacity top-five pooled control, while retaining strong AMBIGUOUS recovery. Raw candidate slots by themselves are insufficient and can be harmful. |
| **Remaining tradeoff** | The strongest `ranked-request-only` arm pays a modest publication-safety / precision cost, accepted capability confusions remain, and three-clause correctness is still low despite improving. |
| **Next unverified hypothesis** | The next bounded question is whether the stronger ranked-request representation can recover publication safety through representation-local confidence/rank evidence or validation-only arbitration **without** undoing its answerable and multi-clause gains. This is not yet a result. |
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

GDM51 introduced a separate clause-local ambiguity discriminator after capability training. GDM57 established that semantic ambiguity input contains useful signal. GDM58 then showed that simply exposing more candidates through one fixed permutation-invariant pooled summary does not improve the answerable/ambiguity tradeoff.

GDM59 keeps **top-five candidate breadth** and the same ambiguity-head input dimension / trainable capacity (`11 + 10*d`, or 7,691 inputs for DistilBERT) in every arm, but changes how semantic evidence is represented. Its matched-capacity arms are:

- `top5-pooled-control`: the canonical GDM58 top-five pooled representation;
- `ranked-candidate-only`: explicit `c1..c5` candidate slots plus a zero-filled second semantic block;
- `ranked-request-only`: per-rank request interactions `q*c1..q*c5` plus `abs(q-ci)`;
- `ranked-candidate-request`: explicit `c1..c5` plus `q*ci`;
- `ranked-candidate-delta`: explicit `c1..c5` plus `abs(q-ci)`.

Missing ranks are zero-padded. Capability training, candidate ranking, ambiguity curriculum, calibration policy, numerical execution, and deterministic GraphQL realization are fixed. GDM59 therefore isolates **semantic representation structure**, not candidate breadth or head size.

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
| GDM58 | Adding top-3/top-4/top-5/all candidates through the same pooled representation produces almost no operation benefit and does not improve schema answerable/ambiguity tradeoff over the top-2 pooled control. | Simple pooled breadth is negative/inconclusive; preserving candidate identity/rank remained untested. |
| **GDM59** | **Top-five rank-preserving request/candidate interaction features outperform the matched pooled control on answerable correctness and target recall while retaining strong ambiguity recovery. Candidate-only rank slots and some combined raw-candidate representations underperform.** | **Rank identity is useful primarily through request-relative interaction, not merely by exposing raw candidate vectors. Publication safety, residual semantic confusions, and multi-clause correctness remain unresolved.** |

# Latest canonical result: GDM59

GDM59 changes **only the semantic representation structure of the ambiguity head**. Every arm sees the same top-five real candidates, uses the same 7,691-input ambiguity-head capacity, and keeps capability training, explicit NONE, the exact GDM56 family-balanced ambiguity curriculum, frozen DistilBERT, the `1e-4` feature boundary, canonical GDM50 numerical path, GDM54 rescue calibration/arbitration, and deterministic GraphQL/Federation realization fixed.

## GDM59 data and denominators

`Permit` / `Chronicle` are calibration-only. `Warranty` / `Dossier` are the fresh GDM59 secondary holdout and are now **regression-only** for subsequent experiments. GDM46–GDM58 inspected holdouts also remain regression-only.

Each task has **84 unique secondary-holdout cases**: 36 answerable, 32 NO_MATCH, and 16 AMBIGUOUS. Rates are means over seeds `5901,5902`. Status counts pool those seeds, so `29/32` represents sixteen unique ambiguous cases evaluated by two independently initialized models—not thirty-two independent semantic worlds. The exact same-source rerun is reproducibility evidence and does not increase the semantic sample size.

The collector independently recomputes the four declared ambiguity families—role, lifecycle-time, representation/name-vs-ID, and object-vs-supplier—from per-example holdout predictions, and the exact-attempt audit requires those metrics/predictions to match across both attempts. The canonical top-level `BATCH_REPORT` does not pool family values across seeds, so this README does **not** invent aggregate family counts or claim that representation/name-vs-ID or object-vs-supplier is solved. Residual accepted errors directly show role and lifecycle mistakes remain.

## Operation generation

| Representation | Answerable accuracy | Risk accuracy | Incorrect publication | Target precision | Target recall | AMBIGUOUS | NO_MATCH |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Top-5 pooled control | 51.39% | 56.25% | **26.79%** | 94.39% | 53.91% | **29/32** | 25/64 |
| Ranked candidate only | 45.83% | 44.79% | 31.55% | 93.73% | 48.44% | 16/32 | **27/64** |
| Ranked candidate + request product | 48.61% | 50.00% | 29.17% | 94.21% | 51.56% | 21/32 | **27/64** |
| Ranked request only | **56.94%** | **58.33%** | 27.98% | **94.92%** | **58.59%** | **29/32** | **27/64** |
| Ranked candidate + request delta | **56.94%** | **58.33%** | 27.98% | **94.92%** | **58.59%** | **29/32** | **27/64** |

`ranked-request-only` and `ranked-candidate-delta` improve answerable accuracy by **5.56 percentage points** and target recall by **4.68 points** versus the matched pooled control, while retaining 29/32 AMBIGUOUS and improving NO_MATCH from 25/64 to 27/64. The cost is a **1.19-point** increase in incorrect publication. Candidate-only slots are negative evidence: they collapse AMBIGUOUS to 16/32 and reduce answerable accuracy.

Exact answerable correctness for the strongest ranked arms is **23/32 one-clause, 14/24 two-clause, and 4/16 three-clause**, versus **21/32, 12/24, 4/16** for the pooled control. The representation helps one- and two-clause operation requests, but does not improve three-clause exact correctness.

Seed-level evidence is directionally useful but still small. The operation pooled control is 15/36 answerable at seed 5901 and 22/36 at seed 5902; ranked-request-only raises the weaker seed to 19/36 while leaving seed 5902 at 22/36. The aggregate gain therefore comes from rescuing the weaker initialization rather than increasing the already-strong seed.

Accepted operation confusions still include `reviews.author.name` versus `reviews.moderator.name` in both fresh domains. The ambiguity representation cannot repair a wrong capability once the request is accepted.

## Schema generation

| Representation | Answerable accuracy | Risk accuracy | Incorrect publication | Target precision | Target recall | AMBIGUOUS | NO_MATCH |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Top-5 pooled control | 47.22% | 56.25% | 26.79% | **93.99%** | 50.78% | **29/32** | 25/64 |
| Ranked candidate only | 33.33% | 51.04% | 25.00% | 91.75% | 35.16% | 20/32 | **29/64** |
| Ranked candidate + request product | 31.94% | 56.25% | **22.02%** | 91.49% | 33.59% | 25/32 | **29/64** |
| Ranked candidate + request delta | 45.83% | 56.25% | 28.57% | 93.84% | 47.66% | 26/32 | 28/64 |
| Ranked request only | **56.94%** | **58.33%** | 29.17% | 93.12% | **62.50%** | **29/32** | 27/64 |

`ranked-request-only` is the strongest aggregate schema answerable/recall arm: answerable accuracy rises by **9.72 points** and target recall by **11.72 points** over the pooled control, with risk accuracy +2.08 points, NO_MATCH +2/64, and the same 29/32 AMBIGUOUS. The tradeoff is **+2.38 points incorrect publication** and **-0.87 points target precision**. Candidate-only and candidate+request-product representations are negative evidence for answerable correctness even when they improve some risk/publication dimensions.

Exact schema correctness for `ranked-request-only` is **21/32 one-clause, 14/24 two-clause, and 6/16 three-clause**, versus **19/32, 11/24, 4/16** for the pooled control. This is the first result in this local representation sequence that improves all three clause-count buckets against its within-batch pooled control, although 6/16 three-clause remains far from solved.

The schema gain is also more stable across seeds than the pooled control. Pooled answerable correctness is 20/36 at seed 5901 but only 14/36 at seed 5902; `ranked-request-only` is 20/36 and 21/36. Risk correctness shifts from 26/48 and 28/48 in the pooled arm to 30/48 and 26/48 in the ranked arm, so risk remains seed-sensitive even while answerable correctness stabilizes.

Accepted schema confusions still include author-versus-moderator mistakes and a `createdAt` request published as `title`. GDM59 therefore does not establish that role or lifecycle ambiguity is solved, and the batch aggregate is insufficient to claim resolution of the representation/name-vs-ID or object-vs-supplier families.

## Regression and interpretation limits

Previously inspected regression suites are **regression-only** and are not a causal cross-batch leaderboard. Mean regression request accuracy across GDM59 arms spans approximately **44.77%–52.62% for operation** and **42.15%–51.91% for schema**. `ranked-request-only` is about **52.16% operation** and **51.21% schema**, near the top of the GDM59 range but not evidence of broad OOD generalization.

GDM59 specifically tests explicit top-five rank slots and simple elementwise request/candidate interactions. It does not test deeper learned set/sequence attention, autoregressive GraphQL generation, arbitrary schema invention, or a learned request-global status model.

## Speed and memory

Attempt 1 mean-per-seed warm-generation measurements are observational:

| Task | p50 range across arms | p95 range across arms | mean RSS-after-evaluation range |
| --- | ---: | ---: | ---: |
| Operation | 69.87–70.70 ms | 137.83–139.93 ms | 1,061,968–1,066,022 KiB |
| Schema | 54.34–56.17 ms | 109.98–113.51 ms | 1,062,?–1,066,? KiB |

More exactly, the canonical GDM59 batch reports schema mean RSS from **1,063,072 to 1,065,994 KiB** across arms. Generation timing covers fresh request-clause encoding, learned-head scoring, deterministic GraphQL realization, and local validation with catalog/NONE vectors cached. Downloads, initial catalog embedding, backend fixture execution, and Rover composition are excluded from these generation quantiles; composition is recorded separately. RSS is process-level and affected by worker reuse, so it is **not isolated model memory**. Timing and memory are observational and need not bit-match across hosted-runner attempts; deterministic model state and predictions do.

## GDM59 reproducibility and provenance

GDM59’s canonical measured source is commit [`035f44836d850b24f4bebe5cd499a572b6052752`](https://github.com/burn2delete/graph-model/commit/035f44836d850b24f4bebe5cd499a572b6052752), source run [`35898256989`](https://github.com/burn2delete/graph-model/actions/runs/35898256989), attempts 1 and 2. Each attempt independently verified **20/20** declared results with no missing, failed, or unexpected configurations. Both preflights passed 58 current/inherited contract tests and the deterministic ranked-candidate-request hash smoke; the hash smoke is contract evidence, not pretrained-model performance.

- Attempt 1 `BATCH_REPORT` artifact **10769046397**, ZIP SHA-256 `b4a2237c074588a61031eb957ede60780b41bf5afc3b020d9c26768149ab5b20`.
- Attempt 2 `BATCH_REPORT` artifact **10770769769**, ZIP SHA-256 `c11e48f18b56d0e28f03720a9a7579957cbc97402642d591e1644789a656b312`.
- Passing exact-attempt audit run [`35910892558`](https://github.com/burn2delete/graph-model/actions/runs/35910892558), audit workflow commit [`a43312fcb1132caef06c6a1e5ef7ff66aa47b5dd`](https://github.com/burn2delete/graph-model/commit/a43312fcb1132caef06c6a1e5ef7ff66aa47b5dd).
- Passing audit artifact **10773722103**, ZIP SHA-256 `66dfea1e8bce33a06c77a80dbb287a29df2ddaba68d57aed289e51899f2b90a3`.

The passing report is `gdm59-reproducibility-audit-v1` with `complete=true`, `evidence_verified=true`, `reproducible=true`, `promotion_eligible=true`, `errors=[]`, exact **operation 10/10 + schema 10/10**, and all 20 comparisons `exact_match=true` with `differing_fields=[]` and no raw-feature differences. The audit independently reconstructs both batch reports and verifies exact attempt provenance, initial/selected full/capability/ambiguity hashes, optimizer/training/calibration records, matched top-five/representation receipts, deterministic regression/holdout/clause/family metrics, exact per-example predictions, and canonical fixed-probe/full-corpus hashes.

Artifact retention is finite. IDs and digests record provenance but do not substitute for retained source evidence if future re-verification is required.

## Prior canonical result: GDM58

GDM58 changed only how many ranked candidates entered a fixed permutation-invariant pooled semantic representation. It found no operation answerable/precision/recall/publication benefit from wider pooled breadth and no schema answerable/ambiguity improvement over the top-two pooled control. That negative/inconclusive result motivated GDM59’s rank-preserving representation test.

GDM58’s measured source is `9ba9caab957c1802feb17776195141f630fd96eb`, source run `35819165142` attempts 1/2, BATCH_REPORT artifacts **10732952674** (`490d4ddcf88bbb517186c15d4dc7f85c643b3c75d480b27409d59f4c7a5c2a1a`) and **10735225675** (`36adadbd454fe064b20242e9ce1d8bdcbabfa4537520192b666331cf5ce6aaff`), passing audit run `35890238431`, and audit artifact **10765506374** (`b3f2f2273f8eca3cf6e12ba04b93144dcc51f59d327a47d91f96dc2bcb0c7163`). The earlier GDM58 2-GiB archive-guard failure and repeated runner-shutdown audit attempts remain preserved as audit/runtime diagnostics, not source-model failures; the bounded-resource exact audit passed without weakening equality.

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
| **GDM59** | source `035f44836d850b24f4bebe5cd499a572b6052752`, run `35898256989` | **`35910892558`** |

## Evidence and promotion policy

All accepted compilation, tests, training, validation, benchmarks, and independent evidence checks run in **GitHub Actions**. Source/artifact inspection may analyze already-produced evidence but does not replace an Actions verifier.

A canonical promotion requires complete declared results, changed learned-head checkpoint evidence, optimizer histories, validation-only checkpoint/calibration selection, per-example predictions/generated GraphQL, real execution/composition checks, raw timing/runtime metadata, an exact same-source/same-seed rerun, and a dedicated independent exact-attempt audit. The root README must be synchronized **before** announcing promotion or launching the next numbered experiment.

Fresh holdouts never select optimizer behavior, checkpoints, thresholds, curriculum, or representation changes. Once inspected, they become regression-only. Repeated seeds and exact reruns are reproducibility evidence, not extra semantic worlds.

## Current limits and next bounded question

The measured system remains a small synthetic catalog-projection research environment. It has not established unrestricted schema invention, general mutation/subscription generation, arbitrary enterprise topology handling, transformer fine-tuning benefits, or production-scale latency/memory behavior.

GDM59 resolves the GDM58 uncertainty in a useful direction: **lower-ranked candidates can help when their rank-specific relationship to the request is preserved instead of pooled away**. The result is more specific than “rank matters”: raw candidate slots alone underperform, while per-rank request interactions are the strongest measured representation. The schema answerable gain is stable across seeds and extends to three-clause exact correctness, while the operation gain primarily rescues the weaker seed.

The next unverified hypothesis should stay bounded around the remaining tradeoff rather than reopen curriculum or backbone changes. A suitable GDM60 would keep the canonical `ranked-request-only` representation, capability objective, curriculum, encoder, numerical path and deterministic realization fixed, then test whether **representation-local confidence/rank summaries or validation-only arbitration** can recover incorrect-publication / NO_MATCH safety without giving back answerable and multi-clause gains. It should include the canonical GDM59 representation as a within-batch control, use fresh calibration/secondary-holdout domains, and avoid mixing in backbone tuning or new curriculum.

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
| [`experiments/followup59/`](experiments/followup59/) | Rank-preserving top-five ambiguity representation, latest canonical experiment |
| [`.github/workflows/`](.github/workflows/) | Actions-only measured batches and exact-attempt audits |

Use each experiment’s declared workflow/execution wrapper when reproducing it. Calling historical Python entrypoints directly can bypass the canonical numerical contract.
