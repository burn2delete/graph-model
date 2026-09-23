# Graph Model

**A GraphQL Decision Model (GDM): learn the decisions that satisfy a request, then construct and evaluate real GraphQL.**

This repository researches models that reason about GraphQL and generate **operations and schemas**. Query planning is out of scope.

The current measured system is deliberately narrower than a general-purpose GraphQL generator: it uses a **frozen semantic encoder plus learned capability and ambiguity heads**, then deterministically realizes selected capabilities as executable GraphQL operations or catalog-grounded Federation schemas. It does not autoregressively generate arbitrary GraphQL tokens, invent arbitrary ontologies, or fine-tune the transformer backbone.

## Current research state

_Last synchronized: 2026-09-23. Latest canonical experiment: GDM58._

| State | Meaning |
| --- | --- |
| **Canonical through GDM58** | GDM50–GDM58 each have repeated measured evidence and a passing exact-attempt reproducibility audit. Canonical means the evidence is usable and reproducible under its declared contract; it does **not** mean every tested arm is a winning architecture. |
| **Latest result: pooled top-k breadth is negative/inconclusive** | GDM58 widened the ambiguity head from the top two ranked candidates to top-3/top-4/top-5/all-candidate **pooled** semantic evidence while keeping head capacity and the rest of the system fixed. Additional pooled breadth did not improve operation answerable metrics and did not beat the within-batch top-2 pooled control on schema answerable/ambiguity tradeoff. This does not rule out top-k architectures that preserve candidate identity or rank. |
| **Next hypothesis: identity/rank-preserving candidate aggregation, unverified** | The next bounded question is whether a matched-capacity ambiguity representation that preserves individual candidate identity/rank—rather than collapsing candidates into summary statistics—can improve ambiguity discrimination without worsening NO_MATCH/publication safety. No result is claimed for that hypothesis yet. |
| **Invalid historical evidence** | Original GDM41–44 jobs wrote manifests and fabricated/simulated metrics rather than measured model results. Their promotions are withdrawn. Only measured repair run `35565425124` (196/196 verified) is the valid GDM41–44 baseline. |

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

The benchmark is synthetic and catalog-bounded. Fresh domain names and paraphrases are useful screening controls, but they are not evidence of enterprise out-of-distribution generalization or arbitrary graph-topology generation.

### Clause decomposition

Multi-capability requests are decomposed using public request syntax into clause-local decisions. This is deterministic benchmark parsing, not a learned natural-language decomposition model. The research line retains clause-local scoring because GDM45 showed whole-request capability scoring collapses particularly badly as requested capability count grows.

### Frozen semantic encoder and numerical boundary

The active line uses `distilbert/distilbert-base-uncased` at the exact revision recorded in every result. The transformer is frozen. Mean-pooled normalized representations cross an explicit **`1e-4` canonical feature boundary** before learned-head training. Request clauses are encoded fresh during timed generation; catalog and NONE vectors are cached.

The canonical CPU/numerical contract fixes one-thread/default CPU dispatch, disables oneDNN, uses PyTorch 2.8 single-tensor AdamW with `foreach=false` and `fused=false`, non-foreach gradient clipping, float32 optimizer moments/state, and evaluates the final decoupled-weight-decay plus bias-corrected parameter application in float64 before storing float32 parameters. This is a reproducibility architecture contract, not transformer fine-tuning.

### Learned capability head and explicit NONE

For a request-clause vector `q` and candidate vector `c`, the capability scorer consumes request and candidate representations, elementwise interaction, absolute difference, catalog-relative candidate information, and structural GraphQL features. A residual feature adapter and MLP score every real capability plus an explicit **NONE** outcome.

Unsupported requests can therefore train toward NONE rather than being forced onto a real field. These are learned feature-space heads over frozen representations. Separate operation/schema checkpoints do not imply separately fine-tuned transformer backbones.

### Learned ambiguity head

GDM51 introduced a separate clause-local ambiguity discriminator after capability training. Before GDM57, the ambiguity representation consisted of eleven scalar summary features derived from top-two candidate evidence: NONE-versus-best and best-versus-second margins, probabilities, encoder similarities, and structural signals such as shared parent, depth, and leaf-kind similarity.

GDM57 established that richer top-two semantic input contains useful ambiguity signal. GDM58 keeps the same **7,691-input** ambiguity-head capacity (`11 + 10*d` for DistilBERT) and evaluates a fixed-dimensional, mathematically permutation-invariant semantic pool while varying only how many ranked real candidates contribute to that pool. The five GDM58 arms use top 2, top 3, top 4, top 5, or all ranked real candidates.

The GDM58 candidate semantic block contains mean, capability-probability-weighted mean, elementwise max, min, and standard deviation. The request-conditioned block contains request vector `q`, `q * weighted_mean`, `abs(q - weighted_mean)`, mean absolute request/candidate difference, and maximum absolute request/candidate difference. The original eleven scalar/structural features remain present.

GDM58 therefore measures **candidate breadth inside this pooled representation**, not arbitrary set attention, sequence attention, or per-candidate learned aggregation. A negative pooled-breadth result must not be generalized to every possible top-k representation.

### Validation-only arbitration

The capability and ambiguity heads are optimizer-trained. Thresholds and request arbitration are not. Validation-only calibration chooses NONE and ambiguity thresholds; the current line keeps the GDM54 high-confidence ambiguity-rescue policy with a 5 percentage-point NO_MATCH-recall budget and an accepted-status recall guardrail.

Request aggregation is deterministic: a remaining NO_MATCH clause rejects the request; otherwise any AMBIGUOUS clause makes the request ambiguous; otherwise selected paths are emitted.

### Deterministic GraphQL realization

Accepted operation capabilities are converted to a closed selection tree, parsed and validated with `graphql-core`, and executed against changing fixtures designed to distinguish author/moderator, name/ID, and creation/update mistakes.

Accepted schema capabilities are projected into a minimal field inventory with deterministic object/key closure and subgraph ownership conventions. Rover composes the actual generated subgraphs, and a separate requirements evaluator checks whether requested capabilities are present. **Composition success is not request correctness.** Schema generation remains catalog projection plus deterministic SDL/Federation realization, not unconstrained schema invention.

## What the experiment line has taught us

| Experiment | Measured learning | Consequence / limit |
| --- | --- | --- |
| GDM45 | Clause-local decomposition improves multi-capability generation relative to whole-request scoring. | Keep clause-local decisions. Its schema risk/status path had a shortcut, so only accepted-case evidence is retained. |
| GDM46–49 | Strong rejection gates can improve risk handling while destroying answerable recall. | Risk correctness and answerable correctness must be measured separately. |
| GDM50 | Explicit NONE gives the model an open-set unsupported outcome, but does not solve ambiguity or multi-clause compounding. Numerical diagnostics also exposed cross-run optimizer drift. | Keep explicit NONE and the hardened numerical evidence contract. |
| GDM51 | A separate ambiguity discriminator improves the capability-vs-risk decomposition. | Ambiguity deserves a learned component, but its representation matters. |
| GDM52–54 | Calibration and arbitration expose precision/recall tradeoffs; ambiguity-first arbitration alone loses too many correct NO_MATCH decisions. | Do not expect threshold ordering to repair weak ambiguity evidence. |
| GDM55 | Relation-family expansion recovers additional ambiguity beyond volume-matched optimizer exposure, but gains are task/seed dependent. A bundled full curriculum can increase answerable recall while worsening ambiguity/publication. | Semantic diversity matters, but “more curriculum” is not a monotonic improvement. |
| GDM56 | Balancing relation-family exposure materially improves operation ambiguity stability and publication safety, but benefits do not transfer uniformly to schema. Counterfactual-negative arms collapse held-out ambiguity to zero. | Curriculum alone does not solve the missing relation distinctions. |
| GDM57 | Request-conditioned top-two semantics recover far more AMBIGUOUS cases than the matched-capacity scalar control, especially for schemas. Combined candidate+request representations improve schema risk balance. | The semantic representation contains useful risk signal, but top-two evidence still creates an AMBIGUOUS-versus-NO_MATCH/publication tradeoff and multi-clause compounding remains weak. |
| **GDM58** | **Adding more ranked candidates through the same fixed pooled representation produces almost no operation benefit and does not improve the schema answerable/ambiguity tradeoff over the top-2 pooled control.** | **Simple pooled breadth is negative/inconclusive. Additional candidates may still help if their identity/rank is preserved instead of collapsed into pooled statistics.** |

# Latest canonical result: GDM58

GDM58 changes **only candidate-evidence breadth inside a fixed pooled semantic representation of the ambiguity head**. Capability training, explicit NONE, the exact GDM56 family-balanced ambiguity curriculum, frozen DistilBERT, the `1e-4` feature boundary, canonical GDM50 numerical execution, GDM54 rescue calibration/arbitration, and deterministic GraphQL/Federation realization remain fixed.

The five matched-capacity arms are `top2-pooled-control`, `top3-pooled`, `top4-pooled`, `top5-pooled`, and `all-pooled`. All use the same pooling operator, the same 7,691-input ambiguity head, the same training counts, and the same seeds `5801,5802`. The only experimental factor is the number of ranked real candidates entering the semantic pool.

## GDM58 data and denominators

`Notebook` / `Dispatch` are calibration-only. `Entitlement` / `Workbook` are the fresh GDM58 secondary holdout and are now **regression-only** for subsequent experiments. GDM46–GDM57 inspected holdouts also remain regression-only.

Each task has **84 unique secondary-holdout cases**: 36 answerable, 32 NO_MATCH, and 16 AMBIGUOUS. Rates are means over seeds `5801,5802`. Status counts pool those two seeds, so `27/32` represents sixteen unique ambiguous cases evaluated by two independently initialized models—not thirty-two independent semantic worlds. The exact reproducibility rerun is reproducibility evidence and does not increase the semantic sample size.

The source collector independently recomputes the four declared ambiguity families—role, lifecycle-time, representation/name-vs-ID, and object-vs-supplier—from per-example holdout predictions for every measured result. The passing exact-attempt audit requires those family metrics and predictions to match across both attempts. The top-level canonical `BATCH_REPORT` does not pool family values across seeds, so this README does **not** invent aggregate family counts or claim that a previously weak family is solved.

## Operation generation

| Candidate breadth | Answerable accuracy | Risk accuracy | Incorrect publication | Target precision | Target recall | AMBIGUOUS | NO_MATCH |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Top 2 pooled control | **43.06%** | 55.21% | 33.93% | 84.36% | 54.69% | 25/32 | **28/64** |
| Top 3 pooled | **43.06%** | **56.25%** | 33.93% | 84.36% | 54.69% | **27/32** | 27/64 |
| Top 4 pooled | **43.06%** | 55.21% | 33.93% | 84.36% | 54.69% | 25/32 | **28/64** |
| Top 5 pooled | **43.06%** | 55.21% | 33.93% | 84.36% | 54.69% | **27/32** | 26/64 |
| All pooled | **43.06%** | **56.25%** | 33.93% | 84.36% | 54.69% | **27/32** | 27/64 |

Operation answerable accuracy, incorrect-publication rate, target precision, and target recall are **identical across all five breadth arms**. Wider candidate pools only move the status balance by one or two pooled seed-cases. Relative to the within-batch top-2 control, top-3/all add two AMBIGUOUS recoveries while losing one NO_MATCH recovery; top-5 also adds two AMBIGUOUS recoveries but loses two NO_MATCH recoveries. This is not a meaningful broad improvement under the declared contract.

Exact answerable correctness is likewise unchanged across every operation arm: **18/32 one-clause, 9/24 two-clause, and 4/16 three-clause** pooled across seeds. Wider pooled evidence therefore does not repair the multi-clause capability-selection problem.

Accepted semantic confusions are also almost unchanged across operation arms. Repeated errors include lifecycle-time (`createdAt` versus `updatedAt`), author-versus-moderator, name-versus-ID-like substitutions, and other wrong-but-valid capability selections. The ambiguity head cannot repair a capability error after the request is accepted.

## Schema generation

| Candidate breadth | Answerable accuracy | Risk accuracy | Incorrect publication | Target precision | Target recall | AMBIGUOUS | NO_MATCH |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Top 2 pooled control | **38.89%** | **56.25%** | 36.31% | 84.19% | **49.22%** | **23/32** | 31/64 |
| Top 3 pooled | **38.89%** | 50.00% | 38.69% | 84.19% | **49.22%** | 17/32 | 31/64 |
| Top 4 pooled | 36.11% | 55.21% | 37.50% | 83.68% | 46.88% | **23/32** | 30/64 |
| Top 5 pooled | 34.72% | 55.21% | **34.52%** | **84.74%** | 45.31% | 21/32 | **32/64** |
| All pooled | 34.72% | 55.21% | 35.12% | **84.74%** | 45.31% | **23/32** | 30/64 |

No broader pooled arm improves schema answerable accuracy or AMBIGUOUS recovery over the within-batch top-2 control. Top-3 preserves answerable accuracy but drops AMBIGUOUS from 23/32 to 17/32 and risk accuracy from 56.25% to 50.00%. Top-5 lowers incorrect publication from 36.31% to 34.52% and recovers 32/64 NO_MATCH cases, but it also lowers answerable accuracy to 34.72% and target recall to 45.31%. No schema arm dominates.

Schema exact answerable correctness also weakens or stays flat as breadth grows. Top-2/top-3 get **18/32 one-clause, 8/24 two-clause, and 2/16 three-clause**; top-4 gets 17/32, 7/24, 2/16; top-5/all get 16/32, 7/24, 2/16. The additional pooled candidates do not improve multi-clause composition of required capabilities.

Accepted semantic confusions remain family-relevant: object-versus-supplier/title errors, author-versus-moderator errors, and lifecycle-time mistakes persist. Because the batch-level artifact does not expose pooled family counts, the supported conclusion is that these errors remain observable—not that any specific family is numerically solved or worsened across all seeds.

## Regression and interpretation limits

Previously inspected regression suites are **not fresh evidence** and are not a causal cross-batch leaderboard. Across GDM58 arms, mean regression request accuracy spans approximately **51.98%–52.53% for operation** and **51.26%–52.03% for schema**. These numbers only show that the experiment did not catastrophically break the retained regression suite.

GDM58 is specifically a test of candidate breadth inside one fixed pooled representation. The result does not show that candidates below rank two are useless; it shows that **collapsing them into these summary statistics does not extract additional useful signal under this training/calibration contract**.

## Speed and memory

Attempt 1 mean-per-seed warm-generation measurements are observational:

| Task | p50 range across arms | p95 range across arms | mean RSS-after-evaluation range |
| --- | ---: | ---: | ---: |
| Operation | 67.08–67.89 ms | 131.38–132.95 ms | 1,020,966–1,023,414 KiB |
| Schema | 52.11–53.35 ms | 103.63–105.84 ms | 1,019,282–1,023,652 KiB |

The generation scope is fresh request-clause encoding, learned-head scoring, deterministic GraphQL realization, and local validation with catalog/NONE vectors cached. Downloads, initial catalog embedding, backend fixture execution, and Rover composition are excluded from these generation quantiles; composition is recorded separately. RSS is process-level and affected by worker reuse, so it is **not isolated model memory**. Timing/memory are observational and do not need to bit-match across hosted-runner attempts; deterministic model state and predictions do.

## GDM58 reproducibility and provenance

GDM58’s canonical measured source is commit [`9ba9caab957c1802feb17776195141f630fd96eb`](https://github.com/burn2delete/graph-model/commit/9ba9caab957c1802feb17776195141f630fd96eb), source run [`35819165142`](https://github.com/burn2delete/graph-model/actions/runs/35819165142), attempts 1 and 2. Each attempt independently verified **20/20** declared results with no missing, failed, or unexpected configurations.

- Attempt 1 `BATCH_REPORT` artifact **10732952674**, ZIP SHA-256 `490d4ddcf88bbb517186c15d4dc7f85c643b3c75d480b27409d59f4c7a5c2a1a`.
- Attempt 2 `BATCH_REPORT` artifact **10735225675**, ZIP SHA-256 `36adadbd454fe064b20242e9ce1d8bdcbabfa4537520192b666331cf5ce6aaff`.
- Passing exact-attempt audit run [`35890238431`](https://github.com/burn2delete/graph-model/actions/runs/35890238431), audit workflow commit [`e0dc6af93ed93e64fa715dce12d02ecf5de3903d`](https://github.com/burn2delete/graph-model/commit/e0dc6af93ed93e64fa715dce12d02ecf5de3903d).
- Passing audit artifact **10765506374**, ZIP SHA-256 `b3f2f2273f8eca3cf6e12ba04b93144dcc51f59d327a47d91f96dc2bcb0c7163`.

The passing report is `gdm58-reproducibility-audit-v1` with `complete=true`, `evidence_verified=true`, `reproducible=true`, `promotion_eligible=true`, `errors=[]`, and exact **operation 10/10 + schema 10/10** matching. All 20 comparisons have `exact_match=true`, `differing_fields=[]`, and there are no raw-feature differences. The audit independently reconstructs both batch reports and verifies exact source-attempt provenance, initial/selected full-state/capability/ambiguity hashes, selected epochs, optimizer counts, complete training/calibration records, regression/holdout/clause/family metrics, exact per-example predictions, matched top-k/pooling receipts, and canonical fixed-probe/full-corpus hashes.

The audit itself has preserved diagnostics. Initial audit run `35827844562` failed before comparison because its archive guard rejected the measured worker artifacts above 2 GiB uncompressed; failed audit artifact **10735783228**, SHA-256 `7389aee24116f641e9a3d921f40fec785d94141ee1097ca1c9e75f806cbb826a`. Audit-only commit `82a28ee89694bc69adb3917fb2ebc0e9e29d9bab` raised that guard without changing evidence gates. Subsequent attempts of run `35833049861` repeatedly terminated with runner shutdown while processing the large evidence set and produced no promotion report. Audit-only commit `e0dc6af93ed93e64fa715dce12d02ecf5de3903d` bounded resource use and added diagnostics while preserving full exact equality; the resulting passing run is `35890238431`. Those audit failures are audit-implementation/runtime diagnostics, not source-model failures.

Artifact retention is finite. IDs and digests record provenance but do not substitute for retained source evidence if future re-verification is required.

## Prior canonical result: GDM57

GDM57 changed only the ambiguity-head semantic representation while keeping capability training, curriculum, numerical execution, calibration/arbitration and deterministic realization fixed. Request-conditioned top-two semantics produced much stronger AMBIGUOUS recovery, while combined candidate+request semantics gave a better schema risk balance. The tradeoff was increased incorrect publication/NO_MATCH loss and weak multi-clause exact correctness. GDM57’s top-two limitation motivated GDM58.

GDM57’s measured source is `3f8b9a968285110ca6c744590724ddd0cb2bd5aa`, source run `35803609103` attempts 1/2, BATCH_REPORT artifacts **10727032548** (`d46fbc4ad35dfa8e27b6702a02b930727e44734e277a950f279e33f3c8a38d82`) and **10728527353** (`3c180f3bba9e17f52526fd4ed734cd9eee1663a22ae3137ea6d246874bb0054d`), passing exact audit run `35811475866`, and audit artifact **10730038016** (`a21dfbe5eb859228305b1c8aa3eb207ac058a9483d68c1d2db4bcbc3a279e998`).

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
| **GDM58** | source `9ba9caab957c1802feb17776195141f630fd96eb`, run `35819165142` | **`35890238431`** |

## Evidence and promotion policy

All accepted compilation, tests, training, validation, benchmarks, and independent evidence checks run in **GitHub Actions**. Source or artifact inspection may analyze already-produced evidence but does not replace an Actions verifier.

A canonical promotion requires complete declared results, changed learned-head checkpoint evidence, optimizer histories, validation-only checkpoint/calibration selection, per-example predictions and generated GraphQL, real execution/composition checks, raw timing/runtime metadata, an exact same-source/same-seed rerun, and a dedicated independent exact-attempt audit. The root README must be synchronized **before** announcing promotion or launching the next numbered experiment.

Fresh holdouts never select optimizer behavior, checkpoints, thresholds, curriculum, or representation changes. Once inspected, they become regression-only. Repeated seeds and exact reruns are reproducibility evidence, not extra semantic worlds.

## Current limits and next bounded question

The measured system is still a small synthetic catalog-projection research environment. It has not established unrestricted schema invention, general mutation/subscription generation, arbitrary enterprise topology handling, transformer fine-tuning benefits, or production-scale latency/memory behavior.

GDM58 resolves a narrower uncertainty: **simply widening a permutation-invariant pooled candidate summary is not enough**. For operation generation, the answerable/precision/recall/publication metrics are unchanged from top 2 through all candidates. For schema generation, broader pools do not improve the top-2 control’s answerable/ambiguity tradeoff and can make it worse.

The next unverified hypothesis is therefore a bounded **candidate identity/rank-preserving ambiguity representation**. Keep the capability objective, GDM56 family-balanced curriculum, frozen encoder, `1e-4` feature boundary, numerical path, validation-only rescue policy, and deterministic GraphQL realization fixed. Preserve a pooled top-k control, but compare it with a matched-capacity representation that can retain per-candidate identity/rank—such as a small learned set/attention aggregator or explicit per-candidate blocks with masking. The experiment must isolate representation structure rather than simultaneously changing curriculum, arbitration, or backbone training. This is a hypothesis, not yet a result.

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
| [`experiments/followup58/`](experiments/followup58/) | Pooled top-k ambiguity evidence, latest canonical experiment |
| [`.github/workflows/`](.github/workflows/) | Actions-only measured batches and exact-attempt audits |

Use each experiment’s declared workflow/execution wrapper when reproducing it. Calling historical Python entrypoints directly can bypass the canonical numerical contract.
