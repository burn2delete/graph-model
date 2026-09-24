# Graph Model

**A GraphQL Decision Model (GDM): given an intent and a provided GraphQL schema, learn the decisions required to construct a correct executable GraphQL operation.**

## Forward research scope

As of **2026-09-24, forward research is operation-generation only**. The target is:

> **natural-language intent/request + provided GraphQL schema/capability catalog → correct executable GraphQL operation**

The system must also know when not to publish an operation: unsupported requests should resolve to **NO_MATCH**, and semantically underdetermined requests should resolve to **AMBIGUOUS**.

**Schema-generation research is suspended. Query planning is out of scope.** Historical schema experiments through GDM61 are preserved as reproducible research history, but they are frozen: they do not drive new architecture choices, new measured arms, promotion decisions, or forward benchmarks.

The measured system is deliberately narrower than unrestricted token-level GraphQL synthesis. A **frozen semantic encoder plus learned capability and ambiguity heads** maps request clauses onto schema-grounded capabilities. Deterministic code then realizes accepted capabilities as real GraphQL, which is parsed, validated, and executed. The transformer backbone is not fine-tuned.

Read [`experiments/MEASUREMENT_POLICY.md`](experiments/MEASUREMENT_POLICY.md) and [`experiments/measured/REPAIR.md`](experiments/measured/REPAIR.md) before interpreting results or changing experiments.

## Current research state

_Last synchronized: 2026-09-24. Latest canonical experiment: GDM61. Forward scope: operation generation only._

| State | Meaning |
| --- | --- |
| **Canonical through GDM61** | GDM50–GDM61 each have repeated measured evidence plus a passing exact-attempt reproducibility audit. Canonical means reproducible usable evidence under the declared contract, not automatic architecture victory. |
| **Forward task** | Generate an executable GraphQL operation from an intent and provided schema/catalog, while correctly handling answerable, NO_MATCH, and AMBIGUOUS requests. |
| **Schema generation** | **Suspended/frozen.** GDM41–61 schema evidence remains historical provenance only. No new schema arms are launched, scored, optimized, audited for promotion, or used to pick forward architecture. |
| **Latest supported operation result** | GDM61 is negative/inconclusive for fixed product/delta rescaling. Product-dominant raises answerable accuracy/recall but loses risk/publication/AMBIGUOUS safety; the GDM60 product-only signal does not replicate on the fresh GDM61 holdout. Three-clause operation exact correctness remains 0/16 for every GDM61 arm. |
| **Active, unpromoted GDM62 transition** | GDM62 was launched before the scope change as a mixed operation/schema batch. Its two source attempts finished successfully. Only its **10 operation configurations** are eligible for forward use. A dedicated operation-only exact-attempt audit is running; its schema artifacts are frozen and excluded from promotion. |
| **Invalid historical evidence** | Original GDM41–44 jobs wrote manifests and fabricated/simulated metrics. Their promotions are withdrawn. Only measured repair run `35565425124` (196/196 verified) is the valid GDM41–44 baseline. |

## Operation-generation architecture

```text
Natural-language intent/request + provided GraphQL schema/catalog
                              |
                deterministic clause decomposition
                              |
                    frozen DistilBERT encoder
             normalized features -> explicit 1e-4 grid
                   /                         \
       fresh request-clause vectors      cached catalog/NONE vectors
                   \                         /
                  clause-candidate pair features
                              |
              learned residual feature adapter + scorer
                 listwise scores over paths + NONE
                              |
                 ranked real candidates + NONE evidence
                              |
                  learned clause-local ambiguity head
          scalar/structural evidence + semantic candidate evidence
                              |
             validation-calibrated thresholds/arbitration
                              |
                deterministic request aggregation
                   /                          \
          NO_MATCH / AMBIGUOUS          accepted capability paths
                                                |
                                    deterministic selection-tree realization
                                                |
                                     executable GraphQL operation
                                                |
                                  parse -> validate -> execute fixtures
                                                |
                                      per-example evidence + metrics
```

### Inputs and bounded capabilities

Each forward example contains a natural-language request and a typed catalog derived from the **provided GraphQL schema**. A capability is a full GraphQL path with coordinates, type signatures, and a public semantic description. Paths such as `reviews.author.name`, `reviews.author.id`, and `reviews.moderator.name` are distinct response contracts even when they share terminal names.

The current benchmark is synthetic and catalog-bounded. Fresh domain names and paraphrases are screening controls; they are not evidence of enterprise out-of-distribution generalization.

### Clause-local decision making

Multi-capability requests are decomposed into clause-local decisions using deterministic benchmark syntax. This is not a learned natural-language decomposition model. Clause-local scoring remains because GDM45 showed that whole-request capability scoring degrades as requested capability count grows.

### Frozen semantic encoder and numerical boundary

The active line uses `distilbert/distilbert-base-uncased` at the exact revision recorded by each result. The transformer is frozen. Mean-pooled normalized representations cross an explicit **`1e-4` canonical feature boundary** before learned-head training. Request clauses are encoded fresh during timed generation; catalog and NONE vectors are cached.

The numerical contract fixes one-thread/default CPU dispatch, disables oneDNN, uses PyTorch 2.8 single-tensor AdamW with `foreach=false` and `fused=false`, non-foreach gradient clipping, float32 optimizer moments/state, and evaluates final decoupled-weight-decay plus bias-corrected parameter application in float64 before storing float32 parameters. This is a reproducibility contract, not transformer fine-tuning.

### Learned capability head and explicit NONE

For request-clause vector `q` and candidate vector `c`, the capability scorer consumes request/candidate semantic and structural features. A residual feature adapter and scorer rank every real capability plus explicit **NONE**. Unsupported requests can therefore train toward NONE instead of being forced onto a real field.

Capability and ambiguity are learned feature-space heads over frozen representations. There is no request-global trained status head in the current canonical architecture.

### Learned ambiguity head

GDM51 introduced a separate clause-local ambiguity discriminator. Subsequent experiments established that representation structure matters:

- GDM57: request-conditioned semantic evidence materially improves ambiguity recovery.
- GDM58: simply widening pooled top-k evidence is negative/inconclusive.
- GDM59: preserving top-five rank/identity helps through request-relative semantic interactions.
- GDM60: capability-confidence attenuation is negative; product-only looked strongest for operation on that batch.
- GDM61: fixed product/delta scaling does not yield a robust answerable/safety frontier and the GDM60 product-only operation endpoint does not replicate on fresh `Voucher` / `Anthology`.

The unresolved operation errors are relational: plausible candidates are confused with other plausible candidates, particularly author versus moderator and adjacent representation/name-vs-ID concepts.

### Validation-only arbitration

Capability and ambiguity heads are optimizer-trained. Status thresholds and request arbitration are validation-calibrated rather than trained status models. The active canonical line retains GDM54's high-confidence ambiguity-rescue policy with a five-percentage-point NO_MATCH-recall budget and accepted-status recall guardrail.

A remaining NO_MATCH clause rejects the request; otherwise an AMBIGUOUS clause makes the request ambiguous; otherwise selected paths are realized as GraphQL.

### Deterministic GraphQL realization

Accepted capabilities are converted to a closed selection tree. The generated operation is parsed and validated with `graphql-core`, then executed against changing fixtures that distinguish author/moderator, name/ID, creation/update, and other semantically close paths. Wrong-but-valid GraphQL fails response comparison.

Real parsing, validation, and execution are mandatory evidence. A green workflow, manifest, log, estimate, or simulated score is not evidence.

## Forward measurement contract

Normal post-suspension numbered batches are expected to contain **10 measured trained configurations: one operation task × five arms × two seeds**, unless a bounded experiment prespecifies another count.

Promotion tracks at minimum:

- answerable exact-operation accuracy;
- risk accuracy;
- incorrect-publication rate;
- target/field precision and recall;
- NO_MATCH correctness;
- AMBIGUOUS correctness;
- exact one-, two-, and three-clause correctness;
- semantic confusion classes and ambiguity families;
- regression versus fresh secondary holdout;
- measured warm-generation p50/p95;
- process-level RSS, with memory scope stated explicitly.

Fresh holdouts never enter optimizer, checkpoint, representation, curriculum, or calibration selection. After inspection they become regression-only. Seeds and exact reruns are reproducibility evidence, not additional semantic worlds.

Every accepted compile, test, training run, validation, benchmark, and independent evidence check runs in **GitHub Actions**. No local-container result may substitute for Actions evidence.

## What the experiment line has taught us

| Experiment | Measured learning | Forward consequence / limit |
| --- | --- | --- |
| GDM45 | Clause-local decisions improve multi-capability behavior over whole-request scoring. | Keep clause-local operation decisions. Historical schema shortcut limitation remains recorded. |
| GDM46–49 | Strong rejection gates can improve risk handling while destroying answerable recall. | Always report risk and answerable correctness separately. |
| GDM50 | Explicit NONE provides an open-set unsupported outcome; hardened numerical work exposed reproducibility drift. | Keep explicit NONE and canonical numerical execution. |
| GDM51 | A separate ambiguity discriminator improves capability-vs-risk decomposition. | Ambiguity deserves a learned component. |
| GDM52–54 | Calibration/arbitration move precision/recall tradeoffs but cannot repair weak representation evidence. | Keep calibration separate from representation capability. |
| GDM55–56 | Ambiguity curriculum diversity and family balance matter, but more curriculum is not monotonic. | Curriculum alone is not the missing representation. |
| GDM57 | Request-conditioned top-two semantics improve AMBIGUOUS recovery. | Useful semantic ambiguity signal exists. |
| GDM58 | Wider top-k through simple pooled representation produces little operation benefit. | Pooled candidate breadth is negative/inconclusive. |
| GDM59 | Rank-preserving top-five request interactions improve useful signal versus pooled controls. | Preserve candidate rank/identity. |
| GDM60 | Simple confidence weighting loses answerable recall; product-only was strongest operation factorization on that batch. | Confidence attenuation is negative; endpoint finding needs replication. |
| **GDM61** | **Fixed product/delta scaling does not improve the within-batch answerable/safety frontier. The GDM60 operation product-only endpoint does not replicate on fresh Voucher/Anthology.** | **Treat scalar rescaling as negative/inconclusive. Three-clause generation and semantic candidate confusions remain major operation bottlenecks.** |

Historical schema findings associated with these batches remain available in prior commits and artifacts but are frozen and are not forward research targets.

# Latest canonical operation evidence: GDM61

GDM61 keeps top-five candidate breadth, rank preservation, explicit NONE, learned capability/ambiguity heads, exact GDM56 family-balanced ambiguity curriculum, frozen DistilBERT, the `1e-4` feature boundary, canonical numerical execution, GDM54 validation-only rescue, and deterministic GraphQL realization fixed. It changes only deterministic scaling of two request-relative semantic blocks.

`Badge` / `Journal` are calibration-only. `Voucher` / `Anthology` are the fresh GDM61 secondary holdout and are now **regression-only**.

The operation holdout has 84 unique cases per seed: **36 answerable + 48 risk**, where risk is **32 NO_MATCH + 16 AMBIGUOUS**. Rates below are means over seeds `6101,6102`; status counts pool the two seeds, so AMBIGUOUS has denominator 32 and NO_MATCH denominator 64. Clause-count denominators pool seeds: **32 one-clause, 24 two-clause, 16 three-clause** evaluations.

| Representation | Answerable accuracy | Risk accuracy | Incorrect publication | Target precision | Target recall | AMBIGUOUS | NO_MATCH |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Product only | 16.67% | 68.75% | 9.52% | **94.12%** | 14.84% | 12/32 | **54/64** |
| Product dominant | **26.39%** | 63.54% | 13.69% | 88.85% | **25.00%** | 12/32 | 49/64 |
| Balanced control | 20.83% | 67.71% | 9.52% | 84.12% | 17.97% | **19/32** | 46/64 |
| Delta dominant | 20.83% | 68.75% | **8.93%** | 84.12% | 17.97% | **19/32** | 47/64 |
| Delta only | 20.83% | **70.83%** | 9.52% | 84.12% | 17.97% | **19/32** | 49/64 |

Product-dominant has the highest answerable accuracy and recall but is not a better frontier: versus balanced control it loses 4.17 points of risk accuracy, worsens incorrect publication by 4.17 points, and drops AMBIGUOUS recovery from 19/32 to 12/32.

Exact one-/two-/three-clause correctness is **13/32, 6/24, 0/16** for product-dominant; **11/32, 4/24, 0/16** for balanced, delta-dominant, and delta-only; and **9/32, 3/24, 0/16** for product-only. No GDM61 arm solves three-clause operation generation.

Seed sensitivity remains material: product-only answerable correctness is 8/36 on seed 6101 versus 4/36 on seed 6102; product-dominant is 8/36 versus 11/36. Role confusions remain visible, including moderator requests accepted as author-name paths.

GDM61 mean operation regression request accuracy spans approximately **43.29%–48.24%** across arms.

Attempt-1 warm-generation measurements are observational: operation p50 spans approximately **67.78–72.31 ms**, p95 **136.02–141.79 ms**, and mean RSS-after-evaluation **1,142,246–1,146,080 KiB** across arms. Generation timing includes fresh request-clause encoding, learned-head scoring, deterministic GraphQL realization, and local validation with catalog/NONE vectors cached; downloads, initial catalog embedding, and backend fixture execution are excluded. RSS is process-level and affected by worker reuse, not isolated model memory.

## GDM61 reproducibility and provenance

The original GDM61 source `913a683d3d294b2475699a7f5e9b31c565779d1c`, run `35935198866`, failed before measured workers because one endpoint contract test incorrectly required coordinate equality for two semantically equivalent but differently slotted delta representations. It produced no model evidence. Failed preflight artifact **10782563755**, SHA-256 `301fb291d9d8d86a77eebd36531909b2dd588a13c873741bc12b93a14cae1ac1`.

Test-only repair `f22fc36caf7327ed3d0d1f7dd2898ea31d837f3a` changed only that contract test. Canonical GDM61 measured source is repaired commit `f22fc36caf7327ed3d0d1f7dd2898ea31d837f3a`, source run `35939206923`, attempts 1 and 2.

- Attempt 1 mixed historical `BATCH_REPORT`: artifact **10784817334**, SHA-256 `d5ee636c17233358b353bac0189a8f05151a403adcd5afc840cf222a7eba2b4e`.
- Attempt 2 mixed historical `BATCH_REPORT`: artifact **10786971655**, SHA-256 `b96dd49e032a20b781b53495329b780c4509c417e5f182d694eae0f81e8ae965`.
- Passing exact-attempt audit run **35948010054**, audit workflow commit `d9d785da08c5866c52db8ae74f9ad597a1a4ba47`.
- Passing audit artifact **10788215936**, SHA-256 `5b81ee234a0c4dfe889a33bf33d85e4b0640dc72a8f96d132063356861ba1d6f`.

The GDM61 audit report has `complete=true`, `evidence_verified=true`, `reproducible=true`, `promotion_eligible=true`, `errors=[]`, exact operation 10/10 plus historical schema 10/10, and all 20 comparisons exact. Post-suspension experiments require operation exactness only; historical schema matching is not a continuing requirement.

## Active transition: GDM62 operation evidence

GDM62 source commit `a3cfa70a570f931e2c78553ca9d72aeb5ed65e79` was launched immediately before schema research was suspended. Its source workflow therefore contained both operation and schema workers. Source run **35953082798** completed two successful exact source attempts. The schema artifacts are retained only as historical diagnostics and are excluded from forward promotion.

The eligible evidence is the five candidate-relational **operation** arms over seeds `6201,6202` (10 configs total). The source collector can independently verify those operation configurations without schema. A dedicated operation-only exact-attempt audit is being used to establish whether the two attempts reproduce exactly and whether this evidence can be promoted under the new scope.

Until that audit passes, **GDM62 is not canonical** and its candidate-relational hypothesis remains unverified for promotion. Do not launch GDM63 merely to bypass this transition.

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
| GDM60 | source `c04784ec33983829f6fe55506bcb74e0f40442b2`, run `35918183757` | `35929537000` |
| **GDM61** | source `f22fc36caf7327ed3d0d1f7dd2898ea31d837f3a`, run `35939206923` | **`35948010054`** |

## Evidence and promotion policy

A post-suspension canonical promotion requires complete declared **operation** results, changed learned-head checkpoint evidence, optimizer histories, validation-only checkpoint/calibration selection, per-example predictions/generated GraphQL, real parse/validation/execution checks, raw timing/runtime metadata, an exact same-source/same-seed rerun, and a dedicated independent operation-only exact-attempt audit.

The root README must be synchronized **before** announcing promotion or launching the next numbered experiment. Promotion commits must record the exact measured source, source attempts, independently rebuilt operation BATCH_REPORT artifact IDs/digests, passing audit artifact ID/digest, denominators, regression/holdout limits, p50/p95, memory scope, supported and negative learnings, and the next unverified operation hypothesis.

Historical schema metrics and artifacts are retained for provenance but are frozen. They are not part of new promotion gates.

## Current limits and next decision gate

The measured system remains a small synthetic, schema-grounded research environment. It has not established unrestricted operation token generation, general mutation/subscription generation, arbitrary enterprise topology handling, learned clause decomposition, transformer fine-tuning benefits, or production-scale latency/memory behavior.

The immediate gate is **GDM62 operation-only reproducibility**. Its candidate-relational representation asks whether explicit relations among plausible ranked candidates add useful ambiguity information beyond request-relative semantics. No architectural conclusion should be drawn until the operation-only exact-attempt audit passes.

If that audit passes, GDM62 must be analyzed and documented in this README before any GDM63 design or launch. If it fails, preserve the attempts and localize the first operation-evidence divergence rather than weakening equality or reverting to schema work.

## Repository map

| Location | Purpose |
| --- | --- |
| [`experiments/MEASUREMENT_POLICY.md`](experiments/MEASUREMENT_POLICY.md) | Operation-only scope, evidence rules, and mandatory README-promotion contract |
| [`experiments/measured/`](experiments/measured/) | Repaired baseline, frozen encoders, shared learned heads, GraphQL evaluation contracts |
| [`experiments/followup50/`](experiments/followup50/) | Explicit NONE, cache boundary, canonical numerical execution |
| [`experiments/followup51/`](experiments/followup51/) | Capability/ambiguity staged training |
| [`experiments/followup52/`](experiments/followup52/), [`followup53`](experiments/followup53/), [`followup54`](experiments/followup54/) | Historical calibration/arbitration experiments |
| [`experiments/followup55/`](experiments/followup55/), [`followup56`](experiments/followup56/) | Historical ambiguity curriculum/family experiments |
| [`experiments/followup57/`](experiments/followup57/) | Top-two semantic ambiguity representation |
| [`experiments/followup58/`](experiments/followup58/) | Pooled top-k ambiguity evidence |
| [`experiments/followup59/`](experiments/followup59/) | Rank-preserving top-five ambiguity representation |
| [`experiments/followup60/`](experiments/followup60/) | Request-relative factorization/confidence scaling |
| [`experiments/followup61/`](experiments/followup61/) | Fixed product/delta balance; latest canonical evidence |
| [`experiments/followup62/`](experiments/followup62/) | Candidate-relational experiment launched pre-suspension; operation evidence only is eligible forward |
| [`.github/workflows/`](.github/workflows/) | Actions-only measured batches and exact-attempt audits |

Use each experiment's declared workflow/execution wrapper when reproducing it. Calling historical Python entrypoints directly can bypass the canonical numerical contract.
