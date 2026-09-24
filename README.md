# Graph Model

**A GraphQL Decision Model (GDM): given an intent and a provided GraphQL schema, learn the decisions required to construct a correct executable GraphQL operation.**

## Forward research scope

As of **2026-09-24, forward research is operation-generation only**:

> **natural-language intent/request + provided GraphQL schema/capability catalog → correct executable GraphQL operation**

The system must also know when not to publish an operation: unsupported requests should resolve to **NO_MATCH**, and semantically underdetermined requests should resolve to **AMBIGUOUS**.

**Schema-generation research is suspended/frozen. Query planning is out of scope.** Historical schema evidence through GDM61 is retained as provenance/regression history only; it does not drive new architecture choices, measured arms, promotion decisions, or forward benchmarks.

The measured system is deliberately narrower than unrestricted token-level GraphQL synthesis. A **frozen semantic encoder plus learned capability and ambiguity heads** maps request clauses onto schema-grounded capabilities. Deterministic code realizes accepted capabilities as real GraphQL, which is parsed, validated, and executed. The transformer backbone is not fine-tuned.

Read [`experiments/MEASUREMENT_POLICY.md`](experiments/MEASUREMENT_POLICY.md) and [`experiments/measured/REPAIR.md`](experiments/measured/REPAIR.md) before interpreting results or changing experiments.

## Current research state

_Last synchronized: 2026-09-24. Latest canonical experiment: GDM64. Forward scope: operation generation only._

| State | Meaning |
| --- | --- |
| **Canonical through GDM64** | GDM50–GDM61 retain historical measured evidence/audits; GDM62–GDM64 are post-suspension operation-only promotions. Canonical means reproducible usable evidence under the declared contract, not automatic architecture victory. |
| **Forward task** | Generate an executable GraphQL operation from an intent and provided schema/catalog, while correctly handling answerable, NO_MATCH, and AMBIGUOUS requests. |
| **Schema generation** | **Suspended/frozen.** GDM41–61 schema evidence remains historical provenance only. No new schema arms are launched, scored, optimized, or promoted. |
| **Latest supported operation result** | **GDM64 is negative/inconclusive for deterministic schema-coordinate parent/leaf factorization.** Parent-only, leaf-only, and split parent/leaf views do not improve the broad operation frontier over the exact whole-request control; split-product materially degrades answerable/recall/role behavior. The retained architecture remains ranked request-relative `q*ci + abs(q-ci)` over whole candidate embeddings. |
| **Next unverified hypothesis** | The persistent errors may require an **explicit structured schema-relation channel**, not another semantic embedding recombination. A bounded next experiment should test topology/type relations derived only from the provided schema/catalog—such as same-parent/sibling relation, same-leaf relation, path-prefix/LCP structure, and field/type-signature relation—alongside the canonical whole-candidate semantic slots under matched capacity. This must not use hidden answer labels or fresh-holdout information. |
| **Invalid historical evidence** | Original GDM41–44 jobs wrote manifests and fabricated/simulated metrics. Their promotions are withdrawn. Only measured repair run `35565425124` (196/196 verified) is the valid GDM41–44 baseline. |

## Retained operation-generation architecture

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
       scalar/structural + rank-preserving request-relative semantics
                      q*ci + abs(q-ci), TOP5
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

**GDM64 does not change the retained architecture.** It replaced the canonical whole-candidate ambiguity slots in experimental arms with deterministic parent-path and/or leaf-field semantic views derived from the provided GraphQL candidate path while holding TOP5 rank identity, dimensionality, downstream ambiguity MLP capacity, capability ranking, curriculum, numerical execution, calibration, and GraphQL realization fixed. Those factored views did not improve the broad frontier or eliminate the dominant role/name-vs-ID confusions, so they remain experimental negatives rather than retained architecture changes.

### Inputs and bounded capabilities

Each forward example contains a natural-language request and a typed catalog derived from the **provided GraphQL schema**. A capability is a full GraphQL path with coordinates, type signatures, and a public semantic description. Paths such as `reviews.author.name`, `reviews.author.id`, and `reviews.moderator.name` are distinct response contracts even when semantically close.

The current benchmark is synthetic and catalog-bounded. Fresh domain names and paraphrases are screening controls, not evidence of enterprise out-of-distribution generalization.

### Clause-local decision making

Multi-capability requests are decomposed into clause-local decisions using deterministic benchmark syntax. This is not a learned natural-language decomposition model. Clause-local scoring remains because GDM45 showed that whole-request capability scoring degrades as requested capability count grows.

### Frozen semantic encoder and numerical boundary

The active line uses `distilbert/distilbert-base-uncased` at the exact revision recorded by each result. The transformer is frozen. Mean-pooled normalized representations cross an explicit **`1e-4` canonical feature boundary** before learned-head training. Request clauses are encoded fresh during timed generation; catalog and NONE vectors are cached.

The numerical contract fixes one-thread/default CPU dispatch, disables oneDNN, uses PyTorch 2.8 single-tensor AdamW with `foreach=false` and `fused=false`, non-foreach gradient clipping, float32 optimizer moments/state, and evaluates final decoupled-weight-decay plus bias-corrected parameter application in float64 before storing float32 parameters.

### Learned capability, NONE, and ambiguity

The capability scorer ranks every real capability plus explicit **NONE**. Unsupported requests can therefore train toward NONE instead of being forced onto a real field. Capability and ambiguity are learned feature-space heads over frozen representations; there is no request-global trained status head in the retained architecture.

The ambiguity line has established:

- GDM57: request-conditioned semantic evidence materially improves ambiguity recovery.
- GDM58: simply widening pooled top-k evidence is negative/inconclusive.
- GDM59: preserving top-five rank/identity helps through request-relative semantic interactions.
- GDM60: simple confidence attenuation is negative.
- GDM61: fixed product/delta rescaling is negative/inconclusive.
- GDM62: fixed candidate-to-candidate relations do not improve the broad operation frontier.
- GDM63: a small learned rank-preserving scalar gate over cross-candidate relations does not improve the broad frontier over a matched-capacity local control.
- **GDM64: deterministic parent-path/leaf-field semantic factorization does not improve the broad frontier over the exact whole-candidate request-relative control; split-product sharply worsens role behavior.**

### Validation-only arbitration and deterministic realization

Capability and ambiguity heads are optimizer-trained. Status thresholds and request arbitration are validation-calibrated rather than trained status models. The active canonical line retains GDM54's high-confidence ambiguity-rescue policy with a five-percentage-point NO_MATCH-recall budget and accepted-status recall guardrail.

A remaining NO_MATCH clause rejects the request; otherwise an AMBIGUOUS clause makes the request ambiguous; otherwise selected paths are converted to a closed selection tree. The operation is parsed and validated with `graphql-core`, then executed against changing fixtures that distinguish author/moderator, name/ID, creation/update, and other semantically close paths. Wrong-but-valid GraphQL fails response comparison.

Real parsing, validation, and execution are mandatory evidence. A green workflow, manifest, estimate, or simulated score is not evidence.

## Forward measurement contract

Normal post-suspension batches contain **10 measured trained configurations: one operation task × five arms × two seeds**, unless a bounded prespecification declares another count.

Promotion reports answerable exact-operation accuracy, risk accuracy, incorrect-publication rate, target precision/recall, NO_MATCH and AMBIGUOUS correctness, exact one-/two-/three-clause correctness, semantic confusion classes and ambiguity families, regression versus fresh secondary holdout, measured p50/p95, and process-level RSS with memory scope stated explicitly.

Fresh holdouts never enter optimizer, checkpoint, representation, curriculum, or calibration selection. After inspection they become regression-only. Seeds and exact reruns are reproducibility evidence, not additional semantic worlds. Every accepted compile, test, training run, validation, benchmark, and independent evidence check runs in **GitHub Actions**.

## What the experiment line has taught us

| Experiment | Measured learning | Forward consequence / limit |
| --- | --- | --- |
| GDM45 | Clause-local decisions improve multi-capability behavior over whole-request scoring. | Keep clause-local operation decisions. |
| GDM46–49 | Strong rejection gates can improve risk handling while destroying answerable recall. | Report risk and answerable correctness separately. |
| GDM50 | Explicit NONE provides an open-set unsupported outcome; hardened numerical work exposed reproducibility drift. | Keep explicit NONE and canonical numerical execution. |
| GDM51 | A separate ambiguity discriminator improves capability-vs-risk decomposition. | Ambiguity deserves a learned component. |
| GDM52–54 | Calibration/arbitration move tradeoffs but cannot repair weak representation evidence. | Keep calibration separate from representation capability. |
| GDM55–56 | Ambiguity curriculum diversity/family balance matter, but more curriculum is not monotonic. | Curriculum alone is not the missing representation. |
| GDM57 | Request-conditioned top-two semantics improve AMBIGUOUS recovery. | Useful semantic ambiguity signal exists. |
| GDM58 | Wider top-k through simple pooling produces little operation benefit. | Pooled candidate breadth is negative/inconclusive. |
| GDM59 | Rank-preserving top-five request interactions improve useful signal versus pooled controls. | Preserve candidate rank/identity. |
| GDM60 | Simple confidence weighting loses answerable recall. | Confidence attenuation is negative. |
| GDM61 | Fixed product/delta scaling does not improve the within-batch answerable/safety frontier. | Scalar rescaling is negative/inconclusive. |
| GDM62 | Fixed rank-1 candidate delta/product blocks do not beat request-relative control and can worsen role discrimination. | Retain request-relative ranked semantics. |
| GDM63 | Matched-capacity learned local/top1/neighbor/all-pairs/competitive gates do not improve answerable accuracy, precision/recall, clause exactness, or pooled ambiguity-family behavior. | Do not retain the learned gate. Scalar candidate-set interaction is not the missing semantic signal. |
| **GDM64** | **Parent-only, leaf-only, and split parent/leaf semantic views fail to improve the exact whole-candidate control. Split-delta is effectively tied; split-product drops answerable accuracy/recall and collapses pooled role correctness to 1/8.** | **Do not retain path-factorized embedding replacement. Keep whole-candidate request-relative semantics and test explicit structured schema relations instead of another embedding recombination.** |

Historical schema findings through GDM61 remain available in prior commits/artifacts but are frozen and are not forward research targets.

# Latest canonical operation evidence: GDM64

## Architecture delta and experiment contract

GDM64 tests whether whole-candidate embeddings collapse distinctions such as parent role (`author` vs `moderator`) and leaf representation (`name` vs `id`) before the ambiguity head sees them. It keeps capability training, explicit NONE, frozen DistilBERT, the `1e-4` boundary, GDM56 family-balanced ambiguity curriculum with zero counterfactuals, GDM50 numerical execution, GDM54 validation-only rescue, TOP5 rank identity, the same `dim -> 32 -> 1` ambiguity MLP capacity, and deterministic real GraphQL realization fixed.

Every arm remains exactly **11 scalar/structural features + ten rank-preserving semantic slots = `11 + 10*d = 7,691` DistilBERT dimensions**. No GDM64 arm adds a learned adapter. Only the deterministic semantic source of the ten slots differs:

- `whole-request-control`: `(q*ci, abs(q-ci))`, the exact retained canonical representation;
- `parent-request-factor`: `(q*pi, abs(q-pi))`, where `pi` encodes `GraphQL parent path: <path excluding root and leaf>`;
- `leaf-request-factor`: `(q*li, abs(q-li))`, where `li` encodes `GraphQL leaf field: <leaf>`;
- `split-parent-leaf-product`: `(q*pi, q*li)`;
- `split-parent-leaf-delta`: `(abs(q-pi), abs(q-li))`.

Parent/leaf strings are derived **only from the provided GraphQL candidate path**. They are cached catalog evidence and never alter capability ranking. `Registry` / `Logbook` are calibration-only. `Folio` / `Casebook` are the fresh GDM64 secondary holdout and, after this canonical inspection, are now **regression-only**.

## Verified operation metrics

The GDM64 holdout has 84 cases per seed: **36 answerable + 48 risk**, where risk comprises **32 NO_MATCH + 16 AMBIGUOUS**. Means below are over seeds `6401,6402`. Status counts pool both seeds: AMBIGUOUS denominator **32**, NO_MATCH denominator **64**. Clause denominators pool both seeds: **32 one-clause, 24 two-clause, 16 three-clause** evaluations.

| Arm | Answerable accuracy | Risk accuracy | Incorrect publication | Target precision | Target recall | AMBIGUOUS | NO_MATCH | 1/2/3-clause exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Whole-request control** | **54.17%** | **58.33%** | 28.57% | **89.46%** | **57.03%** | **27/32** | 29/64 | **20/32 · 12/24 · 7/16** |
| Parent request factor | **54.17%** | 56.25% | 29.17% | **89.46%** | **57.03%** | 25/32 | 29/64 | **20/32 · 12/24 · 7/16** |
| Leaf request factor | **54.17%** | 55.21% | 29.17% | **89.46%** | **57.03%** | 25/32 | 28/64 | **20/32 · 12/24 · 7/16** |
| Split parent/leaf delta | **54.17%** | **58.33%** | **27.98%** | **89.46%** | **57.03%** | **27/32** | 29/64 | **20/32 · 12/24 · 7/16** |
| Split parent/leaf product | 44.44% | 55.21% | 28.57% | 87.10% | 44.53% | 17/32 | **36/64** | 18/32 · 9/24 · 5/16 |

The split-delta arm is effectively tied with the exact whole-request control on answerable, risk, target precision/recall, AMBIGUOUS, NO_MATCH, and clause exactness; its 0.59-point lower incorrect-publication mean does not establish a robust architecture win, especially given the seed variance below. Parent-only and leaf-only preserve answerable/target metrics but reduce ambiguity/risk handling. Split-product shifts toward NO_MATCH while materially losing answerable correctness, recall, AMBIGUOUS recovery, and multi-clause exactness.

The supported conclusion is therefore **negative/inconclusive for replacing whole-candidate request-relative semantic slots with deterministic parent/leaf embedding factorization**. The retained representation remains `q*ci + abs(q-ci)`.

## Fresh ambiguity-family analysis

Each family has 8 pooled cases across the two seeds.

| Arm | Role | Lifecycle-time | Representation / name-vs-ID | Object-vs-supplier |
| --- | ---: | ---: | ---: | ---: |
| **Whole-request control** | **7/8** | **6/8** | **7/8** | **7/8** |
| Parent request factor | **7/8** | **6/8** | **7/8** | 5/8 |
| Leaf request factor | **7/8** | **6/8** | **7/8** | 5/8 |
| Split parent/leaf delta | **7/8** | **6/8** | **7/8** | **7/8** |
| Split parent/leaf product | **1/8** | 4/8 | **7/8** | 5/8 |

Neither parent-only nor leaf-only improves the targeted role or representation families over the whole control. Split-delta again exactly ties those families. Split-product is strongly harmful to role discrimination, reaching only **1/8** pooled role correctness. This is especially important because the purpose of the factorization was to expose role/representation distinctions more clearly.

Dominant semantic confusions persist in the verified aggregate evidence. Whole-control, parent-only, leaf-only, and split-delta each include `folio.reviews.author.name -> folio.reviews.author.id` (**4**), `casebook.reviews.author.name -> casebook.reviews.author.id` (**4**), `folio.reviews.moderator.name -> folio.reviews.author.id` (**2**), `casebook.reviews.moderator.name -> casebook.reviews.author.id` (**2**), and a smaller `folio.updatedAt -> folio.createdAt` confusion (**1**). Split-product changes some counts but does not repair the underlying role/name-vs-ID failure and performs worse overall.

## Seed stability, regression, latency, and memory

GDM64 remains highly seed-sensitive. For the exact whole-request control, seed `6401` vs `6402` is **41.67% vs 66.67% answerable accuracy** (25 points), **62.50% vs 54.17% risk accuracy**, **16.67% vs 40.48% incorrect publication**, **96.15% vs 82.76% target precision**, and **39.06% vs 75.00% target recall**. Split-delta shows the same 25-point answerable gap and 35.94-point recall gap, with incorrect publication **16.67% vs 39.29%**. Parent and leaf factor arms show similarly large seed movement. Split-product is more stable on answerable accuracy (**41.67% vs 47.22%**) but at substantially worse mean quality, so that is not a useful stability gain.

Mean operation regression accuracy is approximately **48.30%–49.82%** across arms. These inspected regressions plus `Folio` / `Casebook` are now regression-only. Fresh holdouts from different numbered batches are not a causal leaderboard.

Hosted-runner generation measurements are observational. Source attempt 1 aggregate means span approximately **77.27–79.59 ms p50**, **152.27–155.69 ms p95**, and **1,264,818–1,268,900 KiB RSS-after-evaluation**. Source attempt 2 spans approximately **73.75–74.65 ms p50**, **145.99–147.68 ms p95**, and **1,260,666–1,267,414 KiB RSS**. RSS is whole-process memory, not isolated model memory, and runner-to-runner timing movement is not a deterministic model field.

## GDM64 reproducibility, audit diagnostics, and provenance

Measured source commit: [`6fe0f89630324fa18c18175c1d7c64ba614a1cff`](https://github.com/burn2delete/graph-model/commit/6fe0f89630324fa18c18175c1d7c64ba614a1cff). Source workflow run: [`35995968508`](https://github.com/burn2delete/graph-model/actions/runs/35995968508), two independently verified 10/10 attempts on the same source and seeds `6401,6402`.

Source attempt evidence:

- Attempt 1 preflight artifact **10806327006**, SHA-256 `3a4a8fd3e45fdf6d2b144ee34c37c5bfb1b5262c3857442c0587227676605044`; operation artifact **10806744697**, SHA-256 `b82bdcecd8287e40f3464b5abed355baae131e93b5fc13d875f62ebdf445d2ac`; source operation `BATCH_REPORT` artifact **10806439103**, SHA-256 `7265c0c821c2b5d483efdcf1dc15ba7bdfd4d7eccd22dfdfaa1ff31a69667a05`.
- Attempt 2 preflight artifact **10808385906**, SHA-256 `eaa0b7a3e3ddf8eae40d2ecfa02d395f4ab4d7c9ed9853b459210827f7378e6e`; operation artifact **10809460402**, SHA-256 `cf81d28921824ce9b15aad4ee3c89d00bb3d38fb1297a2c86343b0a099ef50da`; source operation `BATCH_REPORT` artifact **10809186317**, SHA-256 `701f7edef0c7f521c4b60781e6ce4866fc58b1d109ca90d8fabab34b2b913d2c`.

There were three **audit-orchestration failures that are not model evidence** and are preserved as diagnostics:

1. audit workflow commit `36d6575170252fea817e71c6f725bfc942fce330` produced invalid-YAML run `36007019132` with no jobs;
2. audit-driver addition `1a29ce5b5008e1b8c864872f68ad0eb0aedd8b46` also produced invalid-YAML run `36007277958`;
3. syntax repair `07f71818ace16b253b8d5af54b89adec83174f65` produced valid run `36007347666` / job `107658827524`, but the audit failed before evidence comparison because its template transformer assumed the wrong indentation for `adapter_counts`. Diagnostic artifact **10810464477**, SHA-256 `8ab329fc1716c05c2f68db2050a2edff4f9d995d561d7a19bce41b81b57554dc`.

None of those repairs changed source model architecture, data, training, optimizer, representation, calibration, seeds, holdout, or GraphQL semantics.

The audit driver was repaired audit-only at commit [`39b97e44a822c1a6d31812af88fb8ae62b2885c7`](https://github.com/burn2delete/graph-model/commit/39b97e44a822c1a6d31812af88fb8ae62b2885c7), then triggered at [`d64a8f2fb4ae5aa6d59e0f2b7898a9d3d6dfbfb8`](https://github.com/burn2delete/graph-model/commit/d64a8f2fb4ae5aa6d59e0f2b7898a9d3d6dfbfb8). Passing exact-attempt audit run: [`36007882683`](https://github.com/burn2delete/graph-model/actions/runs/36007882683), job `107660653253`.

Passing audit evidence:

- Independently rebuilt attempt-1 operation `BATCH_REPORT`: artifact **10811083586**, SHA-256 `b16dd44a91953f8ff5038595fd31718dd6205b3c2d18d20da696a1415978cde8`.
- Independently rebuilt attempt-2 operation `BATCH_REPORT`: artifact **10811408339**, SHA-256 `1d51113fd3905852c966193b29d03044ce4eafdb2638bf9dba88d0e3daa3b4b5`.
- Passing audit artifact **10811273396**, SHA-256 `7f4586e5d5463680e5671f0217c3d1c0cc4a6e6212937f98ce2023e69b024786`.

The audit passed **10/10 operation configs exactly with schema evidence excluded**. Its gate is `complete=true`, `evidence_verified=true`, `reproducible=true`, `promotion_eligible=true`, `errors=[]`, `expected_operation_configs=10`, `matched_operation_configs=10`, `schema_configs_compared=0`, with all ten comparisons exact and deterministic fields equal. Aggregate p50/p95/RSS remain observational fields.

An Actions-only promotion analysis run [`36013873159`](https://github.com/burn2delete/graph-model/actions/runs/36013873159) at commit [`906a1ff83468b8656bf63ec1f4ee1f4c0f7f09ff`](https://github.com/burn2delete/graph-model/commit/906a1ff83468b8656bf63ec1f4ee1f4c0f7f09ff) independently checked the passing audit/source receipts and aggregated operation-only arm, family, seed, regression, latency, and memory evidence without retraining. Analysis artifact **10814036333**, SHA-256 `a85a05a2cb7367b6eca049e7a52232d205f8b67522a72a50587880e3cd428f2f`.

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
| GDM61 | source `f22fc36caf7327ed3d0d1f7dd2898ea31d837f3a`, run `35939206923` | `35948010054` |
| GDM62 operation-only | source `a3cfa70a570f931e2c78553ca9d72aeb5ed65e79`, run `35953082798` | `35965921793` |
| GDM63 operation-only | source `43a6d16652a3fd63c155f46e99b42c757c7601a1`, run `35976527471`, attempts 1/2 | `35988393672` |
| **GDM64 operation-only** | source **`6fe0f89630324fa18c18175c1d7c64ba614a1cff`**, run **`35995968508`**, attempts 1/2 | **`36007882683`** |

## Evidence and promotion policy

A post-suspension canonical promotion requires complete declared **operation** results, changed learned-head checkpoint evidence, optimizer histories, validation-only checkpoint/calibration selection, per-example predictions/generated GraphQL, real parse/validation/execution checks, raw timing/runtime metadata, an exact same-source/same-seed rerun, and a dedicated independent operation-only exact-attempt audit.

The root README must be synchronized **before** announcing promotion or launching the next numbered experiment. Promotion commits record the exact measured source, source attempts, independently rebuilt operation BATCH_REPORT artifact IDs/digests, passing audit artifact ID/digest, denominators, regression/holdout limits, p50/p95, memory scope, supported and negative learnings, and the next unverified operation hypothesis.

Historical schema metrics/artifacts are retained for provenance but are frozen and not part of new promotion gates.

## Current limits and next decision gate

The measured system remains a small synthetic, schema-grounded research environment. It has not established unrestricted operation token generation, general mutation/subscription generation, arbitrary enterprise topology handling, learned clause decomposition, transformer fine-tuning benefits, or production-scale latency/memory behavior.

GDM62–GDM64 now rule out three increasingly direct variants of the same broad idea under the tested contracts: **fixed candidate-to-candidate algebraic relations, learned scalar cross-candidate gates, and deterministic parent/leaf semantic embedding factorization**. None repaired the persistent author-name/ID and moderator/author confusions while improving the broad answerable/safety frontier. GDM64 is particularly informative because it explicitly exposed the intended parent and leaf text views, yet parent/leaf replacement did not improve role or representation family correctness over the whole-candidate control, and the product split strongly harmed role discrimination.

The next unverified hypothesis should therefore change the information type rather than make semantic embedding recombination more elaborate: **the ambiguity head may need an explicit structured schema-relation channel describing relations among ranked schema coordinates**. Candidate features can be derived deterministically from the provided schema/catalog—for example same-parent/sibling status, same leaf, path-prefix/longest-common-prefix structure, object-vs-scalar/type-signature relation, or a bounded categorical relation code—while retaining the canonical whole-candidate request-relative semantic slots. Any next experiment must prespecify matched bounded capacity, avoid hidden labels or holdout-derived features, retain capability ranking and the frozen numerical/calibration/GraphQL contracts, and use a fresh operation-only holdout.

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
| [`experiments/followup61/`](experiments/followup61/) | Fixed product/delta balance; historical canonical evidence |
| [`experiments/followup62/`](experiments/followup62/) | Fixed candidate-relational operation evidence |
| [`experiments/followup63/`](experiments/followup63/) | Matched-capacity learned candidate-interaction operation evidence |
| [`experiments/followup64/`](experiments/followup64/) | Schema-coordinate parent/leaf factorization operation evidence; latest canonical experiment |
| [`.github/workflows/`](.github/workflows/) | Actions-only measured batches, audits, and evidence analyses |

Use each experiment's declared workflow/execution wrapper when reproducing it. Calling historical Python entrypoints directly can bypass the canonical numerical contract.