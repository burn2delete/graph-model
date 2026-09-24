# Graph Model

**A GraphQL Decision Model (GDM): given an intent and a provided GraphQL schema, learn the decisions required to construct a correct executable GraphQL operation.**

## Forward research scope

As of **2026-09-24, forward research is operation-generation only**:

> **natural-language intent/request + provided GraphQL schema/capability catalog → correct executable GraphQL operation**

The system must also know when not to publish an operation: unsupported requests should resolve to **NO_MATCH**, and semantically underdetermined requests should resolve to **AMBIGUOUS**.

**Schema-generation research is suspended/frozen. Query planning is out of scope.** Historical schema evidence through GDM61 is retained as provenance/regression history only; it does not drive new architecture choices, measured arms, promotion decisions, or forward benchmarks.

The current measured system is deliberately narrower than unrestricted token-level GraphQL synthesis. A **frozen semantic encoder plus learned capability and ambiguity heads** maps request clauses onto schema-grounded capabilities. Deterministic code realizes accepted capabilities as real GraphQL, which is parsed, validated, and executed. The transformer backbone is not fine-tuned.

Read [`experiments/MEASUREMENT_POLICY.md`](experiments/MEASUREMENT_POLICY.md) and [`experiments/measured/REPAIR.md`](experiments/measured/REPAIR.md) before interpreting results or changing experiments.

## Current research state

_Last synchronized: 2026-09-24. Latest canonical experiment: GDM63. Forward scope: operation generation only._

| State | Meaning |
| --- | --- |
| **Canonical through GDM63** | GDM50–GDM61 retain historical measured evidence/audits; GDM62 and GDM63 are post-suspension operation-only promotions. Canonical means reproducible usable evidence under the declared contract, not automatic architecture victory. |
| **Forward task** | Generate an executable GraphQL operation from an intent and provided schema/catalog, while correctly handling answerable, NO_MATCH, and AMBIGUOUS requests. |
| **Schema generation** | **Suspended/frozen.** GDM41–61 schema evidence remains historical provenance only. No new schema arms are launched, scored, optimized, or promoted. |
| **Latest supported operation result** | **GDM63 is negative/inconclusive for a small matched-capacity learned rank-preserving cross-candidate gate.** All five learned relation modes have identical answerable accuracy, precision, recall, clause exactness, ambiguity-family counts, and the same dominant semantic confusions; only a few NO_MATCH decisions move. The retained architecture therefore remains the canonical ranked request-relative `q*ci + abs(q-ci)` representation without the GDM63 adapter. |
| **Next unverified hypothesis** | The persistent failures may come from **semantic factorization inside each schema coordinate**, not from candidate-set comparison. A bounded next experiment should test whether separately representing path/role semantics (for example parent-role vs leaf/representation semantics) gives the ambiguity head information that whole-candidate embeddings and scalar cross-candidate gates fail to expose. This is unverified and must be prespecified with matched capacity before launch. |
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

**GDM63 does not change the retained architecture.** It added an equal-capacity learned interaction adapter to every experimental arm, including a rank-local matched-capacity control. The cross-candidate relation mode changed, but the raw top-five request-relative representation stayed fixed. The cross-candidate modes did not improve the broad operation frontier or the difficult ambiguity families, so the adapter is an experimental negative rather than a retained architecture change.

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
- **GDM63: a small learned rank-preserving scalar gate over cross-candidate relations also does not improve the broad frontier or ambiguity-family behavior over a matched-capacity rank-local control.**

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
| **GDM63** | **Matched-capacity learned local/top1/neighbor/all-pairs/competitive gates all produce the same answerable accuracy, precision, recall, clause exactness, and pooled ambiguity-family counts. Cross-candidate modes move only a few NO_MATCH calls.** | **Do not retain the learned gate. Candidate-set interaction, at least when reduced to a per-rank scalar modulation of existing slots, is not the missing semantic signal. Test a different information factorization rather than a larger version of the same gate.** |

Historical schema findings through GDM61 remain available in prior commits/artifacts but are frozen and are not forward research targets.

# Latest canonical operation evidence: GDM63

## Architecture delta and experiment contract

GDM63 keeps capability training, explicit NONE, frozen DistilBERT, the `1e-4` feature boundary, exact GDM56 family-balanced ambiguity curriculum with zero counterfactuals, GDM50 numerical execution, GDM54 validation-only rescue, top-five rank identity, and deterministic real GraphQL realization fixed.

Every arm receives the exact canonical raw ambiguity vector: **11 scalar/structural features + five `q*ci` slots + five `abs(q-ci)` slots = `11 + 10*d = 7,691` DistilBERT dimensions**. Every arm also gets the same bounded learned adapter and the same downstream `dim -> 32 -> 1` ambiguity MLP. The adapter uses a shared `Linear(2d -> 4)` projection, five learned rank strengths, temperature, residual scale, and interaction bias. Parameter shapes/counts and optimizer treatment are identical across arms. Only the deterministic relation used to derive a learned per-rank scalar gate differs:

- `learned-local-control`: self-energy only, no cross-rank relation;
- `learned-top1-cross`: rank `i` against rank 1;
- `learned-neighbor-cross`: rank `i` against the preceding rank;
- `learned-allpairs-cross`: mean relation to all other active ranks;
- `learned-competitive-cross`: strongest-other minus mean-other relation.

The primary causal comparator is therefore the **matched-capacity local control**, not historical GDM62 on a different fresh holdout.

`Ledger` / `Index` are calibration-only. `Docket` / `Portfolio` are the fresh GDM63 secondary holdout and, after this canonical inspection, are now **regression-only**.

## Verified operation metrics

The holdout has 84 cases per seed: **36 answerable + 48 risk**, where risk comprises **32 NO_MATCH + 16 AMBIGUOUS**. Means below are over seeds `6301,6302`. Status counts pool both seeds: AMBIGUOUS denominator **32**, NO_MATCH denominator **64**. Clause denominators pool both seeds: **32 one-clause, 24 two-clause, 16 three-clause** evaluations.

| Arm | Answerable accuracy | Risk accuracy | Incorrect publication | Target precision | Target recall | AMBIGUOUS | NO_MATCH |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Learned local control** | **26.39%** | 63.54% | **12.50%** | **91.67%** | **22.66%** | **22/32** | 39/64 |
| Learned top1 cross | **26.39%** | 63.54% | **12.50%** | **91.67%** | **22.66%** | **22/32** | 39/64 |
| Learned neighbor cross | **26.39%** | 61.46% | **12.50%** | **91.67%** | **22.66%** | **22/32** | 37/64 |
| Learned all-pairs cross | **26.39%** | 63.54% | **12.50%** | **91.67%** | **22.66%** | **22/32** | 39/64 |
| Learned competitive cross | **26.39%** | **64.58%** | **12.50%** | **91.67%** | **22.66%** | **22/32** | **40/64** |

All five arms have exact one-/two-/three-clause correctness **12/32, 6/24, 1/16**. Thus cross-candidate learning provides **no answerable, target-level, ambiguity, or multi-clause gain** over the matched-capacity local control. Competitive-cross recovers one additional NO_MATCH case; neighbor-cross loses two. That narrow status movement is insufficient to establish a useful cross-candidate architecture improvement.

The same dominant semantic confusions occur in every arm: `docket.reviews.author.name -> docket.reviews.author.id` (**3**) and `docket.reviews.moderator.name -> docket.reviews.author.id` (**2**). This is direct evidence that the tested interaction modes do not repair the persistent role and name-vs-ID distinctions.

## Fresh ambiguity-family analysis

Each family has 8 pooled cases across the two seeds. **Every GDM63 arm has exactly the same pooled family result**:

| Family | Correct |
| --- | ---: |
| Role | **5/8** |
| Lifecycle-time | **4/8** |
| Representation / name-vs-ID | **5/8** |
| Object-vs-supplier | **8/8** |

Object-vs-supplier remains solved in this synthetic slice, lifecycle-time remains at chance-like 4/8, and the two most relevant unresolved families—role and representation—remain 5/8 regardless of interaction mode. Candidate-to-candidate gating therefore does not expose the missing semantic distinction.

## Seed stability, regression, latency, and memory

GDM63 exposes substantial seed sensitivity that is shared by every arm. Seed 6301 vs 6302 answerable accuracy is **19.44% vs 33.33%** (13.89 points), incorrect publication **4.76% vs 20.24%** (15.48 points), target precision **100% vs 83.33%**, and recall **14.06% vs 31.25%**. Risk seed gaps vary by relation mode from 2.08 to 8.33 points. The cross-candidate arms therefore do not provide a stability advantage.

Mean operation regression accuracy is approximately **49.36%–49.85%** across arms. These inspected regressions plus `Docket` / `Portfolio` are now regression-only. Fresh holdouts from different numbered batches are not a causal leaderboard.

Hosted-runner generation measurements are observational. Source attempt 1 means span approximately **44.75–48.65 ms p50**, **87.42–93.00 ms p95**, and **1,226,844–1,229,306 KiB RSS-after-evaluation**. Source attempt 2 aggregate p50 spans approximately **67.45–68.23 ms**, p95 **135.06–136.56 ms**, and RSS approximately **1,226,026–1,228,368 KiB**. The runner-to-runner timing shift reinforces why p50/p95/RSS are observational rather than exact reproducibility fields. RSS is whole-process memory, not isolated model memory.

## GDM63 failures, reproducibility, and provenance

The original measured source [`dd3ed0df6cc835b90d7a003c661d7caafe7dff0e`](https://github.com/burn2delete/graph-model/commit/dd3ed0df6cc835b90d7a003c661d7caafe7dff0e) failed preflight in run `35971746710` after all 94 tests and the hash smoke because `training.gdm63_change_scope` used a shorter metadata receipt than the independent collector expected. The operation worker and batch verification were skipped; this run contains **no full measured GDM63 model evidence**. Failed preflight artifact **10796397961**, SHA-256 `78d33e0dd629f41825cf784975233354687330acb78c8beb3bc00d28429b099b`, is preserved as diagnostic history.

The in-place repair [`43a6d16652a3fd63c155f46e99b42c757c7601a1`](https://github.com/burn2delete/graph-model/commit/43a6d16652a3fd63c155f46e99b42c757c7601a1) changed only that metadata receipt string. It did **not** change architecture, parameters, data, curriculum, seeds, holdout, optimizer, representation, calibration, GraphQL realization, or the `1e-4` boundary.

The repaired source run [`35976527471`](https://github.com/burn2delete/graph-model/actions/runs/35976527471) produced two independently verified 10/10 attempts on the same source/seeds:

- Attempt 1 source operation artifact **10799480293**, SHA-256 `07e991bbfffdeeaefd45a65e66782fbf19b5ce24889ab9d5f717d4cd7f515050`; source operation `BATCH_REPORT` artifact **10798847385**, SHA-256 `d29a5913e7190997f7f41a3ac2c87c510b1d840cfe4f44c09ca2cab82337c98f`.
- Attempt 2 source operation artifact **10801593060**, SHA-256 `b12ad13130292033f9389ac67db819c6f0a74d228584390ad04a93e56e24e919`; source operation `BATCH_REPORT` artifact **10801696846**, SHA-256 `f2b44d7e91a071838b66daf9316d9130cb93dbe29dca1e42da4d44a0050dc464`.

The dedicated exact-attempt audit is run [`35988393672`](https://github.com/burn2delete/graph-model/actions/runs/35988393672) at audit-only commit [`53ac6ec72857d195b2f74f0975944525e388df55`](https://github.com/burn2delete/graph-model/commit/53ac6ec72857d195b2f74f0975944525e388df55). It independently rebuilt both operation reports and verified actual full/capability/ambiguity/interaction checkpoint hashes, changed learned interaction state including `ambiguity.proj.weight`, matched parameter counts, TOP5/7,691 representation receipts, numerical/curriculum/calibration contracts, canonical feature hashes, exact regression/holdout/clause/family metrics, and exact per-example predictions. It compared all **10/10 operation configs exactly** with `schema_configs_compared=0`; only aggregate observational p50/p95/RSS may differ across source attempts.

Passing audit evidence:

- Independently rebuilt attempt-1 operation `BATCH_REPORT`: artifact **10803551517**, SHA-256 `732210dbecfa54d0afb738dde1cc9b2d9d50a4a691fe8a2f16bf1df8da61d25a`.
- Independently rebuilt attempt-2 operation `BATCH_REPORT`: artifact **10803077837**, SHA-256 `73e3423984c26904c066bc4ef15d6f221c95409289a612fd757df201882bcab8`.
- Passing audit artifact **10803611438**, SHA-256 `f3c0e7e86665cfac54f14b3c81bb04de16fc9e6d3ad76a830935f1836234a6a7`.

The `gdm63-operation-reproducibility-audit-v1` gate has `complete=true`, `evidence_verified=true`, `reproducible=true`, `promotion_eligible=true`, `errors=[]`, `expected_operation_configs=10`, `matched_operation_configs=10`, `schema_configs_compared=0`, all ten comparisons `exact_match=true`, and `differing_fields=[]`.

An Actions-only promotion analysis run [`35994447306`](https://github.com/burn2delete/graph-model/actions/runs/35994447306) at commit [`38b1b59268db6358937b1586aed7fe1c954624ee`](https://github.com/burn2delete/graph-model/commit/38b1b59268db6358937b1586aed7fe1c954624ee) independently checked the passing audit gate and aggregated seed/family/runtime evidence without retraining. Analysis artifact **10805078417**, SHA-256 `8ff5882d9e996106504354d275303445aa65b6a3d2a8a6c23491bd93d432d11c`. `Docket` / `Portfolio` becomes regression-only with this inspection.

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
| **GDM63 operation-only** | source **`43a6d16652a3fd63c155f46e99b42c757c7601a1`**, run **`35976527471`**, attempts 1/2 | **`35988393672`** |

## Evidence and promotion policy

A post-suspension canonical promotion requires complete declared **operation** results, changed learned-head checkpoint evidence, optimizer histories, validation-only checkpoint/calibration selection, per-example predictions/generated GraphQL, real parse/validation/execution checks, raw timing/runtime metadata, an exact same-source/same-seed rerun, and a dedicated independent operation-only exact-attempt audit.

The root README must be synchronized **before** announcing promotion or launching the next numbered experiment. Promotion commits record the exact measured source, source attempts, independently rebuilt operation BATCH_REPORT artifact IDs/digests, passing audit artifact ID/digest, denominators, regression/holdout limits, p50/p95, memory scope, supported and negative learnings, and the next unverified operation hypothesis.

Historical schema metrics/artifacts are retained for provenance but are frozen and not part of new promotion gates.

## Current limits and next decision gate

The measured system remains a small synthetic, schema-grounded research environment. It has not established unrestricted operation token generation, general mutation/subscription generation, arbitrary enterprise topology handling, learned clause decomposition, transformer fine-tuning benefits, or production-scale latency/memory behavior.

GDM63 rules out another narrow architecture family: **a small learned cross-candidate relation that only produces a scalar per-rank gate over the existing request-relative semantic slots does not improve the matched-capacity local control.** The experiment is especially informative because the learned interaction parameters provably changed, yet every arm retained the same answerable accuracy, precision/recall, clause exactness, pooled family outcomes, and dominant semantic confusions.

The next unverified hypothesis shifts the information boundary rather than making the gate larger: **the ambiguity head may need schema-coordinate semantic factorization—separate evidence for parent/role path semantics and leaf/representation semantics—because a single whole-candidate embedding can collapse distinctions such as author vs moderator and name vs ID before candidate-set interaction is applied.** This remains a hypothesis, not a selected GDM64 design. Any GDM64 prespecification must remain operation-only, keep the frozen encoder/numerical/calibration/GraphQL contracts fixed, use matched bounded capacity, and establish that any new path-factorized features come only from the provided schema/catalog and never from the fresh holdout.

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
| [`experiments/followup63/`](experiments/followup63/) | Matched-capacity learned candidate-interaction operation evidence; latest canonical experiment |
| [`.github/workflows/`](.github/workflows/) | Actions-only measured batches, audits, and evidence analyses |

Use each experiment's declared workflow/execution wrapper when reproducing it. Calling historical Python entrypoints directly can bypass the canonical numerical contract.
