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

_Last synchronized: 2026-09-24. Latest canonical experiment: GDM62. Forward scope: operation generation only._

| State | Meaning |
| --- | --- |
| **Canonical through GDM62** | GDM50–GDM61 retain their historical measured evidence/audits; GDM62 is the first post-suspension promotion under the operation-only gate. Canonical means reproducible usable evidence under the declared contract, not automatic architecture victory. |
| **Forward task** | Generate an executable GraphQL operation from an intent and provided schema/catalog, while correctly handling answerable, NO_MATCH, and AMBIGUOUS requests. |
| **Schema generation** | **Suspended/frozen.** GDM41–61 schema evidence remains historical provenance only. No new schema arms are launched, scored, optimized, audited for promotion, or used to pick forward architecture. |
| **Latest supported operation result** | **GDM62 is negative/inconclusive for explicit rank-1 candidate-to-candidate semantic blocks.** The canonical request-relative control remains the retained architecture: no candidate-relational arm improves the broad answerable/safety frontier, clause-count correctness, or fresh ambiguity-family profile. |
| **Next unverified hypothesis** | A learned rank-preserving candidate-set comparison may be required to distinguish plausible alternatives; fixed algebraic relations to the rank-1 candidate are insufficient. The next experiment must remain operation-only and should specifically test whether learned cross-candidate interaction improves persistent role/representation confusions without sacrificing publication safety. |
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

**GDM62 does not change the retained architecture.** It tested four matched alternatives that replaced one or both request-relative semantic blocks with explicit rank-1-candidate relations. Those alternatives are reproducible but do not improve the broad operation frontier, so the retained ambiguity representation remains the ranked request-relative control `q*ci + abs(q-ci)` rather than a GDM62 candidate-relational arm.

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
- **GDM62: fixed rank-1-to-candidate delta/product relations do not improve the broad operation frontier over the ranked request-relative control.**

The unresolved operation errors remain relational. In GDM62 the request-relative control retains **6/8** fresh role-family cases, while candidate-relational variants range from **0/8 to 5/8**; object-vs-supplier is already **8/8** for every arm, lifecycle-time remains **4/8** for every arm, and representation is **6/8** for the control and most arms. Hand-coded candidate-anchor transforms therefore do not resolve the difficult relation families.

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
| GDM61 | Fixed product/delta scaling does not improve the within-batch answerable/safety frontier; the GDM60 operation product-only endpoint does not replicate on fresh Voucher/Anthology. | Treat scalar rescaling as negative/inconclusive. |
| **GDM62** | **Explicit rank-1 candidate-to-candidate delta/product blocks are negative/inconclusive. No arm beats the request-relative control on the broad operation answerable/safety frontier, and the difficult role family gets worse in every candidate-relational arm.** | **Retain request-relative ranked semantics. Test a genuinely learned cross-candidate comparison next rather than another fixed algebraic transform or scalar rescaling.** |

Historical schema findings associated with GDM41–61 remain available in prior commits and artifacts but are frozen and are not forward research targets.

# Latest canonical operation evidence: GDM62

GDM62 keeps top-five candidate breadth and rank identity, explicit NONE, learned capability/ambiguity heads, the exact GDM56 family-balanced ambiguity curriculum, frozen DistilBERT, the `1e-4` feature boundary, canonical numerical execution, GDM54 validation-only rescue, and deterministic GraphQL realization fixed. The ambiguity head retains the same **`11 + 10*d = 7,691` DistilBERT input dimension** and matched trainable capacity across arms.

The experiment changes only the two rank-preserving semantic blocks supplied for each real candidate `ci`. The control is `(q*ci, abs(q-ci))`; the four alternatives introduce `abs(c1-ci)` or `c1*ci`, where `c1` is the rank-1 real candidate. This tests explicit candidate-to-candidate contrast without adding candidate breadth, parameters, transformer tuning, or a trained request-global status head.

`Medallion` / `Gazette` are calibration-only. `Ticket` / `Compendium` are the fresh GDM62 secondary holdout and, after this canonical inspection, are now **regression-only**.

The operation holdout has 84 unique cases per seed: **36 answerable + 48 risk**, where risk is **32 NO_MATCH + 16 AMBIGUOUS**. Rates below are means over seeds `6201,6202`; status counts pool the two seeds, so AMBIGUOUS has denominator 32 and NO_MATCH denominator 64. Clause-count denominators pool seeds: **32 one-clause, 24 two-clause, 16 three-clause** evaluations.

| Representation | Answerable accuracy | Risk accuracy | Incorrect publication | Target precision | Target recall | AMBIGUOUS | NO_MATCH |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Request-relative control** `q*ci + abs(q-ci)` | **37.50%** | **61.46%** | 20.24% | 91.55% | **39.84%** | **24/32** | 35/64 |
| Delta + candidate-product | **37.50%** | 60.42% | 21.43% | 91.55% | **39.84%** | 23/32 | 35/64 |
| Delta + candidate-delta | 34.72% | 54.17% | 22.62% | 90.59% | 36.72% | 19/32 | 33/64 |
| Product + candidate-product | 34.72% | 57.29% | 21.43% | 90.59% | 36.72% | 18/32 | **37/64** |
| Product + candidate-delta | 30.56% | 55.21% | **13.10%** | **94.59%** | 27.34% | 16/32 | **37/64** |

The candidate-relational hypothesis is therefore **negative/inconclusive**. `Delta + candidate-product` exactly matches control answerable accuracy and recall but loses risk accuracy and one AMBIGUOUS case while worsening incorrect publication. `Product + candidate-delta` materially lowers incorrect publication and raises precision, but it pays for that safety with a **6.94-point answerable loss**, a **12.50-point recall loss**, and AMBIGUOUS recovery falling from **24/32 to 16/32**. No candidate-relational arm produces a better overall frontier than the request-relative control.

Exact one-/two-/three-clause correctness is **17/32, 8/24, 2/16** for the request-relative control and delta+candidate-product; **16/32, 8/24, 1/16** for delta+candidate-delta and product+candidate-product; and **15/32, 7/24, 0/16** for product+candidate-delta. The experiment does not improve multi-clause composition, and three-clause exact correctness remains a major bottleneck.

### Fresh ambiguity-family analysis

Each family has 8 pooled cases across the two seeds.

| Representation | Role | Lifecycle-time | Representation/name-vs-ID | Object-vs-supplier |
| --- | ---: | ---: | ---: | ---: |
| **Request-relative control** | **6/8** | 4/8 | **6/8** | **8/8** |
| Delta + candidate-product | 5/8 | 4/8 | **6/8** | **8/8** |
| Delta + candidate-delta | 1/8 | 4/8 | **6/8** | **8/8** |
| Product + candidate-product | 0/8 | 4/8 | **6/8** | **8/8** |
| Product + candidate-delta | 0/8 | 4/8 | 4/8 | **8/8** |

No fresh ambiguity family is solved by the candidate-relational arms. Object-vs-supplier is already saturated across all arms. Lifecycle-time is unchanged. Representation/name-vs-ID does not improve. Most importantly, **role discrimination deteriorates in every candidate-relational arm**, consistent with the remaining author-vs-moderator semantic confusions. This is direct evidence against fixed rank-1 candidate anchoring as the missing relation signal.

### Seed stability, regression and runtime limits

The request-relative control is comparatively stable on answerable correctness (**36.11% vs 38.89%**, a 2.78-point seed gap) and risk accuracy (**60.42% vs 62.50%**, a 2.08-point gap), but incorrect publication still moves from **25.00% to 15.48%** across seeds. Candidate-relational arms do not show a consistent stability advantage: answerable seed gaps range from 0 to 8.33 points and incorrect-publication gaps from 0 to 14.29 points. The perfectly stable product+candidate-delta answerable/publication rates are stable at a substantially worse answerable/recall operating point, not evidence of a better model.

GDM62 mean operation regression request accuracy spans approximately **44.58%–51.16%** across arms. These inspected regressions and `Ticket` / `Compendium` are now regression-only; unlike fresh holdouts from other batches, they must not be used as a cross-batch causal leaderboard.

Hosted-runner generation measurements are observational. In source attempt 1, p50 spans approximately **74.52–76.96 ms**, p95 **136.55–141.05 ms**, and mean RSS-after-evaluation **1,181,294–1,185,910 KiB** across arms. In source attempt 2, p50 spans approximately **55.48–56.52 ms**, p95 **107.25–109.64 ms**, and mean RSS **1,181,040–1,186,302 KiB**. The large timing shift across hosted runners is why timing/RSS are observational rather than exact reproducibility fields. Timing includes fresh request-clause encoding, learned-head scoring, deterministic GraphQL realization, and local validation with catalog/NONE vectors cached; downloads, initial catalog embedding, and backend fixture execution are excluded. RSS is process-level and affected by worker reuse, not isolated model memory.

## GDM62 reproducibility and provenance

GDM62 source commit is **`a3cfa70a570f931e2c78553ca9d72aeb5ed65e79`**, source run **`35953082798`**, attempts 1 and 2. The source workflow was launched before schema research was suspended and therefore also ran schema workers; those schema artifacts are frozen historical diagnostics and are **excluded from this promotion**.

Operation source evidence:

- Attempt 1 operation worker artifact **10789199717**, SHA-256 `93a10707a60ed24c4b81e4bea789f348d78f1e43b2f86f3a046cf0e798b2f560`.
- Attempt 2 operation worker artifact **10791037384**, SHA-256 `d39ea60bb4aec5dc41bc91436b37998ae3bc8d3e483014c8eda50ca481792b5f`.

The first dedicated operation-only audit, run **35961242199** at workflow commit `54a777d87226ea831acfb52be771d794d77f8abb`, correctly exact-matched all **10/10 operation configurations** with `schema_configs_compared=0`, but failed its final aggregate-report assertion because it mistakenly treated hosted-runner p50/p95/RSS observations as deterministic cross-attempt fields. Failed audit artifact **10793190219**, SHA-256 `5b352f711ed7d0d9dfc7f2ee1a75de903606882892d7ad731fb97bac69ed3f01`, is preserved as diagnostic history; it is not promotion evidence.

The bounded audit repair changed no model, data, training, calibration, representation, seeds, predictions, or source evidence. It permits cross-attempt differences only in aggregate `mean_p50_ms`, `mean_p95_ms`, and `mean_rss_kib`, matching the declared observational timing/memory policy, while requiring every other BATCH_REPORT field and every per-config deterministic snapshot/prediction comparison to remain exact.

Passing operation-only audit repair:

- Audit workflow commit **`87436836fda0ce80f1acded1de129c2a848253c1`**.
- Passing audit run **`35965921793`**, job `107524346497`.
- Independently rebuilt attempt-1 operation `BATCH_REPORT`: artifact **10793952970**, SHA-256 `f5d52a09995406a70de44e61dcfd935e305d1e6037b6dadaef83ab6b2be06a02`.
- Independently rebuilt attempt-2 operation `BATCH_REPORT`: artifact **10793669216**, SHA-256 `796ab58772f8c051685d8b55047dc72ad9928c336e825fb5a36344c3dc60c5c3`.
- Passing audit artifact **10793838520**, SHA-256 `9097e616801052520d3e62266bccc1159534e330e41664f2db10d4135f61dff5`.

The repaired `gdm62-operation-reproducibility-audit-v1` gate has `complete=true`, `evidence_verified=true`, `reproducible=true`, `promotion_eligible=true`, `errors=[]`, `expected_operation_configs=10`, `matched_operation_configs=10`, `schema_configs_compared=0`, all 10 configuration comparisons exact with `differing_fields=[]`, and no raw-feature differences. Cross-attempt aggregate differences are restricted to observational p50/p95/RSS.

A separate Actions-only promotion analysis run **35966255656** at commit `283ab6aaef2e0e9bede33107d3f10ee0a7934d05` aggregated the verified seed/family/runtime evidence without retraining. Analysis artifact **10794162808**, SHA-256 `ce7a2435ce6a53809b663a67d77723a3d58cd493b5ad9ee1a43c137a9bcdbe16`. `Ticket` / `Compendium` becomes regression-only with this inspection.

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
| **GDM62 operation-only** | source `a3cfa70a570f931e2c78553ca9d72aeb5ed65e79`, run `35953082798` | **`35965921793`** (bounded repair of diagnostic audit `35961242199`) |

## Evidence and promotion policy

A post-suspension canonical promotion requires complete declared **operation** results, changed learned-head checkpoint evidence, optimizer histories, validation-only checkpoint/calibration selection, per-example predictions/generated GraphQL, real parse/validation/execution checks, raw timing/runtime metadata, an exact same-source/same-seed rerun, and a dedicated independent operation-only exact-attempt audit.

The root README must be synchronized **before** announcing promotion or launching the next numbered experiment. Promotion commits must record the exact measured source, source attempts, independently rebuilt operation BATCH_REPORT artifact IDs/digests, passing audit artifact ID/digest, denominators, regression/holdout limits, p50/p95, memory scope, supported and negative learnings, and the next unverified operation hypothesis.

Historical schema metrics and artifacts are retained for provenance but are frozen. They are not part of new promotion gates.

## Current limits and next decision gate

The measured system remains a small synthetic, schema-grounded research environment. It has not established unrestricted operation token generation, general mutation/subscription generation, arbitrary enterprise topology handling, learned clause decomposition, transformer fine-tuning benefits, or production-scale latency/memory behavior.

GDM62 rules out another narrow family of hand-designed ambiguity representations: **fixed rank-1 candidate-to-candidate product/delta relations do not improve the retained request-relative control**. The strongest remaining fresh error signal is role discrimination, while object-vs-supplier is already saturated and lifecycle/representation do not improve under the tested transforms. Three-clause exact operation generation also remains poor.

The next unverified hypothesis is therefore deliberately different: **a small learned rank-preserving cross-candidate interaction may extract useful contrast among plausible candidates that fixed algebraic `c1↔ci` relations cannot.** Any GDM63 design must remain operation-only, keep the frozen encoder/numerical/calibration/GraphQL contracts fixed, preserve matched bounded capacity across arms, and test whether learned candidate-set interaction improves role/representation ambiguity and multi-clause operation correctness without worsening unsafe publication.

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
| [`experiments/followup62/`](experiments/followup62/) | Candidate-relational operation experiment; latest canonical evidence |
| [`.github/workflows/`](.github/workflows/) | Actions-only measured batches, audits, and evidence analyses |

Use each experiment's declared workflow/execution wrapper when reproducing it. Calling historical Python entrypoints directly can bypass the canonical numerical contract.
