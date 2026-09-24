# Graph Model

**A GraphQL Decision Model (GDM): learn the decisions that satisfy a request, then construct and evaluate real GraphQL.**

This repository researches models that reason about GraphQL and generate **operations and schemas**. Query planning is out of scope.

The measured system is deliberately narrower than a general-purpose GraphQL generator: a **frozen semantic encoder plus learned capability and ambiguity heads** selects catalog-grounded capabilities, then deterministic code realizes those capabilities as executable GraphQL operations or Federation schemas. The transformer backbone is not fine-tuned, and schema generation does not invent arbitrary ontologies.

## Current research state

_Last synchronized: 2026-09-24. Latest canonical experiment: GDM61._

| State | Meaning |
| --- | --- |
| **Canonical through GDM61** | GDM50–GDM61 each have repeated measured evidence plus a passing exact-attempt reproducibility audit. Canonical means usable reproducible evidence under the declared contract; it does **not** mean every tested arm is a winning architecture. |
| **Latest supported result** | GDM61 is a **negative/inconclusive fixed-scaling result**. Fixed nonzero product/delta block balances do not improve the answerable/safety frontier on fresh `Voucher` / `Anthology`. The strong GDM60 operation product-only endpoint does not replicate on this fresh holdout. Schema delta-only improves safety/ambiguity versus the balanced control without improving answerable accuracy or target recall. |
| **Remaining tradeoff** | Operation product-dominant has the best GDM61 answerable/recall aggregate but loses risk/publication/AMBIGUOUS safety. Schema delta-only is the strongest safety endpoint but ties the balanced control on answerable/recall. Three-clause operation correctness is 0/16 for every arm, and role plus name-vs-ID confusions remain. Seed sensitivity is substantial, especially for schema. |
| **Next unverified hypothesis** | Fixed rescaling mostly changes optimization geometry rather than representation content. The next bounded question is whether **explicit rank-preserving candidate-to-candidate semantic contrast** adds ambiguity information that the current request-relative blocks do not expose directly. This is not yet a result. |
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

GDM51 introduced a separate clause-local ambiguity discriminator. GDM57 established useful request-conditioned semantic ambiguity signal; GDM58 showed simple pooled breadth is not enough; GDM59 showed preserving top-five rank in request-relative interactions materially improves useful signal.

GDM60 factorized the canonical GDM59 request-relative representation and showed that simple capability-confidence weighting is negative. Product interaction `q*c_i` was the strongest operation factorization on that batch, while absolute request/candidate delta `abs(q-c_i)` was the strongest schema factorization.

GDM61 keeps **top-five candidate breadth, rank preservation, and exactly the same ambiguity-head capacity** (`11 + 10*d`, or 7,691 DistilBERT inputs) in every arm. It assigns fixed semantic coordinates—five ranked product slots followed by five ranked absolute-delta slots—and changes only deterministic block scales:

- `ranked-product-only`: product `1.0`, delta `0.0`;
- `ranked-product-dominant`: product `1.0`, delta `0.5`;
- `ranked-balanced-control`: product `1.0`, delta `1.0`;
- `ranked-delta-dominant`: product `0.5`, delta `1.0`;
- `ranked-delta-only`: product `0.0`, delta `1.0`.

Product-only is coordinate-identical to the GDM60 product endpoint. GDM60's diagnostic delta-only packed delta into its first generic slot; GDM61 preserves the same delta values in the dedicated second block, so it is **semantic-content-equivalent, not coordinate-identical**. Missing ranks are zero-padded. Capability training/ranking, the GDM56 family-balanced ambiguity curriculum, GDM54 rescue calibration, numerical execution, and deterministic GraphQL realization stay fixed.

Because the ambiguity head begins with learned affine transformations, nonzero deterministic feature rescaling can often be reabsorbed by learned weights. GDM61 therefore provides evidence about the measured optimization/regularization behavior of these fixed scales, not proof that a particular numerical scale is intrinsically semantic.

### Validation-only arbitration

Capability and ambiguity heads are optimizer-trained. Thresholds and request arbitration are not. Validation-only calibration chooses NONE and ambiguity thresholds; the line retains the GDM54 high-confidence ambiguity-rescue policy with a 5 percentage-point NO_MATCH-recall budget and accepted-status recall guardrail.

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
| GDM55–56 | Semantic curriculum diversity and family balance matter, but more curriculum is not monotonic. | Curriculum alone is not the missing representation. |
| GDM57 | Request-conditioned top-two semantics sharply improve AMBIGUOUS recovery. | Useful semantic signal exists, but top-two evidence still trades AMBIGUOUS against NO_MATCH/publication safety. |
| GDM58 | Wider top-k evidence through the same pooled representation produces almost no operation benefit and no schema answerable/ambiguity improvement over top-2. | Simple pooled breadth is negative/inconclusive. |
| GDM59 | Rank-preserving top-five request interactions outperform the matched pooled control on answerable correctness/recall while retaining strong ambiguity recovery. | Rank identity helps through request-relative interaction, not raw candidate slots. |
| GDM60 | Simple capability-confidence weighting loses substantial answerable recall; product-only is strongest for operation on that batch, delta-only strongest for schema. | Confidence attenuation is negative; factorization signal is task-sensitive and limited. |
| **GDM61** | **Fixed product/delta scaling does not produce a better within-batch answerable/safety frontier. The GDM60 operation product-only endpoint does not replicate on fresh Voucher/Anthology; schema delta-only improves risk/AMBIGUOUS safety but not answerable/recall. Seed variance remains large.** | **Treat fixed block balance as negative/inconclusive. Do not choose a task architecture from GDM60/GDM61 cross-holdout differences. Move to genuinely new candidate-relational information rather than more scalar rescaling.** |

# Latest canonical result: GDM61

GDM61 changes **only fixed scaling and coordinate discipline of the two rank-preserving request-relative semantic blocks supplied to the learned ambiguity head**. Every arm sees the same top-five real candidates, uses the same 7,691-input capacity and parameter count, keeps explicit NONE and the same capability model, exact GDM56 family-balanced ambiguity curriculum, frozen DistilBERT, `1e-4` feature boundary, canonical numerical execution, GDM54 validation-only rescue arbitration, and deterministic GraphQL/Federation realization.

## GDM61 data and denominators

`Badge` / `Journal` are calibration-only. `Voucher` / `Anthology` are the fresh GDM61 secondary holdout and are now **regression-only** for subsequent work. GDM46–GDM60 inspected holdouts also remain regression-only.

Each task has **84 unique secondary-holdout cases**: 36 answerable and 48 risk cases, consisting of 32 NO_MATCH and 16 AMBIGUOUS. Rates below are means over seeds `6101,6102`. Status counts pool those seeds, so `19/32` AMBIGUOUS represents sixteen unique ambiguous cases evaluated by two independently initialized models, and `49/64` NO_MATCH represents thirty-two unique unsupported cases evaluated by two seeds. The exact same-source rerun is reproducibility evidence and does not add semantic cases.

Clause-count exact-correctness denominators pool seeds: **32 one-clause, 24 two-clause, 16 three-clause** evaluations, corresponding to 16/12/8 unique semantic cases per seed. Reproducibility attempts do not increase those denominators.

The collector computes four ambiguity families—role, lifecycle-time, representation/name-vs-ID, and object-vs-supplier—from per-example predictions. The exact-attempt audit verifies those family metrics exactly between attempts. The top-level batch report intentionally does not pool family values across seeds, so no unsupported aggregate family count is invented here. Accepted role and name-vs-ID errors remain directly visible, so no family is claimed solved.

## Operation generation

| Representation | Answerable accuracy | Risk accuracy | Incorrect publication | Target precision | Target recall | AMBIGUOUS | NO_MATCH |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Product only | 16.67% | 68.75% | 9.52% | **94.12%** | 14.84% | 12/32 | **54/64** |
| Product dominant | **26.39%** | 63.54% | 13.69% | 88.85% | **25.00%** | 12/32 | 49/64 |
| Balanced control | 20.83% | 67.71% | 9.52% | 84.12% | 17.97% | **19/32** | 46/64 |
| Delta dominant | 20.83% | 68.75% | **8.93%** | 84.12% | 17.97% | **19/32** | 47/64 |
| Delta only | 20.83% | **70.83%** | 9.52% | 84.12% | 17.97% | **19/32** | 49/64 |

`ranked-product-dominant` gives the highest operation answerable accuracy and recall, but versus the balanced control it loses **4.17 points risk accuracy**, worsens incorrect publication by **4.17 points**, and drops AMBIGUOUS recovery from **19/32 to 12/32**. It is not a better answerable/safety frontier.

The GDM60 product-only operation signal does **not** replicate on the fresh GDM61 holdout: product-only is the lowest GDM61 answerable arm at 16.67%. Because the holdout domains differ, this is a robustness warning rather than a causal cross-batch reversal.

Exact one-/two-/three-clause correctness is **13/32, 6/24, 0/16** for product-dominant; **11/32, 4/24, 0/16** for balanced, delta-dominant, and delta-only; and **9/32, 3/24, 0/16** for product-only. No arm solves three-clause operation generation.

Seed sensitivity is material. Product-only answerable correctness is 8/36 for seed 6101 versus 4/36 for seed 6102; product-dominant is 8/36 versus 11/36. The aggregate differences are therefore not stable enough to justify a preferred operation architecture.

Moderator requests are still accepted as author-name paths in multiple arms. Role ambiguity remains unresolved.

## Schema generation

| Representation | Answerable accuracy | Risk accuracy | Incorrect publication | Target precision | Target recall | AMBIGUOUS | NO_MATCH |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Product only | 22.22% | 56.25% | **12.50%** | **94.12%** | 19.53% | 8/32 | **46/64** |
| Product dominant | 27.78% | 61.46% | 22.02% | 82.58% | 27.34% | 18/32 | 41/64 |
| Balanced control | **33.33%** | 61.46% | 24.40% | 83.40% | **37.50%** | 20/32 | 39/64 |
| Delta dominant | **33.33%** | 61.46% | 24.40% | 83.40% | **37.50%** | 20/32 | 39/64 |
| Delta only | **33.33%** | **63.54%** | 23.21% | 83.40% | **37.50%** | **22/32** | 39/64 |

`ranked-delta-only` is the strongest GDM61 schema safety endpoint: compared with balanced control it keeps answerable accuracy and recall unchanged, improves risk accuracy by **2.08 points**, lowers incorrect publication by **1.19 points**, and raises AMBIGUOUS recovery from **20/32 to 22/32**. It does **not** improve answerable capability, target precision, NO_MATCH, or clause correctness.

Balanced, delta-dominant, and delta-only all have exact one-/two-/three-clause correctness **14/32, 8/24, 2/16**. Product-dominant has **13/32, 6/24, 1/16** and product-only **11/32, 5/24, 0/16**. Intermediate fixed balances therefore do not improve multi-clause schema correctness.

Schema seed instability is especially large: balanced, delta-dominant, and delta-only each score **8/36 answerable** on seed 6101 versus **16/36** on seed 6102. Product-dominant moves from 8/36 to 12/36. This is a stronger limitation than the small aggregate delta-only safety gain.

Accepted schema confusions include `author.name` versus `author.id` and moderator versus author paths. Representation/name-vs-ID and role families remain unresolved.

## Regression, holdout, and interpretation limits

All inspected suites, including GDM60 `Certificate` / `Manuscript` and now GDM61 `Voucher` / `Anthology`, are **regression-only** for subsequent experiments. They are not a causal cross-batch leaderboard. GDM61 mean regression request accuracy spans approximately **43.29%–48.24% for operation** and **42.60%–47.55% for schema** across arms.

GDM61 tests fixed deterministic scaling of already-present semantic blocks. Since nonzero scale can be compensated by learned first-layer weights, the experiment should not be overinterpreted as a semantic ablation except at true zero endpoints. The fresh-holdout and seed instability further argue against choosing a product/delta mixture from these numbers.

The measured system remains synthetic and catalog-bounded. GDM61 does not test learned candidate-set attention, explicit candidate-to-candidate semantic relations, unrestricted GraphQL token generation, arbitrary schema invention, a request-global trained status model, or transformer fine-tuning.

## Speed and memory

Attempt 1 mean-per-seed warm-generation measurements are observational:

| Task | p50 range across arms | p95 range across arms | mean RSS-after-evaluation range |
| --- | ---: | ---: | ---: |
| Operation | 67.78–72.31 ms | 136.02–141.79 ms | 1,142,246–1,146,080 KiB |
| Schema | 63.87–64.14 ms | 126.53–127.64 ms | 1,139,878–1,143,852 KiB |

Generation timing covers fresh request-clause encoding, learned-head scoring, deterministic GraphQL realization, and local validation with catalog/NONE vectors cached. Downloads, initial catalog embedding, backend fixture execution, and Rover composition are excluded from these quantiles; composition is recorded separately. RSS is process-level and affected by worker reuse, so it is **not isolated model memory**. Timing/memory are observational and need not bit-match across hosted-runner attempts; deterministic state and predictions do.

## GDM61 reproducibility, diagnostic history, and provenance

The original GDM61 source commit `913a683d3d294b2475699a7f5e9b31c565779d1c`, run `35935198866` attempt 1, failed **before smoke or measured workers** because one endpoint contract test incorrectly required GDM61's dedicated second-slot delta vector to be coordinate-identical to GDM60's diagnostic first-slot packing. 75/76 tests passed; this produced **no model evidence**. The failed preflight artifact is **10782563755**, ZIP SHA-256 `301fb291d9d8d86a77eebd36531909b2dd588a13c873741bc12b93a14cae1ac1`.

Test-only repair commit [`f22fc36caf7327ed3d0d1f7dd2898ea31d837f3a`](https://github.com/burn2delete/graph-model/commit/f22fc36caf7327ed3d0d1f7dd2898ea31d837f3a) changed only the endpoint contract test. It preserved exact product-only coordinates and exact delta semantic content in GDM61's dedicated delta slot; model, data, curriculum, holdout, numerical path, and representation implementation were unchanged.

GDM61's canonical measured source is repaired commit `f22fc36caf7327ed3d0d1f7dd2898ea31d837f3a`, source run [`35939206923`](https://github.com/burn2delete/graph-model/actions/runs/35939206923), attempts 1 and 2. Each attempt independently verified **20/20** declared results with no missing, failed, or unexpected configurations.

- Attempt 1 `BATCH_REPORT` artifact **10784817334**, ZIP SHA-256 `d5ee636c17233358b353bac0189a8f05151a403adcd5afc840cf222a7eba2b4e`.
- Attempt 2 `BATCH_REPORT` artifact **10786971655**, ZIP SHA-256 `b96dd49e032a20b781b53495329b780c4509c417e5f182d694eae0f81e8ae965`.
- Passing exact-attempt audit run [`35948010054`](https://github.com/burn2delete/graph-model/actions/runs/35948010054), audit workflow commit [`d9d785da08c5866c52db8ae74f9ad597a1a4ba47`](https://github.com/burn2delete/graph-model/commit/d9d785da08c5866c52db8ae74f9ad597a1a4ba47).
- Passing audit artifact **10788215936**, ZIP SHA-256 `5b81ee234a0c4dfe889a33bf33d85e4b0640dc72a8f96d132063356861ba1d6f`.

The actual audit report is `gdm61-reproducibility-audit-v1` with `complete=true`, `evidence_verified=true`, `reproducible=true`, `promotion_eligible=true`, `errors=[]`, exact **operation 10/10 + schema 10/10**, and all 20 comparisons `exact_match=true`, `differing_fields=[]`, with `raw_feature_differences=[]`. The audit independently rebuilt both batch reports and verified attempt-specific source/job/artifact/digest windows, repaired preflight/hash-smoke evidence, actual initial/selected full/capability/ambiguity checkpoint hashes, changed learned heads, optimizer/training/calibration records, matched 7,691-input TOP5 fixed-slot representation receipts, fixed family-balanced curriculum and numerical contract, GDM54 validation-only rescue, real GraphQL execution and Rover composition contracts, deterministic regression/holdout/clause/four-family metrics, exact per-example predictions, and canonical fixed-probe/full-corpus hashes.

Artifact retention is finite. IDs and digests record provenance but do not substitute for retained source evidence if future re-verification is required.

## Prior canonical result: GDM60

GDM60 rejected simple capability-confidence attenuation and isolated product-only versus delta-only request-relative blocks. Its measured source is `c04784ec33983829f6fe55506bcb74e0f40442b2`, source run `35918183757` attempts 1/2, BATCH_REPORT artifacts **10777266084** (`9e4db92b71060fa9adac6a0edae8e1581b76bba81ffc6c67998f857c1372483f`) and **10779222122** (`b63d54094c15e039d078f2fa84c26127eea48108a1fdd967ff616e5949ab1975`), passing audit run `35929537000`, and audit artifact **10781490854** (`ed4b27f37b6c85b27359aa40d681cc36705d3ae7c3ba7e24c187e70868155f70`).

## Prior canonical result: GDM59

GDM59 established that preserving top-five candidate rank is useful primarily through request-relative semantic interactions rather than raw candidate slots. Its measured source is `035f44836d850b24f4bebe5cd499a572b6052752`, source run `35898256989` attempts 1/2, BATCH_REPORT artifacts **10769046397** (`b4a2237c074588a61031eb957ede60780b41bf5afc3b020d9c26768149ab5b20`) and **10770769769** (`c11e48f18b56d0e28f03720a9a7579957cbc97402642d591e1644789a656b312`), passing audit run `35910892558`, and audit artifact **10773722103** (`66dfea1e8bce33a06c77a80dbb287a29df2ddaba68d57aed289e51899f2b90a3`).

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

All accepted compilation, tests, training, validation, benchmarks, and independent evidence checks run in **GitHub Actions**. Source/artifact inspection may analyze already-produced evidence but does not replace an Actions verifier.

A canonical promotion requires complete declared results, changed learned-head checkpoint evidence, optimizer histories, validation-only checkpoint/calibration selection, per-example predictions/generated GraphQL, real execution/composition checks, raw timing/runtime metadata, an exact same-source/same-seed rerun, and a dedicated independent exact-attempt audit. The root README must be synchronized **before** announcing promotion or launching the next numbered experiment.

Fresh holdouts never select optimizer behavior, checkpoints, thresholds, curriculum, or representation changes. Once inspected, they become regression-only. Repeated seeds and exact reruns are reproducibility evidence, not extra semantic worlds.

## Current limits and next bounded question

The measured system remains a small synthetic catalog-projection research environment. It has not established unrestricted schema invention, general mutation/subscription generation, arbitrary enterprise topology handling, transformer fine-tuning benefits, or production-scale latency/memory behavior.

GDM61 rejects the proposed fixed-scale interpolation as a reliable next architecture. The result is also a warning against overreading GDM60's product-only operation endpoint across unlike holdouts. The remaining ambiguity errors are relational: the model confuses **one plausible candidate with another plausible candidate** (author versus moderator, name versus ID) even when request-relative features are available.

The next unverified hypothesis is therefore **explicit top-ranked candidate-to-candidate semantic contrast**. Preserve TOP5 rank, head capacity, curriculum, encoder, numerical path, calibration, and GraphQL realization, but replace one request-relative block with nonlinear top-1-to-candidate relations such as `abs(c1-ci)` or `c1*ci`, paired with either `q*ci` or `abs(q-ci)`. This adds genuinely candidate-relational information instead of another scalar rescaling of already-present features. It remains an unverified hypothesis until measured and audited.

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
| [`experiments/followup60/`](experiments/followup60/) | Request-relative factorization and confidence scaling |
| [`experiments/followup61/`](experiments/followup61/) | Fixed product/delta balance, latest canonical experiment |
| [`.github/workflows/`](.github/workflows/) | Actions-only measured batches and exact-attempt audits |

Use each experiment's declared workflow/execution wrapper when reproducing it. Calling historical Python entrypoints directly can bypass the canonical numerical contract.
