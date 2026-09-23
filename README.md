# Graph Model

**A GraphQL Decision Model (GDM): learn the decisions that satisfy a request, then construct and evaluate real GraphQL.**

This repository researches models that reason about GraphQL and generate **operations and schemas**. Query planning is out of scope.

The current measured system is deliberately narrower than a general-purpose GraphQL generator: it uses a **frozen semantic encoder plus learned capability and ambiguity heads**, then deterministically realizes selected capabilities as executable GraphQL operations or catalog-grounded Federation schemas. It does not autoregressively generate arbitrary GraphQL tokens, invent arbitrary ontologies, or fine-tune the transformer backbone.

## Current research state

_Last synchronized: 2026-09-23. Latest canonical experiment: GDM57._

| State | Meaning |
| --- | --- |
| **Canonical through GDM57** | GDM50–GDM57 each have the required repeated measured evidence and exact-attempt reproducibility audit. Canonical means the evidence is usable and reproducible under its declared contract; it does **not** mean every tested arm is a winning architecture or that one GDM57 representation is now the preferred production configuration. |
| **Next hypothesis: broader candidate evidence, unpromoted** | GDM57 establishes that request-conditioned top-two semantics contain substantial ambiguity signal, but the gains trade against NO_MATCH/publication safety and multi-clause correctness remains weak. The next bounded question is whether broader top-k semantic evidence can preserve ambiguity discrimination without forcing the head to infer request risk from only the top two candidates. No result is claimed for that hypothesis yet. |
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
       Top-two real candidates + NONE evidence
                    |
          Learned clause-local ambiguity head
       scalar evidence + optional matched-capacity
       top-two candidate/request semantic blocks
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

Each example contains a task (`operation` or `schema`), a natural-language request, and a typed catalog of available capabilities. A capability is a **full GraphQL path** with coordinates, type signatures, and a public semantic description. Paths such as `reviews.author.name`, `reviews.author.id`, and `reviews.moderator.name` are different response contracts even when they share terminal field names.

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

GDM51 introduced a separate clause-local ambiguity discriminator after capability training. Before GDM57, the canonical ambiguity representation consisted of **eleven scalar summary features** derived from top-two candidate evidence: NONE-versus-best and best-versus-second margins, probabilities, encoder similarities, and structural signals such as shared parent, depth, and leaf-kind similarity.

GDM57 canonically evaluates a richer, matched-capacity input family. Every arm has the same ambiguity-head input dimension and trainable parameter count (`11 + 10*d`, or 7,691 inputs for DistilBERT); unused semantic blocks are zero-filled in control arms. The tested blocks are ordered top-two candidate semantics, request/candidate interaction semantics, their combination, and an order-invariant candidate representation. **GDM57 still uses only the top two real candidates.** It does not establish broader top-k or full-catalog learned reasoning.

Canonical evidence therefore establishes that richer semantic input is useful information, not that a particular GDM57 arm is automatically the production default.

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
| **GDM57** | **Request-conditioned top-two semantic features recover far more AMBIGUOUS cases than the matched-capacity scalar control, especially for schemas. Combined candidate+request representations provide a better schema risk balance than request-only semantics. Candidate semantics alone are insufficient for schema ambiguity.** | **The semantic representation contains useful risk signal, but current top-two evidence still creates a strong AMBIGUOUS-versus-NO_MATCH/publication tradeoff and does not solve multi-clause compounding. Broader candidate evidence remains unmeasured.** |

# Latest canonical result: GDM57

GDM57 changes **only the semantic input representation of the learned ambiguity head**. Capability training, explicit NONE, the exact GDM56 family-balanced ambiguity curriculum, frozen DistilBERT, the `1e-4` feature boundary, canonical numerical execution, GDM54 rescue calibration/arbitration, and deterministic GraphQL/Federation realization remain fixed.

The five matched-capacity arms are:

- `scalar-control`: canonical eleven scalar features plus zero-filled semantic blocks;
- `candidate-directed`: ordered top-two candidate semantic features;
- `request-conditioned`: request/candidate interaction features;
- `candidate-request`: ordered candidate semantics plus request-conditioned features;
- `symmetric-request`: order-invariant candidate semantics plus request-conditioned features.

All arms use the same 7,691-input ambiguity head for DistilBERT. A better result therefore cannot be attributed merely to a larger ambiguity MLP.

## GDM57 data and denominators

`Credential` / `Snapshot` are calibration-only. `Reservation` / `Artifact` are the fresh GDM57 secondary holdout and are now **regression-only** for subsequent experiments. GDM46–GDM56 inspected holdouts also remain regression-only.

Each task has **84 unique holdout cases**: 36 answerable, 32 NO_MATCH, and 16 AMBIGUOUS. Rates are means over seeds `5701,5702`. Status counts pool those two seeds, so `25/32` means sixteen unique ambiguous cases evaluated by two independently initialized models—not thirty-two independent semantic worlds. The exact reproducibility rerun does not increase the semantic sample size.

The independent collector verified the four declared ambiguity families—role, lifecycle-time, representation/name-vs-ID, and object-vs-supplier—from per-example holdout predictions for every measured result. The batch-level aggregate does not expose pooled per-family counts, so this promotion does **not** claim that previously weak representation or object-vs-supplier families are solved merely from the aggregate status totals.

## Operation generation

| Ambiguity representation | Answerable accuracy | Risk accuracy | Incorrect publication | Target precision | Target recall | AMBIGUOUS | NO_MATCH |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Scalar control | 25.00% | 39.58% | **22.62%** | **84.93%** | 27.34% | 12/32 | 26/64 |
| Candidate-directed | 25.00% | 45.83% | 26.79% | 83.48% | 25.00% | 17/32 | **27/64** |
| Request-conditioned | **34.72%** | **46.88%** | 44.05% | 82.64% | **49.22%** | **25/32** | 20/64 |
| Candidate + request | **34.72%** | 42.71% | 42.86% | 82.64% | **49.22%** | 21/32 | 20/64 |
| Symmetric candidate + request | **34.72%** | 42.71% | 46.43% | 82.64% | **49.22%** | 21/32 | 20/64 |

Request-conditioned semantics produce the strongest aggregate AMBIGUOUS recovery for operations, rising from **12/32 to 25/32**, while answerable accuracy increases by 9.72 percentage points and target recall by 21.88 points. That is not a free improvement: NO_MATCH falls from 26/64 to 20/64 and incorrect publication nearly doubles from 22.62% to 44.05%.

Candidate-directed semantics are the more conservative operation result: they raise AMBIGUOUS from 12/32 to 17/32 and risk accuracy from 39.58% to 45.83% while keeping NO_MATCH essentially flat (27/64 versus 26/64), but answerable accuracy does not improve. This supports the narrower claim that candidate identity contains useful risk information without establishing it as a complete solution.

Multi-clause exact correctness remains poor. The scalar and candidate-directed arms get 12/32 one-clause, 6/24 two-clause, and **0/16 three-clause** answerable requests correct. The three request-conditioned variants improve to 17/32 and 8/24 but still get **0/16 three-clause** requests exactly correct.

Accepted semantic-confusion evidence also remains material. Request-conditioned/combined operation arms include supplier-versus-title, lifecycle-time, author-versus-moderator, and rating-versus-update mistakes. Richer ambiguity evidence does not repair capability-selection errors after a request is accepted.

## Schema generation

| Ambiguity representation | Answerable accuracy | Risk accuracy | Incorrect publication | Target precision | Target recall | AMBIGUOUS | NO_MATCH |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Scalar control | 25.00% | 47.92% | **18.45%** | 88.31% | 25.00% | 5/32 | **41/64** |
| Candidate-directed | 31.94% | 42.71% | 27.98% | **91.45%** | 33.59% | 0/32 | **41/64** |
| Request-conditioned | **47.22%** | 46.88% | 44.05% | 83.79% | **60.94%** | **27/32** | 18/64 |
| Candidate + request | 34.72% | **60.42%** | 27.98% | 84.86% | 42.19% | 24/32 | 34/64 |
| Symmetric candidate + request | 34.72% | 57.29% | 32.74% | 83.88% | 44.53% | 21/32 | 34/64 |

Schema results make the representation effect clearest. Request-conditioned semantics raise AMBIGUOUS from **5/32 to 27/32**, answerable accuracy from 25.00% to 47.22%, and target recall from 25.00% to 60.94%. But they reduce NO_MATCH from 41/64 to 18/64 and increase incorrect publication from 18.45% to 44.05%.

The `candidate-request` arm gives the best measured schema risk balance in this batch: **60.42% risk accuracy**, 24/32 AMBIGUOUS, and 34/64 NO_MATCH, while incorrect publication is 27.98%. It does not match request-only answerable recall, so no single arm dominates the full metric set.

Candidate-directed semantics alone are negative evidence for schema ambiguity: AMBIGUOUS falls to **0/32** even though NO_MATCH stays at 41/64. Candidate identity without request-conditioned interaction is therefore insufficient under this training/calibration contract.

For exact answerable requests, request-conditioned schema gets **20/32 one-clause, 11/24 two-clause, and 3/16 three-clause** correct. Candidate-request and symmetric-request get 17/32, 8/24, and 0/16. Multi-clause compounding remains a major limitation even in the strongest answerable arm.

## Ambiguity-family, semantic-error, and regression limits

The source collector independently recomputes the four ambiguity-family buckets from `predictions-holdout.jsonl` and requires exact equality with each result’s recorded family metrics. The exact-attempt audit then requires those deterministic family metrics to match across both source attempts. This establishes reproducibility of the family evidence, but the top-level batch report does not pool those values across seeds/arms; the README therefore avoids inventing family counts that are not present in the canonical aggregate artifact.

Family-relevant accepted errors remain visible in the recorded confusion sets. Schema request-conditioned and combined arms still publish object-versus-supplier/title mistakes, role mistakes (author versus moderator), and lifecycle-time mistakes. No conclusion that representation/name-vs-ID is solved is supported by the aggregate report. These residual errors are one reason broader candidate context remains a bounded next hypothesis.

Previously inspected regression-suite request accuracy spans roughly **38.80%–49.27% for operation** and **43.28%–49.76% for schema** across GDM57 arms. These suites are regression material, not fresh evidence and not a cross-batch progress leaderboard.

## Speed and memory

Attempt 1 mean-per-seed warm-generation measurements are observational:

| Task | p50 range across arms | p95 range across arms | mean RSS-after-evaluation range |
| --- | ---: | ---: | ---: |
| Operation | 64.41–66.16 ms | 125.84–129.53 ms | 1,072,238–1,077,300 KiB |
| Schema | 62.54–64.61 ms | 121.93–125.57 ms | 1,073,946–1,077,344 KiB |

The generation scope is fresh request-clause encoding, learned-head scoring, deterministic GraphQL realization, and local validation with catalog/NONE vectors cached. Downloads, initial catalog embedding, backend fixture execution, and Rover composition are excluded from these generation quantiles; composition is recorded separately. RSS is process-level and affected by worker reuse. The larger process RSS than GDM56 is therefore **not** an isolated model-memory attribution, although GDM57’s matched 7,691-input ambiguity head is materially larger than the previous eleven-feature head.

Timing and memory are observational and need not bit-match across hosted-runner attempts; deterministic model state and predictions do.

## GDM57 reproducibility and provenance

GDM57’s measured source is commit [`3f8b9a968285110ca6c744590724ddd0cb2bd5aa`](https://github.com/burn2delete/graph-model/commit/3f8b9a968285110ca6c744590724ddd0cb2bd5aa), source run [`35803609103`](https://github.com/burn2delete/graph-model/actions/runs/35803609103), attempts 1 and 2. Each attempt independently verified **20/20** declared results with no missing, failed, or unexpected configurations. Both preflights passed the GDM57/current inherited contracts and deterministic symmetric-request hash smoke; the hash smoke is contract evidence, not pretrained-model performance.

- Attempt 1 `BATCH_REPORT` artifact **10727032548**, ZIP SHA-256 `d46fbc4ad35dfa8e27b6702a02b930727e44734e277a950f279e33f3c8a38d82`.
- Attempt 2 `BATCH_REPORT` artifact **10728527353**, ZIP SHA-256 `3c180f3bba9e17f52526fd4ed734cd9eee1663a22ae3137ea6d246874bb0054d`.
- Passing exact-attempt audit run [`35811475866`](https://github.com/burn2delete/graph-model/actions/runs/35811475866), audit workflow commit [`77f273df06d0c1dda3386b9eaca7cfd17e3998d9`](https://github.com/burn2delete/graph-model/commit/77f273df06d0c1dda3386b9eaca7cfd17e3998d9).
- Passing audit artifact **10730038016**, ZIP SHA-256 `a21dfbe5eb859228305b1c8aa3eb207ac058a9483d68c1d2db4bcbc3a279e998`.

The audit report is `gdm57-reproducibility-audit-v1` with `complete=true`, `evidence_verified=true`, `reproducible=true`, `promotion_eligible=true`, `errors=[]`, and exact **operation 10/10 + schema 10/10** matching. All 20 comparisons have `exact_match=true`, `differing_fields=[]`, and `raw_feature_differences=[]`. The audit independently reconstructs both batch reports and verifies exact attempt provenance, checkpoint state/capability/ambiguity hashes, selected epochs, optimizer counts, complete training records, calibration documents, deterministic regression/holdout/clause/family metrics, per-example predictions, and canonical fixed-probe/full-corpus hashes.

Artifact retention is finite. These IDs and digests record provenance but do not substitute for retained source evidence if future re-verification is required.

## Prior canonical result: GDM56

GDM56 changed only ambiguity-head curriculum sampling/factorization while keeping the architecture fixed. Its strongest supported result was operation `family-balanced-rescue`, which raised AMBIGUOUS recovery to 12/32 with 6/16 at each seed and reduced incorrect publication versus the relation-expanded control. Schema did not receive the same ambiguity benefit, and counterfactual-negative arms collapsed AMBIGUOUS to zero. Representation/name-vs-ID and object-vs-supplier remained the clearest unresolved families, motivating GDM57.

GDM56’s repaired measured source is `168317f6495b05e04597ae084b75be6890fe0621`, source run `35774902004` attempts 1/2, BATCH_REPORT artifacts **10715918812** (`12d73d1cbc88ccb091b6228edb1b526b5e55daf63951d788665057190873339c`) and **10719305537** (`103c45141553d902c06e976e9650b9949dab7c82187620a3531cd7661a145ef4`), passing exact audit run `35798481278`, and audit artifact **10724274848** (`f12e03e5e467df4b83f35207246b2b83e723d80ddecfaef5d0bbea11ac803a21`).

Two earlier GDM56 audit runs are retained as audit-implementation diagnostics, not model failures: `35787993532` failed because directory symlinks were not traversed by collector discovery; `35793692057` rebuilt a complete 20/20 batch and then failed on malformed audit summary-discovery code. Neither changed source experiment evidence or justified retraining.

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
| **GDM57** | source `3f8b9a968285110ca6c744590724ddd0cb2bd5aa`, run `35803609103` | **`35811475866`** |

## Evidence and promotion policy

All accepted compilation, tests, training, validation, benchmarks, and independent evidence checks run in **GitHub Actions**. Source or artifact inspection may analyze already-produced evidence but does not replace an Actions verifier.

A canonical promotion requires complete declared results, changed learned-head checkpoint evidence, optimizer histories, validation-only checkpoint/calibration selection, per-example predictions and generated GraphQL, real execution/composition checks, raw timing/runtime metadata, an exact same-source/same-seed rerun, and a dedicated independent exact-attempt audit. The root README must be synchronized **before** announcing promotion or launching the next numbered experiment.

Fresh holdouts never select optimizer behavior, checkpoints, thresholds, curriculum, or representation changes. Once inspected, they become regression-only. Repeated seeds and exact reruns are reproducibility evidence, not extra semantic worlds.

## Current limits and next bounded question

The measured system is still a small synthetic catalog-projection research environment. It has not established unrestricted schema invention, general mutation/subscription generation, arbitrary enterprise topology handling, transformer fine-tuning benefits, or production-scale latency/memory behavior.

GDM57 resolves one important uncertainty: **the previous eleven scalar ambiguity features were discarding useful semantic information.** Request-conditioned top-two semantics substantially improve ambiguity recovery, and combined candidate/request semantics can improve the schema risk balance. But the richer features also expose a strong calibration/publication tradeoff, accepted capability-selection mistakes remain, and top-two-only evidence cannot represent competitors that fall outside the top two.

The next unverified hypothesis is therefore a bounded **broader top-k semantic evidence** experiment: keep the capability objective, GDM56 family-balanced curriculum, frozen encoder, `1e-4` feature boundary, numerical path, validation-only rescue policy, and deterministic GraphQL realization fixed while varying only how many ranked candidate semantics are exposed to a matched-capacity ambiguity model. The experiment should preserve a top-two control and explicitly measure whether additional candidates improve ambiguity without sacrificing NO_MATCH/publication safety. This is a hypothesis, not yet a result.

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
| [`experiments/followup57/`](experiments/followup57/) | Top-two semantic ambiguity representation, latest canonical experiment |
| [`.github/workflows/`](.github/workflows/) | Actions-only measured batches and exact-attempt audits |

Use each experiment’s declared workflow/execution wrapper when reproducing it. Calling historical Python entrypoints directly can bypass the canonical numerical contract.