# Graph Model

**A GraphQL Decision Model (GDM): learn the decisions that satisfy a request, then construct and evaluate real GraphQL.**

This repository researches models that reason about GraphQL and generate **operations and schemas**. Query planning is out of scope.

The current measured system is deliberately narrower than a general-purpose GraphQL generator: it uses a **frozen semantic encoder plus learned capability and ambiguity heads**, then deterministically realizes selected capabilities as executable GraphQL operations or catalog-grounded Federation schemas. It does not autoregressively generate arbitrary GraphQL tokens, invent arbitrary ontologies, or fine-tune the transformer backbone.

## Current research state

_Last synchronized: 2026-09-22. Latest canonical experiment: GDM56._

| State | Meaning |
| --- | --- |
| **Canonical through GDM56** | GDM50–GDM56 each have the required repeated measured evidence and exact-attempt reproducibility audit. Canonical means the evidence is usable and reproducible under its declared contract; it does **not** mean every tested arm is a winning architecture. |
| **Next hypothesis: richer ambiguity evidence, unpromoted** | GDM56 shows that balancing relation-family exposure can stabilize some operation ambiguity recovery, but representation and object-vs-supplier ambiguity remain almost entirely unsolved and schema ambiguity stays weak. The next bounded question is whether the ambiguity head needs richer candidate-pair/top-k semantic evidence rather than more curriculum manipulation. No result is claimed for that hypothesis yet. |
| **Invalid historical evidence** | Original GDM41–44 jobs wrote manifests and fabricated/simulated metrics rather than measured model results. Their promotions are withdrawn. Only the measured repair run `35565425124` (196/196 verified) is the valid GDM41–44 baseline. |

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

Each example contains a task (`operation` or `schema`), a natural-language request, and a typed catalog of available capabilities. A capability is a **full GraphQL path** with coordinates, type signatures, and a public semantic description. Paths such as `reviews.author.name`, `reviews.author.id`, and `reviews.moderator.name` are different response contracts even though some share terminal field names.

The current benchmark is synthetic and catalog-bounded. Fresh domain names and fresh paraphrases are useful screening controls, but they are not evidence of enterprise out-of-distribution generalization or arbitrary graph-topology generation.

### Clause decomposition

Multi-capability requests are decomposed using public request syntax into clause-local decisions. This is deterministic benchmark parsing, not a learned natural-language decomposition model. The research line keeps clause-local scoring because GDM45 showed whole-request capability scoring collapses particularly badly as requested capability count grows.

### Frozen semantic encoder

The active canonical line uses `distilbert/distilbert-base-uncased` at the exact revision recorded in every result. The backbone is frozen. Mean-pooled normalized representations cross an explicit **`1e-4` canonical feature boundary** before learned-head training. Request clauses are encoded fresh during timed generation; catalog and NONE vectors are cached.

The canonical CPU/numerical contract fixes one-thread/default CPU dispatch, disables oneDNN, uses PyTorch 2.8 single-tensor AdamW with `foreach=false` and `fused=false`, non-foreach gradient clipping, float32 optimizer moments/state, and evaluates the final decoupled-weight-decay plus bias-corrected parameter application in float64 before storing float32 parameters. This is a reproducibility architecture contract, not transformer fine-tuning.

### Learned capability head and explicit NONE

For a request-clause vector `q` and candidate vector `c`, the capability scorer consumes the request representation, candidate representation, elementwise interaction, absolute difference, catalog-relative candidate information, and structural GraphQL features. A residual feature adapter and MLP score all real capabilities plus an explicit **NONE** outcome.

Unsupported requests can therefore train toward NONE rather than being forced onto a real field. These are learned feature-space heads over frozen representations. Separate operation/schema checkpoints do not imply separately fine-tuned transformer backbones.

### Learned ambiguity head

The canonical GDM51+ architecture adds a separate clause-local ambiguity discriminator after capability training. It currently sees **eleven summary features** derived from top-two candidate evidence: NONE-versus-best and best-versus-second margins, probabilities, encoder similarities, plus structural signals such as shared parent, depth, and leaf-kind similarity.

That restriction is now important. GDM55 and GDM56 demonstrate that changing curriculum can move ambiguity behavior, but several ambiguity families remain unresolved. The ambiguity head still does **not** see the full request embedding, the full candidate set, or an explicit learned relation between arbitrary candidate pairs.

### Validation-only arbitration

The capability and ambiguity heads are optimizer-trained. Thresholds and request arbitration are not. Validation-only calibration chooses NONE and ambiguity thresholds; the current canonical line keeps the GDM54 high-confidence ambiguity-rescue policy with a 5 percentage-point NO_MATCH-recall budget and an accepted-status recall guardrail.

Request aggregation is deterministic: a remaining NO_MATCH clause rejects the request; otherwise any AMBIGUOUS clause makes the request ambiguous; otherwise selected paths are emitted.

### Deterministic GraphQL realization

Accepted operation capabilities are converted to a closed selection tree, parsed and validated with `graphql-core`, and executed against changing fixtures designed to distinguish author/moderator, name/ID, and creation/update mistakes.

Accepted schema capabilities are projected into a minimal field inventory with deterministic object/key closure and subgraph ownership conventions. Rover composes the actual generated subgraphs, and a separate requirements evaluator checks whether the requested capabilities are present. **Composition success is not request correctness.** Schema generation in this research line is catalog projection plus deterministic SDL/Federation realization, not unconstrained schema invention.

## What the experiment line has taught us

| Experiment | Measured learning | Consequence / limit |
| --- | --- | --- |
| GDM45 | Clause-local decomposition improves multi-capability generation relative to whole-request scoring. | Keep clause-local decisions. Its schema risk/status path had a shortcut, so only accepted-case evidence is retained. |
| GDM46–49 | Strong rejection gates can improve risk handling while destroying answerable recall. | Risk correctness and answerable correctness must be measured separately. |
| GDM50 | Explicit NONE gives the model an open-set unsupported outcome, but does not solve ambiguity or multi-clause compounding. Numerical diagnostics also exposed cross-run optimizer drift. | Keep explicit NONE and the hardened numerical evidence contract. |
| GDM51 | A separate ambiguity discriminator improves the capability-vs-risk decomposition. | Ambiguity deserves a learned component, but its current representation is narrow. |
| GDM52–54 | Calibration and arbitration expose precision/recall tradeoffs; ambiguity-first arbitration alone loses too many correct NO_MATCH decisions. | Do not expect threshold ordering to repair weak ambiguity evidence. |
| GDM55 | Relation-family expansion recovers additional ambiguity beyond volume-matched optimizer exposure, but gains are task/seed dependent. A bundled full curriculum can increase answerable recall while worsening ambiguity/publication. | Semantic diversity matters, but “more curriculum” is not a monotonic improvement. |
| **GDM56** | **Balancing relation-family exposure materially improves operation ambiguity stability and publication safety, but benefits do not transfer uniformly to schema. Counterfactual-negative arms collapse held-out ambiguity to zero. Representation and object-vs-supplier ambiguity remain essentially unsolved.** | **The next bounded bottleneck is the ambiguity representation itself, not another indiscriminate curriculum expansion.** |

# Latest canonical result: GDM56

GDM56 changes **only ambiguity-head relation-family sampling and factorization**. It keeps the learned architecture, capability objective, explicit NONE, frozen DistilBERT encoder, `1e-4` feature boundary, numerical path, deterministic GraphQL realization, and GDM54 rescue arbitration fixed.

The five arms are:

- `relation-expanded-control`: canonical GDM55 relation-expanded curriculum;
- `family-balanced-rescue`: equalized positive exposure across four ambiguity families;
- `family-balanced-counterfactual`: balanced positives plus exactly 25% counterfactual negatives;
- `family-balanced-domain`: balanced positives plus training-only domain randomization;
- `family-balanced-counterfactual-domain`: both isolated factors together.

Every arm is matched to the same positive/negative curriculum counts, so the comparison separates **which semantic examples are shown** from simply performing more optimizer updates.

## GDM56 data and denominators

`Ledger` / `Session` are calibration-only. `Agreement` / `Device` are the fresh GDM56 holdout and are now regression-only for future experiments. Training-only domain randomization uses `Beacon`, `Parcel`, `Memo`, and `Roster`.

Each task has **84 unique holdout cases**: 36 answerable, 32 NO_MATCH, and 16 AMBIGUOUS. Rates are means over seeds `5601,5602`. Status counts below pool the two seeds, so `12/32` means sixteen unique ambiguous cases evaluated by two independently initialized models—not thirty-two independent semantic worlds. The exact reproducibility rerun does not increase sample size.

## Operation generation

| Curriculum | Answerable accuracy | Risk accuracy | Incorrect publication | Target precision | Target recall | AMBIGUOUS | NO_MATCH |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Relation-expanded control | 59.72% | 41.67% | 26.19% | 97.44% | 56.25% | 4/32 | 36/64 |
| **Family-balanced rescue** | **63.89%** | **48.96%** | **21.43%** | **100.00%** | **58.59%** | **12/32** | 35/64 |
| Balanced + counterfactual | 59.72% | 38.54% | 32.74% | 92.00% | **60.94%** | 0/32 | **37/64** |
| Balanced + domain | 59.72% | **48.96%** | 22.02% | **100.00%** | 53.91% | **12/32** | 35/64 |
| Balanced + counterfactual + domain | 59.72% | 38.54% | 32.74% | 92.00% | **60.94%** | 0/32 | **37/64** |

Balancing positive relation-family exposure is a meaningful operation-generation improvement under this holdout: compared with the relation-expanded control it raises exact answerable accuracy by **4.17pp**, risk accuracy by **7.29pp**, and ambiguity recovery from **4/32 to 12/32**, while reducing incorrect publication by **4.76pp**. NO_MATCH changes only from 36/64 to 35/64.

The improvement is also more seed-stable. Relation control gets **0/16** ambiguous cases at seed 5601 and **4/16** at seed 5602. Family-balanced rescue gets **6/16 at both seeds**.

Counterfactual negatives are strongly negative evidence in this implementation: both counterfactual arms fall to **0/32 AMBIGUOUS** despite matched optimizer exposure. They increase answerable-case target recall by accepting more, but precision and publication safety deteriorate. Adding training-only domain randomization does not rescue that collapse.

For exact answerable requests, family-balanced rescue gets **24/32 one-clause, 15/24 two-clause, and 7/16 three-clause** correct. Multi-clause compounding remains material even though this holdout is easier for answerable generation than GDM55’s holdout.

## Schema generation

| Curriculum | Answerable accuracy | Risk accuracy | Incorrect publication | Target precision | Target recall | AMBIGUOUS | NO_MATCH |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Relation-expanded control | **69.44%** | 21.88% | 41.07% | 95.18% | **71.09%** | 3/32 | 18/64 |
| Family-balanced rescue | 61.11% | **34.38%** | **29.76%** | **98.75%** | 57.03% | 3/32 | **30/64** |
| Balanced + counterfactual | 61.11% | 19.79% | 49.40% | 90.56% | 68.75% | 0/32 | 19/64 |
| Balanced + domain | **69.44%** | 23.96% | 41.67% | 95.18% | **71.09%** | 3/32 | 20/64 |
| Balanced + counterfactual + domain | 63.89% | 31.25% | 37.50% | 95.69% | 67.97% | 0/32 | **30/64** |

Schema behavior is different. Family balancing does **not** increase AMBIGUOUS recovery over relation control: both remain at 3/32. Instead, balanced rescue becomes more conservative, improving NO_MATCH from 18/64 to 30/64 and reducing incorrect publication by **11.31pp**, but losing **8.33pp** answerable accuracy and **14.06pp** target recall.

Training-only domain randomization largely restores the relation-control answerable behavior but also restores its publication problem. Under counterfactual training, adding domain randomization improves NO_MATCH from 19/64 to 30/64 and lowers incorrect publication, but AMBIGUOUS remains **0/32**. The two factors therefore interact, but neither solves schema ambiguity.

Family-balanced rescue gets **23/32 one-clause, 14/24 two-clause, and 7/16 three-clause** answerable requests exactly correct. Relation control and balanced-domain reach 25/32, 16/24, and 9/16.

## Ambiguity-family behavior and stability

Counts pool two seeds; each family has eight decisions total (four unique cases per seed).

| Task / arm | Role | Lifecycle time | Representation | Object vs supplier |
| --- | ---: | ---: | ---: | ---: |
| Operation / relation control | 4/8 | 0/8 | 0/8 | 0/8 |
| Operation / family-balanced rescue | **8/8** | **4/8** | 0/8 | 0/8 |
| Operation / balanced + domain | 7/8 | **4/8** | 1/8 | 0/8 |
| Operation / either counterfactual arm | 0/8 | 0/8 | 0/8 | 0/8 |
| Schema / relation control | 2/8 | 1/8 | 0/8 | 0/8 |
| Schema / family-balanced rescue | 2/8 | 1/8 | 0/8 | 0/8 |
| Schema / balanced + domain | 2/8 | 1/8 | 0/8 | 0/8 |
| Schema / either counterfactual arm | 0/8 | 0/8 | 0/8 | 0/8 |

This is the most important structural learning from GDM56. Curriculum balancing can teach the current 11-feature ambiguity head to recognize **role** and some **lifecycle-time** ambiguity for operations. It does not reliably teach **representation (name vs ID)** or **object-vs-supplier** ambiguity, and schema-family behavior barely moves. More training examples are therefore not sufficient evidence that the current ambiguity representation can express the missing distinctions.

Seed behavior reinforces that conclusion. Operation balanced rescue is stable at 6/16 ambiguous cases for both seeds, while schema relation/balanced variants are 3/16 at seed 5601 and 0/16 at seed 5602. The remaining schema ambiguity signal is weak and initialization-sensitive.

## Semantic errors and regression

The operation family-balanced rescue arm records no accepted semantic-confusion pairs in the fresh holdout. The relation control still confuses an Agreement author name with author ID twice. Counterfactual operation arms introduce larger errors, including Device title requests mapped to moderator-name paths.

Schema relation control and balanced-domain both include lifecycle/title and author/moderator confusions. Balanced rescue reduces the recorded fresh-holdout confusion set to one author-vs-moderator mistake, consistent with its more conservative publication behavior rather than better answerable coverage.

Previously inspected regression-suite request accuracy changes only modestly across arms: operation ranges **44.88%–46.70%** and schema **46.70%–48.38%**. These suites are regression material, not fresh evidence and not a cross-batch leaderboard.

## Speed and memory

Attempt 1 mean-per-seed warm-generation measurements are observational:

| Task | p50 range across arms | p95 range across arms | mean RSS-after-evaluation range |
| --- | ---: | ---: | ---: |
| Operation | 62.30–63.68 ms | 112.37–114.96 ms | 655,716–671,998 KiB |
| Schema | 75.01–76.39 ms | 137.29–139.98 ms | 661,840–676,672 KiB |

The scope is fresh request-clause encoding, learned-head scoring, deterministic GraphQL realization, and local validation with catalog/NONE vectors cached. Downloads, initial catalog embedding, backend fixture execution, and Rover composition are excluded from these generation quantiles; composition is recorded separately. RSS is process-level and affected by worker reuse.

The exact audit records materially different hosted-runner timing and RSS on attempt 2 even though deterministic model state and outputs match exactly. Timing and memory are therefore evidence about this execution environment, not bit-reproducible model outputs or a service SLA.

## GDM56 reproducibility and provenance

GDM56’s repaired measured source is commit [`168317f6495b05e04597ae084b75be6890fe0621`](https://github.com/burn2delete/graph-model/commit/168317f6495b05e04597ae084b75be6890fe0621), run [`35774902004`](https://github.com/burn2delete/graph-model/actions/runs/35774902004), attempts 1 and 2. Each attempt independently verified **20/20** declared results with no missing, failed, or unexpected configurations.

- Attempt 1 `BATCH_REPORT` artifact **10715918812**, ZIP SHA-256 `12d73d1cbc88ccb091b6228edb1b526b5e55daf63951d788665057190873339c`.
- Attempt 2 `BATCH_REPORT` artifact **10719305537**, ZIP SHA-256 `103c45141553d902c06e976e9650b9949dab7c82187620a3531cd7661a145ef4`.
- Passing exact-attempt audit run [`35798481278`](https://github.com/burn2delete/graph-model/actions/runs/35798481278), audit workflow commit [`1438c177dbb51da6791ab3f54940b99f5c4807ff`](https://github.com/burn2delete/graph-model/commit/1438c177dbb51da6791ab3f54940b99f5c4807ff).
- Passing audit artifact **10724274848**, ZIP SHA-256 `f12e03e5e467df4b83f35207246b2b83e723d80ddecfaef5d0bbea11ac803a21`.

The audit report is `gdm56-reproducibility-audit-v1` with `complete=true`, `evidence_verified=true`, `reproducible=true`, `promotion_eligible=true`, `errors=[]`, and exact **operation 10/10 + schema 10/10** matching. All 20 comparisons have `exact_match=true`, `differing_fields=[]`, and no raw-feature differences. It independently reconstructs both batch reports and verifies exact provenance, selected full/capability/ambiguity state hashes, selected epochs, optimizer counts, complete training/curriculum records, calibration documents, deterministic regression/holdout/family metrics, predictions, and canonical fixed-probe/full-corpus feature hashes.

The first two audit runs are retained as audit-implementation diagnostics, not model failures: run `35787993532` failed because directory symlinks were not traversed by collector discovery; run `35793692057` rebuilt a complete 20/20 batch and then failed on malformed audit summary-discovery code. Neither changed source experiment evidence or justified retraining.

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
| **GDM56** | source `168317f6495b05e04597ae084b75be6890fe0621`, run `35774902004` | **`35798481278`** |

## Evidence and promotion policy

All accepted compilation, tests, training, validation, benchmarks, and independent evidence checks run in **GitHub Actions**. Local/source inspection may analyze already-produced evidence but does not replace an Actions verifier.

A canonical promotion requires complete declared results, changed learned-head checkpoint evidence, optimizer histories, validation-only checkpoint/calibration selection, per-example predictions and generated GraphQL, real execution/composition checks, raw timing/runtime metadata, an exact same-source/same-seed rerun, and a dedicated independent exact-attempt audit. The root README must be synchronized **before** announcing the promotion or launching the next numbered experiment.

Fresh holdouts never select optimizer behavior, checkpoints, thresholds, or curriculum changes. Once inspected, they become regression-only. Repeated seeds and exact reruns are reproducibility evidence, not extra semantic worlds.

## Current limits and next bounded question

The measured system is still a small synthetic catalog-projection research environment. It has not established unrestricted schema invention, general mutation/subscription generation, arbitrary enterprise topology handling, transformer fine-tuning benefits, or production-scale latency/memory behavior.

The strongest unresolved signal after GDM56 is now specific: **the current ambiguity head can learn role and some lifecycle ambiguity from balanced supervision, but representation and object-vs-supplier ambiguity remain near zero and schema ambiguity remains weak.** Counterfactual-negative curriculum does not repair this; in the tested construction it makes ambiguity detection worse.

The next unverified hypothesis is therefore a bounded **ambiguity-representation** experiment: hold the capability head, curriculum volume, numerical path, feature quantum, calibration/arbitration, and deterministic GraphQL realization fixed while testing whether richer candidate-pair or top-k semantic evidence allows the ambiguity head to represent the missing relation distinctions. This is a hypothesis, not yet a result.

## Repository map

| Location | Purpose |
| --- | --- |
| [`experiments/MEASUREMENT_POLICY.md`](experiments/MEASUREMENT_POLICY.md) | Execution, evidence, scope, and mandatory README-promotion rules |
| [`experiments/measured/`](experiments/measured/) | Repaired baseline, frozen encoders, shared heads, GraphQL contracts/evaluation |
| [`experiments/followup50/`](experiments/followup50/) | Explicit NONE, cache boundary, canonical numerical execution |
| [`experiments/followup51/`](experiments/followup51/) | Capability/ambiguity staged training and structured ambiguity features |
| [`experiments/followup52/`](experiments/followup52/), [`followup53`](experiments/followup53/), [`followup54`](experiments/followup54/) | Calibration and arbitration experiments |
| [`experiments/followup55/`](experiments/followup55/) | Ambiguity curriculum generalization |
| [`experiments/followup56/`](experiments/followup56/) | Relation-family balance/factorization, latest canonical experiment |
| [`.github/workflows/`](.github/workflows/) | Actions-only measured batches and exact-attempt audits |

Use each experiment’s declared workflow/execution wrapper when reproducing it. Calling historical Python entrypoints directly can bypass the canonical numerical contract.