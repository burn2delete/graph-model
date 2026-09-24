# GDM65 — explicit structured schema-relation channel

Status: **prespecified before implementation and before workflow creation**.

Forward scope: **operation generation only**. Schema generation remains suspended/frozen. Query planning remains out of scope.

## Motivation

GDM62, GDM63, and GDM64 rule out three nearby hypotheses under matched bounded experiments:

1. fixed semantic candidate-to-candidate algebraic relations are not sufficient;
2. a small learned scalar gate over semantic candidate relations is not sufficient;
3. replacing canonical whole-candidate semantic slots with deterministic parent-path and leaf-field semantic embeddings is not sufficient.

The remaining persistent errors are structurally typed. The canonical line repeatedly confuses coordinates such as `reviews.author.name` vs `reviews.author.id` (same parent, different leaf/type) and `reviews.moderator.name` vs `reviews.author.id` (different parent/role plus different leaf/type). GDM64 showed that merely encoding parent/leaf strings does not make those distinctions reliably usable by the ambiguity head.

GDM65 tests a different information type: **explicit deterministic schema-relation scalars computed from the provided GraphQL candidate coordinates and type signatures**. These features describe how plausible ranked candidates relate structurally; they do not replace the canonical whole-candidate semantic representation and they never change capability ranking.

## Hypothesis

The canonical request-relative semantic slots `q*ci + abs(q-ci)` already carry useful request/candidate semantics, but the ambiguity head may benefit from exact structural facts that a frozen text encoder can blur. A small explicit relation channel may make sibling-role, sibling-leaf, and type-shape competition visible without relying on another semantic embedding recombination.

The primary hypothesis is:

> Adding a bounded, rank-preserving structural relation channel derived only from the provided schema/catalog can improve AMBIGUOUS recovery—especially role and representation/name-vs-ID families—without reducing answerable operation correctness, target recall, or publication safety relative to a matched-capacity zero-relation control.

## Frozen contract

GDM65 keeps all of the following fixed:

- operation-generation-only task;
- frozen `distilbert/distilbert-base-uncased` encoder and exact pinned revision inherited from the canonical line;
- normalized canonical encoder boundary at exactly `1e-4`;
- explicit NONE candidate;
- capability feature construction, capability training, ranking, and checkpoint selection;
- TOP5 real-candidate rank identity and zero-safe missing-rank padding;
- canonical whole-candidate ambiguity semantic slots: five `q*ci` blocks plus five `abs(q-ci)` blocks;
- the existing 11 scalar/structural ambiguity features;
- GDM56 exact family-balanced ambiguity curriculum and zero-counterfactual contract;
- GDM50 canonical numerical/optimizer path;
- GDM54 validation-only five-percentage-point ambiguity rescue and floors;
- deterministic GraphQL selection-tree realization plus real parse/validate/execute evidence;
- no transformer fine-tuning;
- no request-global trained status head;
- holdout never enters optimizer, checkpoint selection, representation selection, curriculum, or calibration.

## Structural relation source

All GDM65 relation features are deterministic functions of the **provided schema/catalog metadata for the ranked candidates**. They may use only:

- the candidate GraphQL path segments;
- candidate path depth;
- parent path (`path[:-1]` after excluding the benchmark root coordinate where the current catalog contract does so);
- leaf field name;
- canonical GraphQL output type signature already present in the capability catalog, including named type and list/non-null shape;
- TOP5 candidate rank and active/missing-rank mask.

They must not use:

- gold capability labels;
- gold request status;
- family labels;
- fresh-holdout membership/domain names;
- generated-operation correctness;
- any post-hoc error analysis field;
- any value derived from the fresh holdout during training or calibration.

String identity is exact catalog identity; no extra text encoder is introduced for the structured channel.

## Base ambiguity representation and matched capacity

Let `d` be the frozen encoder width. The canonical raw ambiguity representation remains:

- 11 existing scalar/structural features;
- five rank-preserving `q*ci` blocks;
- five rank-preserving `abs(q-ci)` blocks.

This is `11 + 10*d = 7,691` dimensions for DistilBERT.

GDM65 appends a fixed **20-scalar relation channel: four scalars per real-candidate rank × five ranks**. Therefore every GDM65 arm has exactly `31 + 10*d = 7,711` ambiguity-input dimensions and the same downstream `dim -> 32 -> 1` ambiguity MLP shape/parameter count. There is no learned relation adapter.

The matched-capacity control receives exactly twenty zeros. Thus every arm has identical trainable parameter shapes/counts; only the deterministic information in the 20 added inputs differs.

Missing ranks are all-zero in both the canonical semantic slots and the relation channel. Every non-control relation feature is bounded to `[0,1]` before entering the ambiguity MLP.

## Canonical structural primitives

For candidate rank `i` and comparison candidate `j`:

- `same_parent(i,j)`: `1` iff their parent-path segment sequences are identical, else `0`;
- `same_leaf(i,j)`: `1` iff their leaf field names are identical, else `0`;
- `same_named_type(i,j)`: `1` iff their canonical named GraphQL output types are identical, else `0`;
- `same_type_shape(i,j)`: `1` iff their full list/non-null wrapper shape is identical, else `0`;
- `lcp_ratio(i,j)`: longest common prefix segment count divided by `max(depth_i, depth_j, 1)`;
- `depth_similarity(i,j)`: `1 - min(abs(depth_i-depth_j) / max(depth_i,depth_j,1), 1)`.

For set aggregates over active `j != i`, use an arithmetic mean for Boolean/similarity primitives and zero if no other active candidate exists.

## Arms

Exactly five arms are measured, each at seeds `6501` and `6502`, for **10 operation configs**.

### 1. `structured-zero-control`

Four zeros per rank. This is the primary matched-capacity causal control. It preserves the exact canonical semantic representation while giving the ambiguity MLP the same expanded input dimension and parameter count as every relation arm.

Per-rank block:

`[0, 0, 0, 0]`

### 2. `structured-top1-relation`

Expose exact relation to the highest-ranked real candidate `c1`.

Per-rank block:

`[same_parent(i,1), same_leaf(i,1), same_named_type(i,1), lcp_ratio(i,1)]`

Rank 1 therefore receives `[1,1,1,1]`; missing ranks remain zero. This arm asks whether the ambiguity head benefits from knowing the structural kind of the nearest top-ranked competition.

### 3. `structured-set-relation`

Expose candidate-set structural density without privileging rank 1.

Per-rank block:

`[mean_j same_parent(i,j), mean_j same_leaf(i,j), mean_j same_named_type(i,j), mean_j lcp_ratio(i,j)]`, for active `j != i`.

This tests whether ambiguity is better represented as local structural crowding among plausible candidates.

### 4. `structured-type-shape-relation`

Emphasize representation/type competition against rank 1 while preserving parent/leaf distinctions.

Per-rank block:

`[same_parent(i,1), same_leaf(i,1), same_named_type(i,1), same_type_shape(i,1)]`

This is specifically motivated by persistent `name` vs `id` errors, while remaining a schema-only relation channel.

### 5. `structured-hybrid-relation`

Combine direct role/leaf relation with set-level path competition.

Per-rank block:

`[same_parent(i,1), same_leaf(i,1), mean_j lcp_ratio(i,j), mean_j depth_similarity(i,j)]`, for active `j != i` in the mean terms.

This tests whether direct sibling-role/leaf facts plus set-level topology provide complementary ambiguity evidence.

## Causal interpretation

The primary causal comparator is `structured-zero-control` within GDM65, because all five arms:

- use the same fresh holdout;
- use the same seeds;
- train the same capability and ambiguity architecture;
- have exactly the same ambiguity input dimension and trainable parameter count;
- retain the exact same canonical whole-candidate semantic slots;
- differ only in the deterministic 20-scalar structural relation channel.

Historical GDM64/GDM63 controls are context only, not a causal cross-batch leaderboard.

## Calibration and fresh holdout

Use new calibration-only domains **Catalog / Ledgerbook** and a fresh secondary holdout **Manuscript / Dossier**. These names and paraphrases must be new to optimizer/checkpoint/representation/calibration evidence before the GDM65 run.

The fresh holdout must contain the same four ambiguity families used by the canonical line, with explicit family labels used **only for post-training evaluation**:

- role;
- lifecycle-time;
- representation/name-vs-ID;
- object-vs-supplier.

The holdout must also include one-, two-, and three-clause answerable requests plus NO_MATCH and AMBIGUOUS risk cases. It never enters optimizer, checkpoint selection, representation construction, curriculum, or calibration. After canonical inspection it becomes regression-only.

## Required preflight tests

Before measured training, GitHub Actions must prove:

1. exact operation-only matrix: five arms × seeds `6501,6502` = 10 configs and no schema task;
2. all five arms have identical ambiguity input dimension `31 + 10*d`, identical downstream MLP shapes, and identical trainable parameter counts;
3. the first `11 + 10*d` coordinates are byte/coordinate-identical canonical whole-request-control features across every arm;
4. `structured-zero-control` appends exactly twenty zeros;
5. all non-control structural features are deterministic from candidate path/type metadata only and lie in `[0,1]`;
6. changing request text while holding the ranked catalog fixed cannot change the structural relation channel;
7. changing candidate parent/leaf/type metadata in a controlled fixture changes the intended relation features;
8. explicit fixtures prove `author.name` vs `author.id` has same-parent/different-leaf and typically different named type, while `author.name` vs `moderator.name` has different-parent/same-leaf;
9. missing TOP5 ranks zero-pad the entire four-scalar block;
10. capability scores/ranks are unchanged when only the GDM65 ambiguity relation arm changes;
11. exact GDM56 family-balanced curriculum and zero-counterfactual counts are retained;
12. GDM54 rescue/floor receipts are unchanged;
13. the new calibration and holdout domains are operation-only and absent from training inputs;
14. canonical GDM64 source/audit provenance is recorded as the parent evidence line;
15. deterministic hash smoke on `structured-hybrid-relation`, seed `6501`, independently passes the GDM65 collector.

Inherited GDM64–GDM50 contract tests must also run in preflight.

## Required measured evidence

For every one of the 10 configs, the source workflow and independent collector must require:

- changed capability and ambiguity checkpoint hashes with complete optimizer histories;
- exact numerical-contract receipts;
- exact 7,711-dimensional representation receipt and 20-scalar relation-channel kind/hash receipt;
- validation-only calibration/rescue evidence;
- per-example regression and fresh-holdout predictions;
- pooled and per-seed NO_MATCH/AMBIGUOUS counts;
- one-/two-/three-clause exact-operation metrics;
- four-family recount;
- semantic-confusion recount;
- real GraphQL parse, validation, and execution evidence;
- cached catalog relation-feature construction, with request encoding still fresh in timed generation;
- p50/p95 and process-level RSS as observational fields;
- no missing/failed/unexpected configs;
- `schema_configs_compared=0`.

## Promotion and reproducibility gate

If attempt 1 independently verifies 10/10, first confirm there are no active/queued/waiting/pending repository runs, then rerun the **same workflow run/source/seeds exactly once**.

Only after attempt 2 independently verifies 10/10 may a dedicated bounded-resource operation-only audit be created. The audit must reconstruct both BATCH reports with the source collector and require exact deterministic equality across all 10 configs, allowing cross-attempt differences only in aggregate observational p50/p95/RSS. It must verify checkpoint/optimizer histories, canonical feature hashes, relation-channel receipts/hashes, dimensions/parameter counts, curriculum/calibration contracts, regression/holdout/clause/family metrics, and exact per-example operation predictions.

Canonical promotion requires:

- `complete=true`;
- `evidence_verified=true`;
- `reproducible=true`;
- `promotion_eligible=true`;
- `errors=[]`;
- `expected_operation_configs=10`;
- `matched_operation_configs=10`;
- `schema_configs_compared=0`;
- all comparisons exact with `differing_fields=[]`.

The root README must be updated with the result and exact provenance **before** GDM65 is announced canonical or GDM66 is designed/launched.

## Decision rule

A structural relation arm is architecture-promotable only if, relative to `structured-zero-control`, the repeated measured evidence shows a meaningful improvement in the broad operation frontier—not merely one status count—including:

- no degradation in answerable exact-operation accuracy;
- no meaningful target-recall loss;
- no increase in incorrect-publication risk that overwhelms the ambiguity gain;
- improved AMBIGUOUS recovery and/or targeted role/representation family correctness;
- no multi-clause collapse, especially at three clauses;
- evidence that the gain is not carried by only one seed.

A narrow NO_MATCH or family-count movement without broad support is negative/inconclusive rather than an architecture win.
