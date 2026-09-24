# GDM63 prespecified design — learned rank-preserving cross-candidate interaction

## Scope

GDM63 is **operation-generation only**. Schema generation remains suspended/frozen and query planning remains out of scope. The target is natural-language intent/request + provided GraphQL schema/capability catalog -> correct executable GraphQL operation, including safe `NO_MATCH` / `AMBIGUOUS` handling.

This design is committed before implementation and before the workflow exists so the measured run cannot silently redefine the comparison after seeing results.

## Motivation

Canonical GDM62 showed that fixed rank-1 candidate-to-candidate products/deltas do not improve the broad operation answerable/safety frontier and can materially damage the fresh role ambiguity family. The retained representation is therefore the canonical rank-preserving request-relative pair `q*ci` + `abs(q-ci)` over the top five real candidates.

The next bounded question is whether a **small learned interaction over those already-canonical ranked request-relative slots** can extract useful candidate-set contrast that fixed algebraic transforms cannot.

## Fixed components

All arms keep fixed:

- operation task only;
- seeds `6301,6302`;
- top-five ranked real candidates and rank identity;
- the exact canonical raw ambiguity input `11 + 10*d`: eleven scalar/structural features plus five `q*ci` slots and five `abs(q-ci)` slots;
- explicit `NONE` and capability training;
- frozen DistilBERT and the `1e-4` canonical feature boundary;
- exact GDM56 family-balanced ambiguity curriculum with zero counterfactual-negative fraction;
- GDM50 canonical numerical execution;
- GDM54 validation-only five-percentage-point ambiguity-rescue arbitration;
- deterministic schema-grounded GraphQL operation realization and real parse/validate/execute evidence;
- ambiguity MLP width `32` after the interaction adapter;
- the same learned interaction parameterization and exact trainable parameter count across all five arms.

No holdout example may enter optimizer updates, checkpoint selection, representation design, threshold calibration, or rescue selection.

## Matched-capacity learned interaction

Each ambiguity head receives the unchanged canonical `11 + 10*d` vector. The ten semantic blocks are reinterpreted as five rank slots, each slot concatenating the candidate's request-product and request-delta vectors, giving shape `[5, 2*d]`.

Every arm uses the same small shared projection from each rank slot to latent width `4`, the same learned rank gates, temperature, residual scale and interaction bias, and the same downstream `dim -> 32 -> 1` ambiguity MLP. Thus all five arms have exactly equal trainable parameter count and optimizer treatment.

The arms differ **only** in how the five projected rank slots are combined to produce a scalar gate for each rank:

1. `learned-local-control` — each rank uses only its own latent self-energy; this is the matched-capacity non-cross-candidate control over the canonical request-relative representation.
2. `learned-top1-cross` — each rank interacts with the rank-1 slot.
3. `learned-neighbor-cross` — each rank interacts with the immediately preceding rank; rank 1 uses itself.
4. `learned-allpairs-cross` — each rank uses the mean learned similarity to all other active ranks.
5. `learned-competitive-cross` — each rank uses strongest-other similarity minus mean-other similarity, testing whether relative competition rather than average relation is useful.

The resulting learned scalar gates multiplicatively modulate the existing request-product and request-delta vectors in-place. Rank order and ambiguity input dimension remain unchanged. The module is a learned feature-space interaction over frozen representations, **not transformer fine-tuning** and not a trained request-global status head.

## Fairness and interpretation

The primary causal comparison is the four cross-candidate arms against `learned-local-control`, because they share identical raw inputs, parameter count, initialization family, optimizer, curriculum and downstream MLP. The prior GDM62 canonical request-relative model remains an external historical baseline but is not a same-holdout causal comparator.

A positive result requires a cross-candidate arm to improve useful operation evidence, particularly role / representation ambiguity and multi-clause exactness, without merely trading away answerable correctness or publication safety. A lower incorrect-publication rate alone is not sufficient if answerable accuracy, recall or `AMBIGUOUS` recovery collapse.

## Fresh evaluation

Calibration-only domains: `Ledger` / `Index`.

Fresh secondary holdout domains: `Docket` / `Portfolio`.

The holdout preserves the four ambiguity families — role, lifecycle-time, representation/name-vs-ID, object-vs-supplier — using new paraphrases. After canonical inspection it becomes regression-only.

## Required measured matrix and evidence

Exactly ten trained configurations: one operation task × five arms × seeds `6301,6302`.

Preflight must compile GDM63 and inherited contracts and run a deterministic hash-backbone smoke for one learned cross arm. Each measured result must contain changed capability and ambiguity checkpoints, optimizer histories, adapter receipts/parameter counts, fixed numerical/curriculum/calibration receipts, per-example regression and fresh-holdout predictions, real GraphQL parse/validate/execute evidence, raw timing samples and process RSS.

Attempt 1 must independently verify `10/10`, then the same workflow run/commit/seeds is rerun exactly once. Promotion requires a dedicated operation-only exact-attempt audit matching all ten deterministic configurations/predictions; only p50/p95/RSS are observational. README synchronization is mandatory before canonical announcement or GDM64 design.
