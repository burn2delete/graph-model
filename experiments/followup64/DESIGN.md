# GDM64 design: schema-coordinate semantic factorization for operation ambiguity

## Scope and causal question

GDM64 is **operation-generation only**. Schema generation remains suspended/frozen and query planning remains out of scope. The forward task is unchanged: natural-language intent/request + provided GraphQL schema/capability catalog -> correct executable GraphQL operation, including calibrated `NO_MATCH` / `AMBIGUOUS` abstention.

GDM63 established a negative/inconclusive result for a small learned rank-preserving cross-candidate interaction adapter. The learned interaction state changed and the five arms had matched capacity, but all five arms produced the same answerable accuracy, target precision/recall, clause exactness, pooled ambiguity-family results, and dominant role/name-vs-ID confusions. Only a few `NO_MATCH` decisions moved.

GDM64 therefore changes the **information factorization**, not candidate-set interaction or model size. The hypothesis is:

> Whole-candidate embeddings may collapse distinct schema-coordinate semantics before the ambiguity head sees them. Separately exposing parent/role path semantics and leaf/representation semantics may improve distinctions such as author vs moderator and name vs ID while preserving the canonical rank-preserving TOP5 request-relative structure.

This is an unverified hypothesis. No GDM64 result may be interpreted before the declared GitHub Actions measurement, exact same-source rerun, independent audit, and README promotion gate.

## Frozen contracts

The following remain fixed from the canonical operation line through GDM63:

- operation task only;
- frozen `distilbert/distilbert-base-uncased` encoder at the pinned source revision;
- explicit `1e-4` canonical frozen-feature boundary;
- TOP5 real candidates with rank identity preserved;
- explicit NONE candidate and unchanged capability training;
- exact GDM56 family-balanced ambiguity curriculum with zero counterfactuals;
- GDM50 numerical execution path: one-thread/default CPU dispatch, oneDNN off, PyTorch 2.8 single-tensor AdamW `foreach=false` / `fused=false`, non-foreach clipping, float32 moments/state, final decoupled-weight-decay plus bias-corrected parameter application evaluated float64 -> stored float32;
- GDM54 validation-only 5pp ambiguity-rescue arbitration;
- same canonical `11` scalar/structural ambiguity features;
- same ambiguity MLP architecture and trainable parameter count in every arm (`dim -> 32 -> 1`);
- deterministic schema-grounded GraphQL selection-tree realization;
- real GraphQL parse, validation, and fixture execution;
- no transformer fine-tuning;
- no request-global trained status head;
- holdout never enters optimizer, checkpoint, representation, curriculum, or calibration selection.

The canonical whole-candidate request-relative control is included in the same batch as the primary comparator. Historical GDM62/GDM63 fresh holdouts are not causal comparators.

## Deterministic schema-coordinate semantic views

For each real candidate option `ci` with GraphQL path `path`, GDM64 deterministically constructs two additional **catalog-derived textual views** using only the provided schema/catalog coordinate:

- **parent/role view `pi`**: `"GraphQL parent path: " + ".".join(path[1:-1])`. If there is no non-root parent segment, use the literal token `<root>`.
- **leaf/representation view `li`**: `"GraphQL leaf field: " + path[-1]`.

These strings do not use reference labels, holdout family labels, expected answers, or example-specific annotations. They are derived solely from the candidate's provided GraphQL path.

`pi` and `li` are embedded by the same frozen encoder and cross the same canonical `1e-4` feature boundary as other frozen semantic vectors. They are static catalog evidence and must be cached outside timed request generation exactly like existing candidate/NONE vectors. Request clause `q` remains freshly encoded in timed generation.

Missing candidate ranks are zero-padded exactly as in the canonical TOP5 representation. Parent/leaf vectors must never alter capability scoring or candidate rank; they are ambiguity-head evidence only.

## Matched representation capacity

Every arm supplies exactly two `d`-dimensional semantic blocks per TOP5 rank, preserving the existing ambiguity dimension:

- DistilBERT: `11 + 10*768 = 7,691`;
- hash smoke/control: `11 + 10*128 = 1,291`.

Every arm uses the same canonical `dim -> 32 -> 1` ambiguity MLP and therefore the same trainable parameter count. GDM64 adds **no learned adapter parameters**. Only the deterministic semantic source of the two rank-preserving blocks differs.

For candidate rank `i`, define whole-candidate vector `ci`, parent/role vector `pi`, leaf/representation vector `li`, and request vector `q`.

## Arms

Exactly five arms are measured with seeds `6401,6402`, for exactly **10 operation configurations**:

1. **`whole-request-control`** — exact retained canonical representation: `(q * ci, abs(q - ci))`.
2. **`parent-request-factor`** — parent/role-only request-relative evidence: `(q * pi, abs(q - pi))`.
3. **`leaf-request-factor`** — leaf/representation-only request-relative evidence: `(q * li, abs(q - li))`.
4. **`split-parent-leaf-product`** — separate role and representation product evidence: `(q * pi, q * li)`.
5. **`split-parent-leaf-delta`** — separate role and representation delta evidence: `(abs(q - pi), abs(q - li))`.

This matrix has explicit ablations: parent-only tests role/path semantics, leaf-only tests representation semantics, split arms expose both factors without increasing dimensions, and the whole-request control anchors the canonical representation on the same fresh holdout/seeds.

No arm may add candidate-to-candidate interaction, confidence attenuation, extra ranks, additional dimensions, extra trainable parameters, or hidden holdout-derived features.

## Calibration and fresh holdout

Use new calibration-only domains **Registry / Logbook**. Use new fresh secondary-holdout domains **Folio / Casebook**.

The fresh holdout must preserve the four prespecified ambiguity families:

- `role`;
- `lifecycle-time`;
- `representation` (name-vs-ID style);
- `object-vs-supplier`.

Fresh Folio/Casebook examples and paraphrases must not influence optimizer updates, checkpoint selection, representation choice, curriculum, or threshold calibration. After the first canonical inspection, Folio/Casebook becomes regression-only.

## Preflight gates

Before any full measured worker runs, GitHub Actions must prove:

- exact operation-only matrix: 5 arms × 2 seeds = 10;
- no schema task/config in the forward batch;
- inherited GDM63–GDM50 tests still pass;
- exact canonical GDM62/GDM63 operation provenance receipts are present;
- whole-request-control is coordinate-identical to the retained canonical `q*ci + abs(q-ci)` representation;
- parent/leaf strings are deterministic functions of the provided candidate path only;
- parent and leaf vectors are frozen/cached catalog evidence and do not alter capability ranking;
- all arms have matched `1,291` hash / `7,691` DistilBERT ambiguity dimensions and identical ambiguity-head parameter counts;
- parent factor distinguishes author vs moderator coordinates in deterministic fixtures;
- leaf factor distinguishes name vs ID coordinates in deterministic fixtures;
- missing ranks zero-pad safely;
- GDM56 family-balanced curriculum and zero-counterfactual receipts remain exact;
- GDM54 rescue/calibration contracts remain exact;
- a deterministic hash smoke for `split-parent-leaf-product`, seed `6401`, independently passes the GDM64 collector.

## Measured evidence requirements

The source workflow must run one operation worker covering all 10 configs and an independent collector. A valid `gdm64-operation-batch-report-v1` must have:

- `forward_scope=operation-generation-only`;
- `complete=true`, `verified=true`;
- `expected_results=10`, `verified_results=10`;
- `schema_configs_compared=0`;
- no missing/failed/unexpected configs;
- changed capability and ambiguity checkpoints for every trained config;
- optimizer histories and selected-checkpoint receipts;
- exact representation receipts identifying `whole`, `parent`, and `leaf` block sources;
- exact dimension/parameter/TOP5/cache receipts;
- canonical numerical metadata;
- exact GDM56 curriculum and GDM54 validation-only rescue receipts;
- per-example regression and fresh Folio/Casebook predictions;
- four-family recount;
- real GraphQL parse/validate/execute evidence;
- raw generation p50/p95 and process RSS with declared scope.

## Reproducibility and promotion gate

If attempt 1 verifies 10/10, first confirm no active/queued/waiting/pending repository work, then rerun the **same workflow run/source/seeds** exactly once. After attempt 2 verifies 10/10, run a dedicated bounded-resource operation-only exact-attempt audit pinned to the actual source commit.

The audit must independently rebuild both source reports and compare all 10 configurations exactly across attempts. Deterministic BATCH_REPORT equality may exclude only declared aggregate observational p50/p95/RSS fields. Per-example equality may exclude only the already-declared composition elapsed-time observation. Canonical fixed-probe/full-corpus feature hashes must match even if raw encoder hashes vary for a documented runner reason.

Promotion requires `complete=true`, `evidence_verified=true`, `reproducible=true`, `promotion_eligible=true`, `errors=[]`, `matched_operation_configs=10`, `schema_configs_compared=0`, and all 10 comparisons exact.

After audit pass, analyze operation answerable/risk correctness, incorrect publication, target precision/recall, NO_MATCH/AMBIGUOUS, all four families, seed stability, one/two/three-clause exactness, semantic confusions, regression, p50/p95, and process RSS. Update root README.md in the same promotion step **before** announcing GDM64 canonical or designing/launching GDM65.

## Decision rule

The schema-coordinate factorization hypothesis is supported only if one or more factorized arms improve the **within-GDM64** answerable/safety frontier and/or prespecified role/representation families without a material publication-safety or multi-clause regression. A family-specific improvement with broad degradation is a tradeoff, not an architecture victory. Equal or worse results across factorized arms are negative/inconclusive evidence and retain the whole-request control.
