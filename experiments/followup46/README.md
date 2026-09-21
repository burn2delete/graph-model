# GDM46 — corrected risk gating and dynamic hard negatives

GDM45 completed with a verified 40/40 evidence report. Its strongest reproducible result was that clause decomposition improved **answerable** request correctness for both tasks with DistilBERT. On the secondary holdout, the DistilBERT clause arms reached 24/32 answerable operation requests (75%) and 23/32 answerable schema requests (71.875%). DistilBERT was also materially faster and smaller than mmBERT in this CPU screening setup.

GDM45 also exposed two issues that require a bounded follow-up rather than a broad new backbone sweep.

First, operation risk handling failed completely: the DistilBERT clause arms accepted all ten NO_MATCH/AMBIGUOUS holdout cases. The existing status head sees the whole request plus frozen-encoder similarity statistics, but it does not see the learned capability scorer's confidence distribution.

Second, the shared synthetic dataset contains a benchmark shortcut for **schema risk cases**: accepted schema requests begin with `Expose capabilities for`, while the original NO_MATCH generator always used `Return`. That lexical cue means GDM45's schema status/risk figures are not valid evidence about semantic rejection. GDM46 corrects the public request construction so accepted, NO_MATCH and AMBIGUOUS examples use task-consistent syntax. GDM45's accepted-case capability-selection evidence remains useful; its schema risk comparison is not promoted.

GDM46 therefore fixes the backbone to DistilBERT and the decoder to clause decomposition, then tests four bounded hypotheses over two fixed seeds and both tasks:

- `joint-raw`: corrected-data replication of GDM45's jointly trained raw-similarity risk head.
- `balanced-raw`: capability training and status training are separated; the status classes are deterministically balanced.
- `balanced-learned`: the balanced status head additionally receives detached statistics from the learned capability scorer for every public clause.
- `balanced-learned-hardneg`: adds an online hardest-negative margin loss on accepted training clauses. The negative is mined only from the training example's public catalog; no test or holdout target is used.

The secondary holdout uses new type names and new paraphrase templates. It is synthetic and is **not** claimed to be enterprise OOD. Schema generation remains catalog projection plus deterministic SDL/Federation realization. Transformer backbones remain frozen; learned components are feature-space adapters and heads, not transformer fine-tuning.

Every arm must provide changed checkpoints, optimizer histories, per-example corrected-regression and secondary-holdout predictions, real GraphQL execution evidence for operations, Rover composition evidence for accepted schema predictions, raw single-request timings and an independently reproduced batch report. The batch report separates answerable accuracy, risk accuracy and incorrect-publication rate so overall accuracy cannot hide a broken rejection gate.
