# GDM45 — measured clause decomposition follow-up

GDM45 is the first follow-up promoted from the fully verified measured repair of GDM41–44. It does **not** use the fabricated/simulated metrics from the original GDM41–44 workflows.

## Measured hypothesis

The repaired batch showed that whole-request independent candidate scoring performs poorly on requests containing multiple semicolon-separated capabilities. Global retrieval `k` also acts as a cardinality ceiling: top-1/top-2 retrieval cannot recover three requested capabilities, while larger `k` admits adjacent false positives. Risk decisions (`accepted`, `NO_MATCH`, `AMBIGUOUS`) were entangled with capability selection.

GDM45 therefore compares five trained feature-model arms:

- `whole-risk`: cardinality-aware whole-request baseline plus an explicit 3-way risk/status head.
- `clause-risk`: split accepted-style public requests into atomic clauses and select one capability per clause, plus the same risk/status head.
- `clause-hardneg`: clause decomposition with extra margin loss for known semantic confusions such as created vs updated, author vs moderator, and author name vs author id.
- `clause-top2`: clause decomposition with same-backbone cosine top-2 candidate restriction at inference, followed by the trained reranker.
- `clause-top4`: the corresponding top-4 restriction.

Each arm is trained separately for schema generation and operation generation with DistilBERT and mmBERT frozen encoders and two seeds. This is a bounded 40-result batch. The encoder weights are frozen; adapters are in **feature space**. No result may be described as transformer fine-tuning.

## Evaluation

Checkpoint selection uses only the existing train/validation splits. The repaired GDM41–44 test set is retained as a regression suite because it has already been inspected. A secondary synthetic holdout uses new type names and paraphrases and is not used for checkpoint or hyperparameter selection. It is still synthetic and must not be described as human-authored OOD evidence.

Operation generation is evaluated by real GraphQL parse/validation and execution against three changing fixtures. Schema generation remains **catalog projection + deterministic SDL/Federation realization**, not unconstrained schema invention; accepted schema predictions are checked with Rover composition. Timing uses fresh request/query encoding with cached catalog vectors. Memory reports current process RSS and worker high-water; workers reuse one resident frozen backbone across arms.

The workflow must emit changed checkpoints, optimizer/update histories, per-example predictions for both regression and secondary holdout sets, raw timings, and a verified `BATCH_REPORT.json`. The collector fails closed on missing or malformed evidence.
