# Parallel Research Program

We now treat schema generation and operation generation as separate model tasks sharing infrastructure.

## Batch A — operation generation
Run in parallel:
1. current-state legal-action policy;
2. terminal-coordinate prediction + deterministic path derivation;
3. Qwen retrieval top-2 + task head;
4. Qwen retrieval top-4 + task head;
5. typed pairwise option head;
6. independent candidate scorer;
7. frozen shared encoder + operation head;
8. operation-specific adapter.

Primary metrics: complete executable response correctness, path correctness, invalid-operation rate, p50/p95 latency, encoder calls, peak RSS.

## Batch B — schema generation
Run in parallel:
1. recursive schema-action policy;
2. capability-set prediction + deterministic realization;
3. typed pairwise schema-action head;
4. multilabel capability prediction;
5. frozen shared encoder + schema head;
6. schema-specific adapter;
7. generated structural curriculum + semantic supervision;
8. semantic-only supervision.

Primary metrics: requirement satisfaction, Rover composition, unnecessary capability rate, missing capability rate, ambiguity/refusal correctness, p50/p95 latency.

## Batch C — backbone sweep
For the best 2 operation architectures and best 2 schema architectures:
- DistilBERT
- ModernBERT
- mmBERT
- NeoBERT
- Qwen GraphQL embeddings as retrieval stage

## Batch D — sharing ablation
Compare:
- one shared policy;
- shared frozen encoder + separate heads;
- shared encoder + separate adapters;
- fully separate checkpoints.

## Evaluation invariants
- No expected-target fallback.
- Operation references are full paths / expected responses, not terminal coordinates alone.
- Schema references are requirement contracts, not one exact SDL unless the design is uniquely constrained.
- Held-out wording cannot be used to synthesize training examples.
- Every run reports speed and quality.
- Previously inspected tests are regression suites, not fresh generalization evidence.
