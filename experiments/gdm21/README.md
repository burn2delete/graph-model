# GDM21 — Retrieval + Decision Cascade

GDM21 evaluates a two-stage architecture rather than using one large encoder over the entire GraphQL coordinate space.

1. GraphQL-specialized Qwen embedding retrieves top-K schema coordinates.
2. A decision encoder reranks only those candidates.
3. Constrained GraphQL/Federation construction validates the selected action.

## Required measurements

For K = 1, 2, 4, 8, 16, 32:

- retrieval recall@K;
- final decision accuracy;
- executable GraphQL success;
- end-to-end p50/p95 latency;
- candidates scored by the decision encoder;
- peak RSS;
- model parameters;
- incorrect publication and abstention.

Compare ModernBERT, mmBERT, and NeoBERT as rerankers after the GDM20 bake-off completes. Include direct single-stage inference as a control.

The promotion target is the Pareto frontier, not maximum accuracy alone.
