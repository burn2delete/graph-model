# Graph Model

Research and implementation for a **GraphQL Decision Model (GDM)**: a recursive, constrained alternative to token-by-token LLM generation for GraphQL schemas, operations, Federation structures, and query plans.

## Current direction

The current architecture combines:

- pretrained bidirectional semantic representations;
- request-defined candidate scoring;
- independent candidate support;
- explicit `NO_MATCH` and `AMBIGUOUS` outcomes;
- typed GraphQL/Federation construction;
- signed requirement constraints;
- recursive search and deterministic validation;
- Rover composition and executable GraphQL evaluation.

## Benchmark policy

Model promotion is based on generated and executed programs, not isolated classifier accuracy.

Every benchmark must include shortcut controls (candidate-position, decision-kind-only, lexical, and refusal baselines), split-integrity checks, randomized candidate order, paired requests that require different answers from the same candidates, and separate schema/operation/full-chain metrics.

## Current experiment

`experiments/gdm18/` is the first benchmark designed around those rules. It replaces the earlier GDM17 classifier-only benchmark as the promotion gate.
