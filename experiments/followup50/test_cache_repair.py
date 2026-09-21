from __future__ import annotations

import os
import unittest

from experiments.followup45.run import clauses
from experiments.followup50 import run as g50
from experiments.followup50.execute import ReproducibleEncoder, features_with_none_cached
from experiments.measured.models import Encoder


class GDM50CacheRepairTests(unittest.TestCase):
    def test_online_request_reencodes_only_query_clause(self):
        rows, _ = g50.fresh_holdout("operation")
        item = rows[0]
        clause = clauses(item)[0]
        encoder = Encoder("hash")

        # Populate immutable catalog/NONE embeddings first, as the measured
        # worker does during training/evaluation before warm-latency sampling.
        features_with_none_cached(encoder, item, clause, online=False)
        before_calls = encoder.calls
        before_texts = encoder.texts

        features_with_none_cached(encoder, item, clause, online=True)

        self.assertEqual(encoder.calls - before_calls, 1)
        self.assertEqual(encoder.texts - before_texts, 1)

    def test_cached_and_online_feature_shapes_match(self):
        rows, _ = g50.fresh_holdout("schema")
        item = rows[0]
        clause = clauses(item)[0]
        encoder = Encoder("hash")
        cached_x, cached_opts, _ = features_with_none_cached(encoder, item, clause, online=False)
        online_x, online_opts, _ = features_with_none_cached(encoder, item, clause, online=True)
        self.assertEqual(cached_x.shape, online_x.shape)
        self.assertEqual([o["id"] for o in cached_opts], [o["id"] for o in online_opts])

    def test_reproducible_encoder_records_runtime_and_corpus_contract(self):
        self.assertEqual(os.environ.get("ATEN_CPU_CAPABILITY"), "default")
        encoder = ReproducibleEncoder("hash")
        encoder.encode(["alpha GraphQL capability", "beta GraphQL capability"], cached=True)
        cpu = encoder.meta["cpu_reproducibility"]
        corpus = encoder.meta["feature_corpus"]
        self.assertEqual(cpu["aten_cpu_capability_env"], "default")
        self.assertEqual(cpu["aten_cpu_capability_runtime"], "DEFAULT")
        self.assertEqual(corpus["unique_texts"], 2)
        self.assertEqual(len(corpus["raw_text_feature_sha256"]), 64)
        self.assertEqual(len(corpus["canonical_text_feature_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
