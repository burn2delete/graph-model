from __future__ import annotations

import unittest

from experiments.followup45.run import clauses
from experiments.followup50 import run as g50
from experiments.followup50.execute import features_with_none_cached
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


if __name__ == "__main__":
    unittest.main()
