import unittest
from experiments.followup51 import run as g51

class GDM51Contracts(unittest.TestCase):
    def test_fresh_holdout_and_calibration_are_disjoint(self):
        for task in ("operation","schema"):
            cr,cf=g51.calibration_set(task); hr,hf=g51.fresh_holdout(task)
            self.assertTrue(cr and hr)
            self.assertTrue(set(cf).isdisjoint(hf))
            self.assertEqual(set(cf),{r["id"] for r in cr})
            self.assertEqual(set(hf),{r["id"] for r in hr})
            prefix="Return " if task=="operation" else "Expose capabilities for "
            self.assertTrue(all(r["request"].startswith(prefix) for r in cr+hr))
            self.assertTrue(any(v["status"]=="AMBIGUOUS" for v in hf.values()))
            self.assertTrue(any(v["status"]=="NO_MATCH" for v in hf.values()))

    def test_request_aggregation_prioritizes_nomatch_then_ambiguity(self):
        cal={"none_margin_threshold":0.1,"ambiguity_probability_threshold":0.6}
        accepted={"best_real_id":"a","none_margin":-1.0,"ambiguity_probability":0.1}
        ambiguous={"best_real_id":"b","none_margin":-1.0,"ambiguity_probability":0.9}
        missing={"best_real_id":"c","none_margin":0.5,"ambiguity_probability":0.1}
        p=g51.request_from_evidence([accepted],cal,True); self.assertEqual(p["status"],"accepted"); self.assertEqual(p["selected"],["a"])
        p=g51.request_from_evidence([accepted,ambiguous],cal,True); self.assertEqual(p["status"],"AMBIGUOUS"); self.assertEqual(p["selected"],[])
        p=g51.request_from_evidence([ambiguous,missing],cal,True); self.assertEqual(p["status"],"NO_MATCH"); self.assertEqual(p["selected"],[])

    def test_ambiguity_feature_dimensions_are_explicit(self):
        self.assertEqual(g51._ambiguity_dim(False),7)
        self.assertEqual(g51._ambiguity_dim(True),11)
        self.assertEqual(set(g51.ARMS),{"null-margin-control","ambiguity-linear","ambiguity-mlp","ambiguity-structured","ambiguity-structured-hard"})

    def test_holdout_domains_do_not_overlap_public_train_or_validation(self):
        for task in ("operation","schema"):
            hr,_=g51.fresh_holdout(task)
            roots={r["catalog"]["root"] for r in hr}
            self.assertEqual(roots,{"incident","contract"})
            self.assertTrue(roots.isdisjoint({"product","book","project","ticket"}))

if __name__=="__main__": unittest.main()
