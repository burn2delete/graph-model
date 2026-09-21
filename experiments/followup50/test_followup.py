from __future__ import annotations
import unittest
import torch
from experiments.measured.models import Encoder
from experiments.followup50 import run

class GDM50Contracts(unittest.TestCase):
    def test_bounded_arm_matrix(self):
        self.assertEqual(set(run.ARMS), {"listwise-hybrid-control","null-argmax","null-calibrated","null-balanced","null-contrast"})
        self.assertEqual(run.ARMS["listwise-hybrid-control"]["family"], "gdm49-listwise-control")
        for name,cfg in run.ARMS.items():
            if name!="listwise-hybrid-control": self.assertEqual(cfg["family"], "explicit-none-listwise")

    def test_calibration_and_holdout_disjoint(self):
        for task in ("operation","schema"):
            cr,_=run.calibration_set(task); hr,_=run.fresh_holdout(task)
            self.assertTrue(cr and hr)
            self.assertTrue({r["id"] for r in cr}.isdisjoint({r["id"] for r in hr}))
            self.assertTrue({r["request"] for r in cr}.isdisjoint({r["request"] for r in hr}))

    def test_holdout_task_prefix_and_risk_coverage(self):
        for task,prefix in (("operation","Return "),("schema","Expose capabilities for ")):
            rows,refs=run.fresh_holdout(task); statuses={refs[r["id"]]["status"] for r in rows}
            self.assertEqual(statuses,{"accepted","NO_MATCH","AMBIGUOUS"})
            self.assertTrue(all(r["request"].startswith(prefix) for r in rows))
            self.assertTrue(any("; " in r["request"] and refs[r["id"]]["status"]=="accepted" for r in rows))
            self.assertTrue(any("; " in r["request"] and refs[r["id"]]["status"]!="accepted" for r in rows))

    def test_language_partitions(self):
        hold=set(run.HOLDOUT_LANGUAGE+run.HOLDOUT_UNSUPPORTED+run.HOLDOUT_AMBIGUOUS)
        cal=set(run.CALIBRATION_LANGUAGE+run.CALIBRATION_UNSUPPORTED+run.CALIBRATION_AMBIGUOUS)
        self.assertTrue(hold.isdisjoint(cal))
        self.assertTrue(hold.isdisjoint(set(run.g49.NO_MATCH_TRAINING+[p for p,_ in run.g49.AMBIGUOUS_TRAINING])))

    def test_explicit_none_feature_contract(self):
        enc=Encoder("hash"); row,_=run._make_case("operation","Widget","widget",run.HOLDOUT_LANGUAGE[0],"accepted",targets=[run.SEMANTIC_SUFFIXES[0]],tag="unit")
        x,opts,raw=run.features_with_none(enc,row,run.HOLDOUT_LANGUAGE[0])
        self.assertEqual(opts[-1]["id"],run.NONE_ID)
        self.assertEqual(x.shape[0],len(row["catalog"]["options"])+1)
        self.assertEqual(x.shape[1],enc.dim*5+4)
        self.assertEqual(len(raw),len(opts))

    def test_status_aggregation(self):
        ev=[{"best_real_id":"x","none_margin":-1.0,"real_margin":1.0},{"best_real_id":"y","none_margin":0.4,"real_margin":1.0}]
        p=run.request_from_evidence(ev,{"none_margin_threshold":0.0,"ambiguity_margin_threshold":0.2})
        self.assertEqual(p["status"],"NO_MATCH")
        ev=[{"best_real_id":"x","none_margin":-1.0,"real_margin":0.1}]
        p=run.request_from_evidence(ev,{"none_margin_threshold":0.0,"ambiguity_margin_threshold":0.2})
        self.assertEqual(p["status"],"AMBIGUOUS")

    def test_schema_scope_catalog_projection(self):
        rows,refs=run.fresh_holdout("schema")
        for row in rows:
            if refs[row["id"]]["status"]!="accepted": continue
            available={o["id"] for o in row["catalog"]["options"]}
            self.assertTrue(set(refs[row["id"]]["paths"]).issubset(available))

if __name__=="__main__": unittest.main()
