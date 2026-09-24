from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from experiments.followup51 import collect as c51
from experiments.followup52 import collect as c52
from experiments.followup65 import run as r

def verify_summary(path):
 oa,of=c51.ARMS,c51.FORMAT;c51.ARMS=r.ARMS;c51.FORMAT=r.FORMAT
 try:s,d=c51.verify_summary(path)
 finally:c51.ARMS=oa;c51.FORMAT=of
 q=s["config"];t=s["training"];m=r.ARMS[q["arm"]]["structured"]
 checks=[q.get("task")=="operation",q.get("forward_scope")=="operation-generation-only",q.get("schema_generation")=="suspended-frozen",q.get("ambiguity_feature_dim") in (1311,7711),q.get("ambiguity_relation_feature_count")==20,q.get("ambiguity_relation_contract_hash")==r._relation_hash(m),t.get("gdm65_base_semantics")=="exact canonical ranked top5 q*ci + abs(q-ci)",t.get("gdm65_relation_feature_count")==20,t.get("gdm65_relation_contract_hash")==r._relation_hash(m),t.get("gdm65_relation_request_independent") is True,t.get("canonical_gdm64_source_commit")==r.CANONICAL_GDM64_SOURCE,t.get("canonical_gdm64_audit_run")==r.CANONICAL_GDM64_AUDIT_RUN,t.get("canonical_gdm64_promotion_commit")==r.CANONICAL_GDM64_PROMOTION]
 if not all(checks):raise AssertionError(path.as_posix()+": GDM65 receipt contract failed")
 return s,d

def main():
 p=argparse.ArgumentParser(add_help=False);p.add_argument("root");p.add_argument("--seeds",default="6501,6502");p.add_argument("--tasks",default="operation");a,_=p.parse_known_args()
 if a.tasks!="operation":raise SystemExit("operation only")
 oa,of,ov,av=c52.ARMS,c52.FORMAT,c52.verify_summary,list(sys.argv);c52.ARMS=r.ARMS;c52.FORMAT=r.FORMAT;c52.verify_summary=verify_summary;sys.argv=[sys.argv[0],a.root,"--seeds",a.seeds,"--tasks","operation"]
 try:c52.main()
 finally:c52.ARMS=oa;c52.FORMAT=of;c52.verify_summary=ov;sys.argv=av
 f=Path(a.root)/"BATCH_REPORT.json";x=json.loads(f.read_text());x.update({"format":"gdm65-operation-batch-report-v1","forward_scope":"operation-generation-only","schema_configs_compared":0});f.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
