"""Independent evidence gate for GDM51."""
from __future__ import annotations
import argparse, hashlib, json, os, statistics
from collections import Counter, defaultdict
from pathlib import Path
import torch
from experiments.measured.evidence import timing
from experiments.measured.models import state_hash
from experiments.followup49.collect import derived, metrics_for, recompute_clause_metrics
from experiments.followup50.collect import holdout_detail
from .run import ARMS, FORMAT

def file_hash(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def load_json(path): return json.loads(Path(path).read_text())
def rows(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def verify_summary(path):
    directory=path.parent; s=load_json(path); errors=[]
    if s.get("format")!=FORMAT: errors.append("wrong format")
    if s.get("evidence_kind")!="trained-feature-model": errors.append("wrong evidence kind")
    cfg=s.get("config",{}); arm_name=cfg.get("arm"); task=cfg.get("task"); expected=ARMS.get(arm_name)
    if expected is None: errors.append("unexpected arm"); expected={}
    if cfg.get("family")!=expected.get("family"): errors.append("family mismatch")
    if cfg.get("model_scope")!="frozen pretrained encoder + learned feature-space adapter/head": errors.append("model scope mismatch")
    if cfg.get("backbone") not in {"distilbert","hash"}: errors.append("unexpected backbone")
    if task=="schema" and cfg.get("schema_scope")!="catalog projection + deterministic SDL/Federation realization": errors.append("schema scope mismatch")
    tr=s.get("training",{})
    if tr.get("optimizer_updates",0)<=0 or tr.get("capability_optimizer_updates",0)<=0: errors.append("missing capability optimizer evidence")
    if tr.get("selected_state_hash")==tr.get("initial_state_hash"): errors.append("selected state unchanged")
    if tr.get("selected_capability_hash")==tr.get("initial_capability_hash"): errors.append("capability unchanged")
    if tr.get("independent_backbone_finetuning") is not False or tr.get("adapter_location")!="frozen-embedding feature space": errors.append("backbone/adapter scope mismatch")
    if tr.get("gdm51_family")!=cfg.get("family"): errors.append("training family mismatch")
    caldoc=load_json(directory/"calibration.json") if (directory/"calibration.json").exists() else {}; cal=caldoc.get("selected",{})
    if cfg.get("family")=="explicit-none-plus-ambiguity":
        if tr.get("ambiguity_optimizer_updates",0)<=0: errors.append("missing ambiguity optimizer evidence")
        if tr.get("selected_ambiguity_hash")==tr.get("initial_ambiguity_hash"): errors.append("ambiguity head unchanged")
        if tr.get("risk_training_kind")!="explicit NONE listwise capability model plus separately trained top-two ambiguity discriminator; request thresholds validation-calibrated": errors.append("ambiguity risk kind mismatch")
        if cal.get("mode")!="explicit-none-plus-ambiguity-discriminator" or cal.get("selection_split")!="validation": errors.append("ambiguity calibration mismatch")
        if cal.get("calibration_scope")!="expanded-validation-only": errors.append("ambiguity calibration scope mismatch")
        for key in ("none_margin_threshold","ambiguity_probability_threshold"):
            if not isinstance(cal.get(key),(int,float)): errors.append("invalid "+key)
    else:
        if tr.get("risk_training_kind")!="explicit NONE candidate learned jointly with listwise capability scores; status thresholds validation-calibrated": errors.append("control risk kind mismatch")
        if cal.get("mode")!="explicit-none-margins" or cal.get("selection_split")!="validation": errors.append("control calibration mismatch")
    if s.get("calibration_only_sha256")==s.get("secondary_holdout_sha256"): errors.append("calibration/holdout hash collision")
    if "never used for optimizer/checkpoint/calibration selection" not in s.get("secondary_holdout_scope",""): errors.append("holdout scope missing prohibition")
    if "never used for optimizer updates or checkpoint selection" not in s.get("calibration_scope",""): errors.append("calibration scope missing optimizer prohibition")

    initial,selected=directory/"initial.pt",directory/"selected.pt"
    if not initial.exists() or not selected.exists(): errors.append("checkpoint missing")
    else:
        if file_hash(initial)==file_hash(selected): errors.append("checkpoint files identical")
        try:
            a=torch.load(initial,map_location="cpu",weights_only=True); b=torch.load(selected,map_location="cpu",weights_only=True)
            if state_hash(a["state"])!=tr.get("initial_state_hash"): errors.append("initial state hash mismatch")
            if state_hash(b["state"])!=tr.get("selected_state_hash"): errors.append("selected state hash mismatch")
        except Exception as exc: errors.append("checkpoint load failure:"+type(exc).__name__)
    for name,expected_hash in s.get("file_hashes",{}).items():
        p=directory/name
        if not p.exists() or file_hash(p)!=expected_hash: errors.append("file hash mismatch:"+name)

    hold_rows=[]
    for split,count in (("regression",s.get("regression_examples")),("holdout",s.get("secondary_holdout_examples"))):
        p=directory/f"predictions-{split}.jsonl"
        if not p.exists(): errors.append("missing "+split); continue
        pr=rows(p)
        if not pr or len(pr)!=count: errors.append("bad prediction count:"+split)
        correct=0
        for r in pr:
            if not all(k in r for k in ("public","reference","prediction","judgment")): errors.append("malformed prediction row"); break
            correct+=int(bool(r["judgment"].get("request_correct"))); ref_status=r["reference"]["status"]; pred_status=r["prediction"]["status"]
            prefix="Return " if r["public"]["task"]=="operation" else "Expose capabilities for "
            if not r["public"]["request"].startswith(prefix): errors.append("task prefix violated"); break
            if r["public"]["task"]=="operation" and ref_status=="accepted" and pred_status=="accepted":
                if not isinstance(r["judgment"].get("graphql_valid"),bool) or len(r["judgment"].get("response_matches",[]))!=3: errors.append("operation lacks execution evidence"); break
            if r["public"]["task"]=="schema" and pred_status=="accepted":
                comp=r["judgment"].get("composition")
                if not isinstance(comp,dict) or not isinstance(comp.get("success"),bool): errors.append("schema lacks Rover evidence"); break
            ev=r["prediction"].get("clause_evidence")
            if not isinstance(ev,list) or not ev or any(not all(k in e for k in ("best_real_id","none_margin","real_margin")) for e in ev): errors.append("missing clause evidence"); break
            if cfg.get("family")=="explicit-none-plus-ambiguity" and any(not isinstance(e.get("ambiguity_probability"),(int,float)) for e in ev): errors.append("missing ambiguity probability evidence"); break
        if pr:
            field="regression_metrics" if split=="regression" else "secondary_holdout_metrics"
            reported=float(metrics_for(s,field,task)["accuracy"])
            if abs(reported-correct/len(pr))>1e-12: errors.append(split+" accuracy not reproducible")
            cf="regression_clause_metrics" if split=="regression" else "secondary_holdout_clause_metrics"
            if recompute_clause_metrics(pr)!=s.get(cf): errors.append(split+" clause metrics not reproducible")
            if split=="holdout": hold_rows=pr

    td=load_json(directory/"timings.json") if (directory/"timings.json").exists() else {}; samples=td.get("generation_ms",[])
    if len(samples)<10 or any(not isinstance(x,(int,float)) or x<=0 for x in samples): errors.append("insufficient timings")
    elif timing(samples)!=s.get("generation_latency"): errors.append("timing summary mismatch")
    if "cached catalog/NONE" not in td.get("scope",""): errors.append("timing scope missing cache contract")
    enc=s.get("encoder",{})
    if cfg.get("backbone")!="hash" and (enc.get("frozen") is not True or not enc.get("revision")): errors.append("encoder not frozen/versioned")
    if errors: raise AssertionError(path.as_posix()+": "+"; ".join(sorted(set(errors))))
    return s,holdout_detail(hold_rows)

def main():
    p=argparse.ArgumentParser(); p.add_argument("root"); p.add_argument("--seeds",default="5101,5102"); p.add_argument("--tasks",default="operation,schema"); a=p.parse_args(); root=Path(a.root); seeds=[int(x) for x in a.seeds.split(",") if x]; tasks=[x for x in a.tasks.split(",") if x]
    expected={(t,arm,"distilbert",seed) for t in tasks for arm in ARMS for seed in seeds}; summaries={}; details={}; failures=[]
    for w in root.rglob("worker.json"):
        d=load_json(w)
        if not d.get("complete") or d.get("errors"): failures.append({"worker":str(w),"errors":d.get("errors")})
    for path in root.rglob("summary.json"):
        try:
            s,detail=verify_summary(path); cfg=s["config"]; key=(cfg["task"],cfg["arm"],cfg["backbone"],int(s["seed"]));
            if key in summaries: raise AssertionError("duplicate summary "+repr(key))
            summaries[key]=s; details[key]=detail
        except Exception as exc: failures.append({"summary":str(path),"error":str(exc)})
    missing=sorted(expected-set(summaries)); unexpected=sorted(set(summaries)-expected); dataset_hashes=sorted({s["dataset_sha256"] for s in summaries.values()}); holdout_hashes={t:sorted({s["secondary_holdout_sha256"] for k,s in summaries.items() if k[0]==t}) for t in tasks}
    if len(dataset_hashes)!=1: failures.append({"error":"dataset hashes differ","hashes":dataset_hashes})
    if any(len(v)!=1 for v in holdout_hashes.values()): failures.append({"error":"holdout hashes differ","hashes":holdout_hashes})

    groups={}
    for key,s in summaries.items():
        task,arm,backbone,seed=key; g=groups.setdefault((task,arm,backbone),{"reg":[],"hold":[],"p50":[],"p95":[],"rss":[],"recall":[],"precision":[],"clause":defaultdict(Counter),"risk":defaultdict(Counter),"conf":Counter()})
        g["reg"].append(derived(metrics_for(s,"regression_metrics",task))); g["hold"].append(derived(metrics_for(s,"secondary_holdout_metrics",task))); g["p50"].append(s["generation_latency"]["p50_ms"]); g["p95"].append(s["generation_latency"]["p95_ms"]); g["rss"].append(s["memory"].get("rss_after_evaluation_kib")); cm=s["secondary_holdout_clause_metrics"]
        if cm.get("target_recall") is not None:g["recall"].append(cm["target_recall"])
        if cm.get("target_precision") is not None:g["precision"].append(cm["target_precision"])
        d=details[key]
        for n,v in d["answerable_by_clause_count"].items(): g["clause"][n].update(v)
        for st,v in d["risk_by_reference_status"].items(): g["risk"][st].update(v)
        for c in d["semantic_confusions"]: g["conf"][(c["reference"],c["predicted"])]+=c["count"]
    ag=[]
    for (task,arm,backbone),v in sorted(groups.items()):
        ag.append({"task":task,"arm":arm,"backbone":backbone,"runs":len(v["hold"]),"mean_regression_accuracy":statistics.mean(x["accuracy"] for x in v["reg"]),"mean_secondary_holdout_accuracy":statistics.mean(x["accuracy"] for x in v["hold"]),"mean_answerable_holdout_accuracy":statistics.mean(x["answerable_accuracy"] for x in v["hold"]),"mean_risk_holdout_accuracy":statistics.mean(x["risk_accuracy"] for x in v["hold"]),"mean_incorrect_publication_rate":statistics.mean(x["incorrect_publication_rate"] for x in v["hold"]),"mean_target_recall":statistics.mean(v["recall"]) if v["recall"] else None,"mean_target_precision":statistics.mean(v["precision"]) if v["precision"] else None,"mean_p50_ms":statistics.mean(v["p50"]),"mean_p95_ms":statistics.mean(v["p95"]),"mean_rss_kib":statistics.mean(x for x in v["rss"] if x is not None),"answerable_by_clause_count":{k:dict(c) for k,c in sorted(v["clause"].items())},"risk_by_reference_status":{k:dict(c) for k,c in sorted(v["risk"].items())},"semantic_confusions":[{"reference":a,"predicted":b,"count":c} for (a,b),c in v["conf"].most_common(12)]})
    complete=not failures and not missing and not unexpected and len(summaries)==len(expected)
    report={"format":"gdm51-batch-report-v1","source_commit":os.environ.get("GITHUB_SHA","unrecorded"),"complete":complete,"verified":complete,"expected_results":len(expected),"verified_results":len(summaries),"missing_results":missing,"unexpected_results":unexpected,"failed_results":failures,"dataset_hashes":dataset_hashes,"secondary_holdout_hashes":holdout_hashes,"aggregates":ag,"hypotheses":["Retain GDM50 explicit NONE to represent open-set NO_MATCH rather than forcing a real catalog winner.","Replace the weak single real-margin ambiguity threshold with a separately trained top-two ambiguity discriminator.","Adding candidate-relationship structure to the ambiguity discriminator may distinguish genuinely ambiguous sibling concepts from merely close but resolvable candidates.","Hard-negative weighting may improve author/moderator and creation/revision ambiguity without restoring a request-global status head."],"interpretation_limits":["Frozen DistilBERT plus learned feature-space capability and ambiguity heads; no transformer fine-tuning.","The ambiguity discriminator is clause-local; request status is deterministic aggregation over clause decisions.","Schema generation remains catalog projection plus deterministic SDL/Federation realization, not unconstrained schema invention.","The Incident/Contract secondary holdout is synthetic and not enterprise OOD."]}
    (root/"BATCH_REPORT.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); print(json.dumps(report,indent=2,sort_keys=True))
    if not complete: raise SystemExit(1)

if __name__=="__main__": main()
