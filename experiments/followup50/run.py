"""GDM50 measured explicit-NONE listwise open-set follow-up.

GDM49 verified that independent binary matching improves risk precision but collapses
answerable recall, especially for multi-clause requests. GDM50 keeps frozen DistilBERT
and clause decomposition fixed while restoring listwise capability discrimination and
adding an explicit NONE candidate per clause. NONE can absorb unsupported clauses;
a validation-calibrated real-option margin handles ambiguity.

All compilation, training, evaluation and timing execute only in GitHub Actions. The
transformer remains frozen; learned modules are feature-space adapters/heads. Schema
output remains catalog projection plus deterministic SDL/Federation realization.
"""
from __future__ import annotations

import argparse, copy, gc, json, math, os, random, resource, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.measured.contracts import catalog, judge, option_text, sha
from experiments.measured.evidence import Compositor, aggregate, file_hash, timing, write_json
from experiments.measured.models import Encoder, Head, state_hash
from experiments.followup45.run import clauses
from experiments.followup46 import run as g46
from experiments.followup46.run import corrected_dataset, current_rss_kib, task_prefix, validate_emission
from experiments.followup47 import run as g47
from experiments.followup48 import run as g48
from experiments.followup49 import run as g49

FORMAT = "gdm50-measured-v1"
SEEDS_DEFAULT = "5001,5002"
NONE_ID = "__NONE__"
NONE_TEXT = "No available GraphQL capability in this catalog satisfies the request clause."

ARMS = {
    "listwise-hybrid-control": {"family":"gdm49-listwise-control", "calibration":"public-validation"},
    "null-argmax": {"family":"explicit-none-listwise", "none_balance":1.0, "contrast":False, "fixed_none_threshold":0.0},
    "null-calibrated": {"family":"explicit-none-listwise", "none_balance":1.0, "contrast":False, "fixed_none_threshold":None},
    "null-balanced": {"family":"explicit-none-listwise", "none_balance":1.75, "contrast":False, "fixed_none_threshold":None},
    "null-contrast": {"family":"explicit-none-listwise", "none_balance":1.25, "contrast":True, "fixed_none_threshold":None},
}
SEMANTIC_SUFFIXES = g47.SEMANTIC_SUFFIXES

CALIBRATION_LANGUAGE = [
    "visible heading used to identify the object to end users",
    "moment persistent storage first recorded the object",
    "moment persistent storage most recently changed the object",
    "whole number score carried by each review",
    "public names belonging to review writers",
    "public names belonging to review moderators",
    "registered business name of the supplying organization",
    "stable identifiers belonging to review writers",
]
CALIBRATION_UNSUPPORTED = [
    "warehouse shelf assigned to the object", "cancellation explanation attached to the object",
    "tax amount charged for the object", "boolean indicating whether the object is archived",
]
CALIBRATION_AMBIGUOUS = [
    "review participant name without choosing writer or moderator",
    "object timestamp without choosing first creation or latest revision",
]
HOLDOUT_LANGUAGE = [
    "display caption shown to customers for this record",
    "timestamp of the record's first durable write",
    "timestamp of the record's latest durable rewrite",
    "integer score recorded on every review",
    "names shown for people who wrote each review",
    "names shown for people who moderated each review",
    "company name shown for the record's supplier",
    "IDs assigned to people who wrote each review",
]
HOLDOUT_UNSUPPORTED = [
    "physical aisle where the record is stored", "reason the record was voided",
    "delivery surcharge charged for the record", "flag indicating whether the record is quarantined",
]
HOLDOUT_AMBIGUOUS = [
    "review participant name where author versus moderator is unspecified",
    "record timestamp where original write versus latest rewrite is unspecified",
]

class NullBundle(nn.Module):
    def __init__(self, pair_dim:int):
        super().__init__(); self.capability = Head(pair_dim, adapter=True)

def _subhash(module): return state_hash(copy.deepcopy(module.state_dict()))

def _copy_catalog(base, options=None):
    out={k:copy.deepcopy(v) for k,v in base.items() if k!="options"}; out["options"]=copy.deepcopy(base["options"] if options is None else options); return out

def _suffix(option): return tuple(option["path"][1:])

def _option_for_suffix(options, suffix):
    m=[o for o in options if _suffix(o)==tuple(suffix)]
    if len(m)!=1: raise AssertionError((suffix,len(m)))
    return m[0]

def features_with_none(encoder, item, clause, online=False):
    real=item["catalog"]["options"]
    q=encoder.query(clause, cached=not online)
    texts=[option_text(o,True) for o in real]+[NONE_TEXT]
    c=encoder.encode(texts, cached=not online)
    qx=q.expand_as(c); centered=c-c.mean(0,keepdim=True)
    state=[]
    for o in real:
        state.append([len(o["path"])/5.0, float(item["task"]=="schema"), len(o.get("coordinates",[]))/5.0, float(o["path"][-1] in {"name","id"})])
    state.append([0.0,float(item["task"]=="schema"),0.0,0.0])
    x=torch.cat([qx,c,qx*c,(qx-c).abs(),centered,torch.tensor(state,dtype=torch.float32)],dim=-1)
    opts=real+[{"id":NONE_ID,"path":[],"coordinates":[],"types":[],"gloss":NONE_TEXT}]
    return x,opts,c@q

def training_examples(task, public, refs):
    rows=[r for r in public["train"] if r["task"]==task]
    return g49._pair_examples_from_rows(task, rows, refs["train"])+g49.training_curriculum(task, rows)

def null_loss(bundle, encoder, row, clause, positives, arm):
    x,opts,_=features_with_none(encoder,row,clause)
    logits=bundle.capability(x); ids=[o["id"] for o in opts]
    target=torch.zeros_like(logits)
    if positives:
        idx=[ids.index(p) for p in positives]; target[idx]=1.0/len(idx); source="real"
    else:
        target[-1]=1.0; source="none"
    loss=-(target*F.log_softmax(logits,dim=-1)).sum()
    if source=="none": loss=loss*float(arm["none_balance"])
    if len(positives)>1: loss=loss*1.15
    if arm.get("contrast"):
        pos=logits[target>0]; neg=logits[target==0]
        if len(pos) and len(neg): loss=loss+0.35*F.softplus(neg.max()-pos.min()+0.65)
    return loss

def validation_rank(bundle, encoder, task, rows, refs):
    accepted_total=accepted_correct=none_total=none_correct=0
    bundle.eval()
    with torch.no_grad():
        for row in rows:
            if row["task"]!=task: continue
            ref=refs[row["id"]]; parts=clauses(row)
            if ref["status"]=="accepted" and len(parts)==len(ref["paths"]):
                ok=True
                for clause,target in zip(parts,ref["paths"]):
                    _,opts,_=features_with_none(encoder,row,clause); logits=bundle.capability(features_with_none(encoder,row,clause)[0]); pred=opts[int(torch.argmax(logits))]["id"]; ok &= pred==target
                accepted_total+=1; accepted_correct+=int(ok)
            elif ref["status"]=="NO_MATCH":
                ok=True
                for clause in parts:
                    x,opts,_=features_with_none(encoder,row,clause); pred=opts[int(torch.argmax(bundle.capability(x)))]["id"]; ok &= pred==NONE_ID
                none_total+=1; none_correct+=int(ok)
    ar=accepted_correct/accepted_total if accepted_total else 0.0; nr=none_correct/none_total if none_total else 0.0
    hm=2*ar*nr/(ar+nr) if ar+nr else 0.0
    return {"accepted_top1_accuracy":ar,"nomatch_none_top1_accuracy":nr,"rank_hmean":hm,"accepted_examples":accepted_total,"nomatch_examples":none_total}

def train_null(bundle, encoder, arm, task, public, refs, seed, epochs, directory):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    initial=copy.deepcopy(bundle.state_dict()); ih=state_hash(initial); ich=_subhash(bundle.capability)
    torch.save({"state":initial,"arm":arm,"task":task},directory/"initial.pt")
    examples=training_examples(task,public,refs); val=[r for r in public["validation"] if r["task"]==task]
    opt=torch.optim.AdamW(bundle.capability.parameters(),lr=8e-4,weight_decay=1e-3)
    best=None; best_key=None; updates=sel_updates=sel_epoch=0; max_grad=0.0; hist=[]
    for epoch in range(1,epochs+1):
        order=list(examples); random.Random(seed+epoch).shuffle(order); losses=[]; bundle.train()
        for row,clause,positives,source in order:
            opt.zero_grad(set_to_none=True); loss=null_loss(bundle,encoder,row,clause,positives,arm); loss.backward()
            norm=float(torch.nn.utils.clip_grad_norm_(bundle.capability.parameters(),5.0))
            if not math.isfinite(norm): raise FloatingPointError("nonfinite capability gradient")
            opt.step(); updates+=1; max_grad=max(max_grad,norm); losses.append(float(loss.detach()))
        metrics=validation_rank(bundle,encoder,task,val,refs["validation"])
        entry={"epoch":epoch,"loss":float(np.mean(losses)),"optimizer_updates":updates,**metrics}; hist.append(entry)
        key=(metrics["rank_hmean"],metrics["accepted_top1_accuracy"],metrics["nomatch_none_top1_accuracy"],-entry["loss"])
        if best_key is None or key>best_key: best_key=key; best=copy.deepcopy(bundle.state_dict()); sel_updates=updates; sel_epoch=epoch
    bundle.load_state_dict(best); selected=copy.deepcopy(bundle.state_dict()); sh=state_hash(selected); sch=_subhash(bundle.capability)
    if sh==ih or sch==ich or max_grad<=0: raise RuntimeError("explicit-NONE capability model did not change")
    torch.save({"state":selected,"arm":arm,"task":task,"encoder":encoder.meta,"seed":seed},directory/"selected.pt")
    receipt={"optimizer_updates":updates,"capability_optimizer_updates":updates,"selected_capability_optimizer_updates":sel_updates,"selected_epoch":sel_epoch,"max_gradient_norm":max_grad,"initial_state_hash":ih,"selected_state_hash":sh,"initial_capability_hash":ich,"selected_capability_hash":sch,"train_clause_examples":len(examples),"independent_backbone_finetuning":False,"adapter_location":"frozen-embedding feature space","risk_training_kind":"explicit NONE candidate learned jointly with listwise capability scores; status thresholds validation-calibrated","none_balance":arm["none_balance"],"contrastive_margin":bool(arm["contrast"])}
    write_json(directory/"training.json",{"history":hist,**receipt}); return receipt

def clause_evidence(bundle, encoder, item, online=False):
    out=[]
    bundle.eval()
    with torch.no_grad():
        for clause in clauses(item):
            x,opts,raw=features_with_none(encoder,item,clause,online=online); logits=bundle.capability(x); probs=torch.softmax(logits,-1)
            none_i=len(opts)-1; real_logits=logits[:-1]; order=torch.argsort(real_logits,descending=True,stable=True); a=int(order[0]); b=int(order[1]) if len(order)>1 else a
            out.append({"clause":clause,"best_real_id":opts[a]["id"],"second_real_id":opts[b]["id"],"none_logit":float(logits[none_i]),"best_real_logit":float(logits[a]),"second_real_logit":float(logits[b]),"none_prob":float(probs[none_i]),"best_real_prob":float(probs[a]),"none_margin":float(logits[none_i]-logits[a]),"real_margin":float(logits[a]-logits[b]),"raw_best":float(raw[a])})
    return out

def request_from_evidence(evidence, calibration):
    statuses=[]; selected=[]
    nt=float(calibration["none_margin_threshold"]); at=float(calibration["ambiguity_margin_threshold"])
    for e in evidence:
        if e["none_margin"]>=nt: s="NO_MATCH"
        elif e["real_margin"]<=at: s="AMBIGUOUS"
        else: s="accepted"; selected.append(e["best_real_id"])
        statuses.append(s)
    if any(s=="NO_MATCH" for s in statuses): status="NO_MATCH"; selected=[]
    elif any(s=="AMBIGUOUS" for s in statuses): status="AMBIGUOUS"; selected=[]
    else: status="accepted"; selected=sorted(set(selected))
    return {"status":status,"selected":selected,"clause_statuses":statuses}

def infer_null(bundle,encoder,item,calibration,online=False):
    evidence=clause_evidence(bundle,encoder,item,online=online); pred=request_from_evidence(evidence,calibration); pred.update({"clause_evidence":evidence,"calibration":calibration}); return pred

def _exact(pred,ref): return pred["status"]==ref["status"] and (ref["status"]!="accepted" or set(pred.get("selected",[]))==set(ref["paths"]))

def _threshold_candidates(vals): return g48._threshold_candidates(vals)

def _cal_metrics(records,nt,at):
    ac=atot=rc=rtot=exact=incorrect=0
    for r in records:
        pred=request_from_evidence(r["evidence"],{"none_margin_threshold":nt,"ambiguity_margin_threshold":at}); ref=r["reference"]; ok=_exact(pred,ref); exact+=int(ok)
        if ref["status"]=="accepted": atot+=1; ac+=int(pred["status"]=="accepted")
        else: rtot+=1; rc+=int(pred["status"]==ref["status"])
        incorrect+=int(pred["status"]=="accepted" and not ok)
    ar=ac/atot if atot else 0.; rr=rc/rtot if rtot else 0.; hm=2*ar*rr/(ar+rr) if ar+rr else 0.
    return {"accepted_status_recall":ar,"risk_accuracy":rr,"selective_hmean":hm,"exact_accuracy":exact/len(records) if records else 0.,"incorrect_publication_rate":incorrect/len(records) if records else 0.,"examples":len(records)}

def calibrate_null(bundle,encoder,rows,refs,fixed_none=None,scope="expanded-validation-only"):
    records=[{"reference":refs[r["id"]],"evidence":clause_evidence(bundle,encoder,r)} for r in rows]
    nvals=[e["none_margin"] for r in records for e in r["evidence"]]; avals=[e["real_margin"] for r in records for e in r["evidence"]]
    nts=[float(fixed_none)] if fixed_none is not None else _threshold_candidates(nvals); ats=_threshold_candidates(avals)
    best=None
    for nt in nts:
        for at in ats:
            m=_cal_metrics(records,nt,at); key=(m["selective_hmean"],m["exact_accuracy"],-m["incorrect_publication_rate"],m["accepted_status_recall"])
            if best is None or key>best[0]: best=(key,nt,at,m)
    cal={"mode":"explicit-none-margins","none_margin_threshold":best[1],"ambiguity_margin_threshold":best[2],"selection_split":"validation","calibration_scope":scope}
    return cal,best[3]

def _make_case(task,entity,root,phrase,status,targets=None,remove_suffix=None,tag="case"):
    base=catalog(entity,root); options=copy.deepcopy(base["options"])
    if remove_suffix is not None:
        rid=_option_for_suffix(options,remove_suffix)["id"]; options=[o for o in options if o["id"]!=rid]
    request=task_prefix(task)+phrase+f" for the {root}."; uid=sha(["gdm50",tag,task,entity,phrase,status,remove_suffix])[:24]; random.Random(uid).shuffle(options)
    row={"id":uid,"task":task,"request":request,"catalog":_copy_catalog(base,options)}; paths=[]
    if targets: paths=[_option_for_suffix(base["options"],s)["id"] for s in targets]
    return row,{"status":status,"paths":paths,"fixture_seeds":[13,47,101]}

def _dataset(task,domains,language,unsupported,ambiguous,tag):
    accepted_combos=[*((i,) for i in range(8)),(0,1),(1,2),(4,5),(3,4),(0,6),(3,7),(0,3,4),(1,2,6),(4,5,7),(0,1,6)]
    composite_missing=[(0,1),(1,2),(4,5),(3,7)]; rows=[]; refs={}
    for entity,root in domains:
        base=catalog(entity,root)
        for combo in accepted_combos:
            phrase="; ".join(language[i] for i in combo); request=task_prefix(task)+phrase+f" for the {root}."; uid=sha(["gdm50",tag,task,entity,combo,phrase,"accepted"])[:24]; opts=copy.deepcopy(base["options"]); random.Random(uid).shuffle(opts)
            row={"id":uid,"task":task,"request":request,"catalog":_copy_catalog(base,opts)}; rows.append(row); refs[uid]={"status":"accepted","paths":[_option_for_suffix(base["options"],SEMANTIC_SUFFIXES[i])["id"] for i in combo],"fixture_seeds":[13,47,101]}
        for i,suffix in enumerate(SEMANTIC_SUFFIXES):
            row,ref=_make_case(task,entity,root,language[i],"NO_MATCH",remove_suffix=suffix,tag=tag+"-missing"); rows.append(row); refs[row["id"]]=ref
        for p in unsupported:
            row,ref=_make_case(task,entity,root,p,"NO_MATCH",tag=tag+"-unsupported"); rows.append(row); refs[row["id"]]=ref
        for present,missing in composite_missing:
            opts=copy.deepcopy(base["options"]); rid=_option_for_suffix(opts,SEMANTIC_SUFFIXES[missing])["id"]; opts=[o for o in opts if o["id"]!=rid]; phrase=language[present]+"; "+language[missing]; request=task_prefix(task)+phrase+f" for the {root}."; uid=sha(["gdm50",tag,task,entity,phrase,"composite-missing",missing])[:24]; random.Random(uid).shuffle(opts)
            row={"id":uid,"task":task,"request":request,"catalog":_copy_catalog(base,opts)}; rows.append(row); refs[uid]={"status":"NO_MATCH","paths":[],"fixture_seeds":[13,47,101]}
        for p in ambiguous:
            row,ref=_make_case(task,entity,root,p,"AMBIGUOUS",tag=tag+"-ambiguous"); rows.append(row); refs[row["id"]]=ref
        for i,p in enumerate(ambiguous):
            phrase=language[i]+"; "+p; row,ref=_make_case(task,entity,root,phrase,"AMBIGUOUS",tag=tag+"-composite-ambiguous"); rows.append(row); refs[row["id"]]=ref
    return rows,refs

def calibration_set(task): return _dataset(task,[("Workspace","workspace"),("Ticket","ticket")],CALIBRATION_LANGUAGE,CALIBRATION_UNSUPPORTED,CALIBRATION_AMBIGUOUS,"calibration")
def fresh_holdout(task): return _dataset(task,[("Subscription","subscription"),("Claim","claim")],HOLDOUT_LANGUAGE,HOLDOUT_UNSUPPORTED,HOLDOUT_AMBIGUOUS,"holdout")

def run_one(arm_name,task,seed,encoder,public,refs,data_report,compositor,root,epochs):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); arm=ARMS[arm_name]; directory=root/f"{task}-{arm_name}-{encoder.name}-seed{seed}"; directory.mkdir(parents=True,exist_ok=False)
    config={"id":directory.name,"arm":arm_name,"task":task,"backbone":encoder.name,**arm,"model_scope":"frozen pretrained encoder + learned feature-space adapter/head","schema_scope":"catalog projection + deterministic SDL/Federation realization" if task=="schema" else None,"research_question":"can an explicit NONE listwise candidate recover open-set risk without GDM49 pairwise recall collapse"}; write_json(directory/"config.json",config)
    rss_before=current_rss_kib(); started=time.perf_counter_ns(); cal_rows,cal_refs=calibration_set(task)
    if arm["family"]=="gdm49-listwise-control":
        bundle=g47.Bundle(encoder.dim*5+4,encoder.dim); control={"gate":"clause-hybrid","hardneg":True}; receipt,calibration=g48.train(bundle,encoder,control,task,public,refs,seed,epochs,directory); receipt["gdm50_family"]=arm["family"]; infer=lambda item,online=False:g48.infer(bundle,encoder,control,item,calibration,online=online)
    else:
        bundle=NullBundle(encoder.dim*5+4); receipt=train_null(bundle,encoder,arm,task,public,refs,seed,epochs,directory)
        val=[r for r in public["validation"] if r["task"]==task]; vrefs={r["id"]:refs["validation"][r["id"]] for r in val}; rows=val+cal_rows; merged={**vrefs,**cal_refs}
        calibration,cm=calibrate_null(bundle,encoder,rows,merged,fixed_none=arm["fixed_none_threshold"]); write_json(directory/"calibration.json",{"selected":calibration,"validation_metrics":cm,"status_optimizer_updates":0}); receipt.update({"gdm50_family":arm["family"],"selected_calibration":calibration,"selected_calibration_validation_metrics":cm,"calibration_optimizer_updates":0}); tr=json.loads((directory/"training.json").read_text()); tr.update(receipt); write_json(directory/"training.json",tr); infer=lambda item,online=False:infer_null(bundle,encoder,item,calibration,online=online)
    training_ms=(time.perf_counter_ns()-started)/1e6
    if arm["family"]=="gdm49-listwise-control": tr=json.loads((directory/"training.json").read_text()); tr.update({"gdm50_family":receipt["gdm50_family"]}); write_json(directory/"training.json",tr)
    base=[r for r in public["test"] if r["task"]==task]; bref={r["id"]:refs["test"][r["id"]] for r in base}; reg_rows=list(base); reg_refs=dict(bref)
    for mod in (g46,g47,g48,g49):
        rr,rf=mod.fresh_holdout(task); reg_rows+=rr; reg_refs.update(rf)
    hold_rows,hold_refs=fresh_holdout(task)
    regression=g49.evaluate(reg_rows,reg_refs,lambda x:infer(x,False),compositor,directory/"predictions-regression.jsonl"); holdout=g49.evaluate(hold_rows,hold_refs,lambda x:infer(x,False),compositor,directory/"predictions-holdout.jsonl")
    bench=hold_rows[:12]
    for row in bench[:2]: validate_emission(row,infer(row,True))
    samples=[]; before=encoder.calls
    for _ in range(2):
        for row in bench:
            t=time.perf_counter_ns(); pred=infer(row,True); validate_emission(row,pred); samples.append((time.perf_counter_ns()-t)/1e6)
    calls=encoder.calls-before; write_json(directory/"timings.json",{"generation_ms":samples,"warmup_requests":min(2,len(bench)),"encoder_calls_measured":calls,"scope":"single-request clause query encoding + explicit-NONE/listwise gating + generation/validation; cached catalog; excludes Rover/backend"})
    gc.collect(); rss_after=current_rss_kib(); files=["config.json","training.json","calibration.json","initial.pt","selected.pt","predictions-regression.jsonl","predictions-holdout.jsonl","timings.json"]
    summary={"format":FORMAT,"evidence_kind":"trained-feature-model","config":config,"source_commit":os.environ.get("GITHUB_SHA","unrecorded"),"run_id":os.environ.get("GITHUB_RUN_ID"),"seed":seed,"encoder":encoder.meta,"training":receipt,"training_and_selection_ms":training_ms,"dataset_sha256":data_report["dataset_sha256"],"dataset_scope":data_report["scope"],"calibration_only_sha256":sha({"rows":cal_rows,"references":cal_refs}),"calibration_scope":"disjoint Workspace/Ticket calibration-only cases" if arm["family"]=="explicit-none-listwise" else "not used by control; control uses public validation","secondary_holdout_sha256":sha({"rows":hold_rows,"references":hold_refs}),"secondary_holdout_scope":"new GDM50 Subscription/Claim synthetic holdout; never used for optimizer/checkpoint/calibration selection","regression_scope":"corrected public test plus inspected GDM46-GDM49 holdouts","regression_examples":len(regression),"secondary_holdout_examples":len(holdout),"regression_metrics":aggregate(regression),"secondary_holdout_metrics":aggregate(holdout),"regression_clause_metrics":g49.clause_metrics(regression),"secondary_holdout_clause_metrics":g49.clause_metrics(holdout),"generation_latency":timing(samples),"latency_scope":"online query encoding + clause scoring/status aggregation + generation/validation; cached catalog vectors","memory":{"rss_before_training_kib":rss_before,"rss_after_evaluation_kib":rss_after,"peak_worker_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"scope":"current process RSS plus worker high-water; worker reuses frozen backbone across arms"},"file_hashes":{n:file_hash(directory/n) for n in files}}; write_json(directory/"summary.json",summary)
    print(json.dumps({"event":"completed_arm","config":config["id"],"holdout":summary["secondary_holdout_metrics"][task],"clauses":summary["secondary_holdout_clause_metrics"],"p50_ms":summary["generation_latency"]["p50_ms"]}),flush=True); return directory.name

def main():
    p=argparse.ArgumentParser(); p.add_argument("--backbone",default="distilbert",choices=["distilbert","hash"]); p.add_argument("--task",required=True,choices=["operation","schema"]); p.add_argument("--arms",default=",".join(ARMS)); p.add_argument("--seeds",default=SEEDS_DEFAULT); p.add_argument("--epochs",type=int,default=12); p.add_argument("--output",default="artifacts/results"); p.add_argument("--smoke",action="store_true"); a=p.parse_args()
    if os.environ.get("GITHUB_ACTIONS")!="true": raise RuntimeError("Project execution belongs in GitHub Actions")
    torch.set_num_threads(2); torch.set_num_interop_threads(1); root=Path(a.output); root.mkdir(parents=True,exist_ok=True); public,refs,report=corrected_dataset(root/"dataset"); arms=[x for x in a.arms.split(",") if x]; unknown=set(arms)-set(ARMS)
    if unknown: raise ValueError(sorted(unknown))
    seeds=[int(x) for x in a.seeds.split(",") if x]
    if a.smoke: arms=["null-calibrated"]; seeds=[5001]; a.epochs=2
    write_json(root/"plan.json",{"backbone":a.backbone,"task":a.task,"arms":arms,"seeds":seeds,"epochs":a.epochs}); encoder=Encoder(a.backbone); compositor=Compositor(root/"composition")
    control=next(r for r in public["train"] if r["task"]=="schema" and refs["train"][r["id"]]["status"]=="accepted"); paths=[o["id"] for o in control["catalog"]["options"]]
    if not compositor.compose(control,paths)["success"]: raise RuntimeError("Known-valid Federation composition control failed")
    completed=[]; errors=[]
    for arm in arms:
        for seed in seeds:
            try: completed.append(run_one(arm,a.task,seed,encoder,public,refs,report,compositor,root,a.epochs))
            except Exception:
                err={"arm":arm,"task":a.task,"seed":seed,"traceback":traceback.format_exc()}; errors.append(err); print(json.dumps({"event":"failed_arm",**err}),flush=True)
    write_json(root/"worker.json",{"complete":not errors,"completed":completed,"errors":errors,"expected":len(arms)*len(seeds),"encoder":encoder.meta})
    if errors: raise SystemExit(1)

if __name__=="__main__": main()
