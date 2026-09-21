"""GDM51: explicit NONE for open-set NO_MATCH plus a separate ambiguity discriminator.

GDM50 verified that an explicit NONE candidate improves NO_MATCH behavior without the
severe pairwise recall collapse from GDM49, but AMBIGUOUS detection remained weak and
multi-clause answerable recall still degraded. GDM51 keeps the frozen DistilBERT
encoder, listwise capability head, and explicit NONE mechanism fixed. It replaces the
single top-two-margin ambiguity threshold with a task-local discriminator trained only
on training data and calibrated only on validation/calibration data.

No transformer fine-tuning occurs. Learned modules operate over frozen feature-space
representations. Schema generation remains catalog projection plus deterministic
SDL/Federation realization. All execution belongs in GitHub Actions.
"""
from __future__ import annotations

import argparse, copy, gc, json, math, os, random, resource, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.measured.contracts import catalog, option_text, sha
from experiments.measured.evidence import Compositor, aggregate, file_hash, timing, write_json
from experiments.measured.models import Encoder, Head, state_hash
from experiments.followup45.run import clauses
from experiments.followup46 import run as g46
from experiments.followup46.run import corrected_dataset, current_rss_kib, task_prefix, validate_emission
from experiments.followup47 import run as g47
from experiments.followup48 import run as g48
from experiments.followup49 import run as g49
from experiments.followup50 import run as g50
from experiments.followup50.execute import features_with_none_cached

# Preserve the cache-corrected GDM50 timing boundary in every GDM51 path.
g50.features_with_none = features_with_none_cached

FORMAT = "gdm51-measured-v1"
SEEDS_DEFAULT = "5101,5102"

ARMS = {
    "null-margin-control": {"family":"gdm50-null-margin-control", "head":"margin", "structured":False, "ambiguous_weight":0.0, "hard_negative":False},
    "ambiguity-linear": {"family":"explicit-none-plus-ambiguity", "head":"linear", "structured":False, "ambiguous_weight":1.0, "hard_negative":False},
    "ambiguity-mlp": {"family":"explicit-none-plus-ambiguity", "head":"mlp", "structured":False, "ambiguous_weight":1.0, "hard_negative":False},
    "ambiguity-structured": {"family":"explicit-none-plus-ambiguity", "head":"mlp", "structured":True, "ambiguous_weight":1.0, "hard_negative":False},
    "ambiguity-structured-hard": {"family":"explicit-none-plus-ambiguity", "head":"mlp", "structured":True, "ambiguous_weight":1.75, "hard_negative":True},
}

SEMANTIC_SUFFIXES = g47.SEMANTIC_SUFFIXES

CALIBRATION_LANGUAGE = [
    "customer-facing label for this object",
    "time the object first entered durable storage",
    "time durable storage most recently changed the object",
    "numeric review score",
    "display names of review authors",
    "display names of review moderators",
    "legal name of the supplying company",
    "stable IDs of review authors",
]
CALIBRATION_UNSUPPORTED = [
    "internal warehouse bin for the object",
    "refund explanation for the object",
    "tax jurisdiction code for the object",
    "flag saying the object is export controlled",
]
CALIBRATION_AMBIGUOUS = [
    "review participant display name without saying author or moderator",
    "object write timestamp without saying first or latest write",
]

HOLDOUT_LANGUAGE = [
    "caption presented to users for this item",
    "instant when the item was first persisted",
    "instant when the item was most recently rewritten",
    "whole-number rating attached to each review",
    "names of the people who authored each review",
    "names of the people who moderated each review",
    "registered company name for the item's supplier",
    "identifiers of the people who authored each review",
]
HOLDOUT_UNSUPPORTED = [
    "rack position where the item is physically stored",
    "reason the item was cancelled",
    "customs fee charged for the item",
    "flag saying the item requires manual inspection",
]
HOLDOUT_AMBIGUOUS = [
    "review participant identity without specifying author versus moderator",
    "item persistence time without specifying initial versus latest write",
]

AMBIGUITY_TEMPLATES = [
    (("createdAt",),("updatedAt",),[
        "record timestamp without distinguishing initial creation from latest revision",
        "when the record changed without saying first write or newest write",
        "record write time with creation versus revision left unspecified",
    ]),
    (("reviews","author","name"),("reviews","moderator","name"),[
        "review participant name without saying author or moderator",
        "name of the review person with writer versus moderator unspecified",
        "review person's display name without a role distinction",
    ]),
    (("reviews","author","name"),("reviews","author","id"),[
        "review author identity without saying name or identifier",
        "review writer identity with name versus ID unspecified",
        "author identity field without choosing display name or stable ID",
    ]),
]

class AmbiguityHead(nn.Module):
    def __init__(self, dim:int, kind:str):
        super().__init__()
        self.kind = kind
        if kind == "linear": self.net = nn.Linear(dim, 1)
        elif kind == "mlp": self.net = nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, 1))
        else: raise ValueError(kind)
    def forward(self, x): return self.net(x).squeeze(-1)

class Bundle(nn.Module):
    def __init__(self, pair_dim:int, ambiguity_dim:int, kind:str):
        super().__init__()
        self.capability = Head(pair_dim, adapter=True)
        self.ambiguity = AmbiguityHead(ambiguity_dim, kind)

def _subhash(module): return state_hash(copy.deepcopy(module.state_dict()))

def _suffix(option): return tuple(option["path"][1:])

def _option_for_suffix(options, suffix):
    m=[o for o in options if _suffix(o)==tuple(suffix)]
    if len(m)!=1: raise AssertionError((suffix,len(m)))
    return m[0]

def _ambiguity_dim(structured:bool): return 11 if structured else 7

def clause_state(bundle, encoder, item, clause, structured=False, online=False):
    x,opts,raw = g50.features_with_none(encoder,item,clause,online=online)
    logits = bundle.capability(x)
    probs = torch.softmax(logits, -1)
    real_logits = logits[:-1]
    order = torch.argsort(real_logits, descending=True, stable=True)
    a = int(order[0]); b = int(order[1]) if len(order)>1 else a
    none_i = len(opts)-1
    base = [
        float(logits[none_i]-logits[a]),
        float(logits[a]-logits[b]),
        float(probs[a]),
        float(probs[b]),
        float(probs[none_i]),
        float(raw[a]),
        float(raw[b]),
    ]
    if structured:
        dim = encoder.dim
        c1 = x[a, dim:2*dim]; c2 = x[b, dim:2*dim]
        cos = float(F.cosine_similarity(c1.unsqueeze(0),c2.unsqueeze(0))[0])
        p1=opts[a]["path"]; p2=opts[b]["path"]
        same_parent=float(p1[:-1]==p2[:-1])
        same_depth=float(len(p1)==len(p2))
        same_leaf_kind=float((p1[-1] in {"name","id"})==(p2[-1] in {"name","id"}))
        base += [cos,same_parent,same_depth,same_leaf_kind]
    return {
        "clause": clause,
        "best_real_id": opts[a]["id"],
        "second_real_id": opts[b]["id"],
        "none_margin": float(logits[none_i]-logits[a]),
        "real_margin": float(logits[a]-logits[b]),
        "best_real_prob": float(probs[a]),
        "second_real_prob": float(probs[b]),
        "none_prob": float(probs[none_i]),
        "raw_best": float(raw[a]),
        "raw_second": float(raw[b]),
        "ambiguity_features": base,
    }

def ambiguity_training_examples(task, public, refs):
    examples=[]
    seen_catalogs={}
    for row in public["train"]:
        if row["task"]!=task: continue
        ref=refs["train"][row["id"]]
        if ref["status"]=="accepted" and len(clauses(row))==len(ref["paths"]):
            for clause,target in zip(clauses(row),ref["paths"]): examples.append((row,clause,0,target,"accepted"))
        root=row["catalog"]["root"]
        seen_catalogs.setdefault(root,row)
    for row in seen_catalogs.values():
        opts=row["catalog"]["options"]
        available={_suffix(o) for o in opts}
        for left,right,templates in AMBIGUITY_TEMPLATES:
            if tuple(left) not in available or tuple(right) not in available: continue
            for phrase in templates:
                examples.append((row,phrase,1,None,"synthetic-ambiguous"))
    return examples

def validation_ambiguity_examples(task, public, refs):
    out=[]
    for row in public["validation"]:
        if row["task"]!=task: continue
        ref=refs["validation"][row["id"]]
        if ref["status"]=="accepted" and len(clauses(row))==len(ref["paths"]):
            for clause,target in zip(clauses(row),ref["paths"]): out.append((row,clause,0,target))
        elif ref["status"]=="AMBIGUOUS":
            for clause in clauses(row): out.append((row,clause,1,None))
    return out

def train_model(bundle, encoder, arm, task, public, refs, seed, epochs, directory):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    initial=copy.deepcopy(bundle.state_dict()); ih=state_hash(initial); ich=_subhash(bundle.capability); iah=_subhash(bundle.ambiguity)
    torch.save({"state":initial,"arm":arm,"task":task},directory/"initial.pt")

    # Stage 1: explicit-NONE listwise capability model, selected only on public validation.
    cap_examples=g50.training_examples(task,public,refs)
    val=[r for r in public["validation"] if r["task"]==task]
    cap_opt=torch.optim.AdamW(bundle.capability.parameters(),lr=8e-4,weight_decay=1e-3)
    best_cap=None; best_cap_key=None; cap_updates=sel_cap_updates=sel_cap_epoch=0; max_cap_grad=0.0; cap_hist=[]
    cap_arm={"none_balance":1.0,"contrast":False}
    for epoch in range(1,epochs+1):
        order=list(cap_examples); random.Random(seed+epoch).shuffle(order); losses=[]; bundle.train()
        for row,clause,positives,source in order:
            cap_opt.zero_grad(set_to_none=True); loss=g50.null_loss(bundle,encoder,row,clause,positives,cap_arm); loss.backward()
            norm=float(torch.nn.utils.clip_grad_norm_(bundle.capability.parameters(),5.0))
            if not math.isfinite(norm): raise FloatingPointError("nonfinite capability gradient")
            cap_opt.step(); cap_updates+=1; max_cap_grad=max(max_cap_grad,norm); losses.append(float(loss.detach()))
        metrics=g50.validation_rank(bundle,encoder,task,val,refs["validation"])
        cap_hist.append({"epoch":epoch,"loss":float(np.mean(losses)),"optimizer_updates":cap_updates,**metrics})
        key=(metrics["rank_hmean"],metrics["accepted_top1_accuracy"],metrics["nomatch_none_top1_accuracy"],-cap_hist[-1]["loss"])
        if best_cap_key is None or key>best_cap_key:
            best_cap_key=key; best_cap=copy.deepcopy(bundle.capability.state_dict()); sel_cap_updates=cap_updates; sel_cap_epoch=epoch
    bundle.capability.load_state_dict(best_cap)

    # Stage 2: task-local top-two ambiguity discriminator. Capability weights are frozen.
    amb_examples=ambiguity_training_examples(task,public,refs)
    val_examples=validation_ambiguity_examples(task,public,refs)
    amb_opt=torch.optim.AdamW(bundle.ambiguity.parameters(),lr=1.2e-3,weight_decay=1e-3)
    best_amb=None; best_amb_key=None; amb_updates=sel_amb_updates=sel_amb_epoch=0; max_amb_grad=0.0; amb_hist=[]
    for epoch in range(1,epochs+1):
        order=list(amb_examples); random.Random(seed+1000+epoch).shuffle(order); losses=[]; bundle.train()
        for row,clause,label,target,source in order:
            with torch.no_grad(): st=clause_state(bundle,encoder,row,clause,structured=arm["structured"])
            feat=torch.tensor(st["ambiguity_features"],dtype=torch.float32).unsqueeze(0)
            y=torch.tensor([float(label)],dtype=torch.float32)
            amb_opt.zero_grad(set_to_none=True); logit=bundle.ambiguity(feat)
            loss=F.binary_cross_entropy_with_logits(logit,y,reduction="none")
            weight=float(arm["ambiguous_weight"] if label else 1.0)
            if arm["hard_negative"] and not label and st["real_margin"]<0.6: weight*=1.75
            loss=(loss*weight).mean(); loss.backward()
            norm=float(torch.nn.utils.clip_grad_norm_(bundle.ambiguity.parameters(),5.0))
            if not math.isfinite(norm): raise FloatingPointError("nonfinite ambiguity gradient")
            amb_opt.step(); amb_updates+=1; max_amb_grad=max(max_amb_grad,norm); losses.append(float(loss.detach()))
        tp=tn=fp=fn=0; bundle.eval()
        with torch.no_grad():
            for row,clause,label,target in val_examples:
                st=clause_state(bundle,encoder,row,clause,structured=arm["structured"])
                p=float(torch.sigmoid(bundle.ambiguity(torch.tensor(st["ambiguity_features"],dtype=torch.float32).unsqueeze(0)))[0]); pred=int(p>=0.5)
                if label and pred: tp+=1
                elif label and not pred: fn+=1
                elif not label and pred: fp+=1
                else: tn+=1
        tpr=tp/(tp+fn) if tp+fn else 0.0; tnr=tn/(tn+fp) if tn+fp else 0.0; bal=0.5*(tpr+tnr)
        amb_hist.append({"epoch":epoch,"loss":float(np.mean(losses)),"optimizer_updates":amb_updates,"validation_balanced_accuracy":bal,"validation_tpr":tpr,"validation_tnr":tnr,"validation_examples":len(val_examples)})
        key=(bal,tpr,tnr,-amb_hist[-1]["loss"])
        if best_amb_key is None or key>best_amb_key:
            best_amb_key=key; best_amb=copy.deepcopy(bundle.ambiguity.state_dict()); sel_amb_updates=amb_updates; sel_amb_epoch=epoch
    bundle.ambiguity.load_state_dict(best_amb)

    selected=copy.deepcopy(bundle.state_dict()); sh=state_hash(selected); sch=_subhash(bundle.capability); sah=_subhash(bundle.ambiguity)
    if sh==ih or sch==ich or sah==iah or max_cap_grad<=0 or max_amb_grad<=0: raise RuntimeError("GDM51 learned modules did not change")
    torch.save({"state":selected,"arm":arm,"task":task,"encoder":encoder.meta,"seed":seed},directory/"selected.pt")
    receipt={
        "optimizer_updates":cap_updates+amb_updates,
        "capability_optimizer_updates":cap_updates,
        "ambiguity_optimizer_updates":amb_updates,
        "selected_capability_optimizer_updates":sel_cap_updates,
        "selected_ambiguity_optimizer_updates":sel_amb_updates,
        "selected_capability_epoch":sel_cap_epoch,
        "selected_ambiguity_epoch":sel_amb_epoch,
        "max_capability_gradient_norm":max_cap_grad,
        "max_ambiguity_gradient_norm":max_amb_grad,
        "initial_state_hash":ih,"selected_state_hash":sh,
        "initial_capability_hash":ich,"selected_capability_hash":sch,
        "initial_ambiguity_hash":iah,"selected_ambiguity_hash":sah,
        "train_clause_examples":len(cap_examples),"ambiguity_train_examples":len(amb_examples),"ambiguity_validation_examples":len(val_examples),
        "independent_backbone_finetuning":False,"adapter_location":"frozen-embedding feature space",
        "risk_training_kind":"explicit NONE listwise capability model plus separately trained top-two ambiguity discriminator; request thresholds validation-calibrated",
        "ambiguity_head":arm["head"],"ambiguity_structured_features":bool(arm["structured"]),"ambiguity_positive_weight":float(arm["ambiguous_weight"]),"ambiguity_hard_negative_weighting":bool(arm["hard_negative"]),
    }
    write_json(directory/"training.json",{"capability_history":cap_hist,"ambiguity_history":amb_hist,**receipt})
    return receipt

def evidence_for_request(bundle,encoder,item,arm,online=False):
    out=[]; bundle.eval()
    with torch.no_grad():
        for clause in clauses(item):
            st=clause_state(bundle,encoder,item,clause,structured=arm["structured"],online=online)
            feat=torch.tensor(st["ambiguity_features"],dtype=torch.float32).unsqueeze(0)
            st["ambiguity_probability"]=float(torch.sigmoid(bundle.ambiguity(feat))[0]); out.append(st)
    return out

def request_from_evidence(evidence, calibration, learned=True):
    statuses=[]; selected=[]; nt=float(calibration["none_margin_threshold"])
    at=float(calibration["ambiguity_probability_threshold"] if learned else calibration["ambiguity_margin_threshold"])
    for e in evidence:
        if e["none_margin"]>=nt: s="NO_MATCH"
        elif (e["ambiguity_probability"]>=at if learned else e["real_margin"]<=at): s="AMBIGUOUS"
        else: s="accepted"; selected.append(e["best_real_id"])
        statuses.append(s)
    if any(s=="NO_MATCH" for s in statuses): status="NO_MATCH"; selected=[]
    elif any(s=="AMBIGUOUS" for s in statuses): status="AMBIGUOUS"; selected=[]
    else: status="accepted"; selected=sorted(set(selected))
    return {"status":status,"selected":selected,"clause_statuses":statuses}

def infer_model(bundle,encoder,item,arm,calibration,online=False):
    ev=evidence_for_request(bundle,encoder,item,arm,online=online); pred=request_from_evidence(ev,calibration,learned=True); pred.update({"clause_evidence":ev,"calibration":calibration}); return pred

def _exact(pred,ref): return pred["status"]==ref["status"] and (ref["status"]!="accepted" or set(pred.get("selected",[]))==set(ref["paths"]))

def _threshold_candidates(vals): return g48._threshold_candidates(vals)

def _cal_metrics(records,nt,at,learned=True):
    ac=atot=rc=rtot=exact=incorrect=0
    for r in records:
        pred=request_from_evidence(r["evidence"],{"none_margin_threshold":nt,"ambiguity_probability_threshold":at,"ambiguity_margin_threshold":at},learned=learned); ref=r["reference"]; ok=_exact(pred,ref); exact+=int(ok)
        if ref["status"]=="accepted": atot+=1; ac+=int(pred["status"]=="accepted")
        else: rtot+=1; rc+=int(pred["status"]==ref["status"])
        incorrect+=int(pred["status"]=="accepted" and not ok)
    ar=ac/atot if atot else 0.; rr=rc/rtot if rtot else 0.; hm=2*ar*rr/(ar+rr) if ar+rr else 0.
    return {"accepted_status_recall":ar,"risk_accuracy":rr,"selective_hmean":hm,"exact_accuracy":exact/len(records) if records else 0.,"incorrect_publication_rate":incorrect/len(records) if records else 0.,"examples":len(records)}

def calibrate_model(bundle,encoder,arm,rows,refs):
    records=[{"reference":refs[r["id"]],"evidence":evidence_for_request(bundle,encoder,r,arm)} for r in rows]
    nvals=[e["none_margin"] for r in records for e in r["evidence"]]; avals=[e["ambiguity_probability"] for r in records for e in r["evidence"]]
    best=None
    for nt in _threshold_candidates(nvals):
        for at in _threshold_candidates(avals):
            m=_cal_metrics(records,nt,at,True); key=(m["selective_hmean"],m["exact_accuracy"],-m["incorrect_publication_rate"],m["accepted_status_recall"])
            if best is None or key>best[0]: best=(key,nt,at,m)
    cal={"mode":"explicit-none-plus-ambiguity-discriminator","none_margin_threshold":best[1],"ambiguity_probability_threshold":best[2],"selection_split":"validation","calibration_scope":"expanded-validation-only"}
    return cal,best[3]

def calibration_set(task):
    return g50._dataset(task,[("Project","project"),("Audit","audit")],CALIBRATION_LANGUAGE,CALIBRATION_UNSUPPORTED,CALIBRATION_AMBIGUOUS,"gdm51-calibration")

def fresh_holdout(task):
    return g50._dataset(task,[("Incident","incident"),("Contract","contract")],HOLDOUT_LANGUAGE,HOLDOUT_UNSUPPORTED,HOLDOUT_AMBIGUOUS,"gdm51-holdout")

def run_one(arm_name,task,seed,encoder,public,refs,data_report,compositor,root,epochs):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); arm=ARMS[arm_name]
    directory=root/f"{task}-{arm_name}-{encoder.name}-seed{seed}"; directory.mkdir(parents=True,exist_ok=False)
    config={"id":directory.name,"arm":arm_name,"task":task,"backbone":encoder.name,**arm,"model_scope":"frozen pretrained encoder + learned feature-space adapter/head","schema_scope":"catalog projection + deterministic SDL/Federation realization" if task=="schema" else None,"research_question":"can explicit NONE handle NO_MATCH while a separate top-two discriminator improves AMBIGUOUS without sacrificing multi-clause answerable recall"}
    write_json(directory/"config.json",config); rss_before=current_rss_kib(); started=time.perf_counter_ns(); cal_rows,cal_refs=calibration_set(task)

    if arm["family"]=="gdm50-null-margin-control":
        bundle=g50.NullBundle(encoder.dim*5+4); control=g50.ARMS["null-calibrated"]; receipt=g50.train_null(bundle,encoder,control,task,public,refs,seed,epochs,directory)
        val=[r for r in public["validation"] if r["task"]==task]; vrefs={r["id"]:refs["validation"][r["id"]] for r in val}; rows=val+cal_rows; merged={**vrefs,**cal_refs}
        calibration,cm=g50.calibrate_null(bundle,encoder,rows,merged,fixed_none=None); write_json(directory/"calibration.json",{"selected":calibration,"validation_metrics":cm,"status_optimizer_updates":0})
        receipt.update({"gdm51_family":arm["family"],"selected_calibration":calibration,"selected_calibration_validation_metrics":cm,"calibration_optimizer_updates":0})
        tr=json.loads((directory/"training.json").read_text()); tr.update(receipt); write_json(directory/"training.json",tr)
        infer=lambda item,online=False:g50.infer_null(bundle,encoder,item,calibration,online=online)
    else:
        bundle=Bundle(encoder.dim*5+4,_ambiguity_dim(arm["structured"]),arm["head"]); receipt=train_model(bundle,encoder,arm,task,public,refs,seed,epochs,directory)
        val=[r for r in public["validation"] if r["task"]==task]; vrefs={r["id"]:refs["validation"][r["id"]] for r in val}; rows=val+cal_rows; merged={**vrefs,**cal_refs}
        calibration,cm=calibrate_model(bundle,encoder,arm,rows,merged); write_json(directory/"calibration.json",{"selected":calibration,"validation_metrics":cm,"status_optimizer_updates":0})
        receipt.update({"gdm51_family":arm["family"],"selected_calibration":calibration,"selected_calibration_validation_metrics":cm,"calibration_optimizer_updates":0})
        tr=json.loads((directory/"training.json").read_text()); tr.update(receipt); write_json(directory/"training.json",tr)
        infer=lambda item,online=False:infer_model(bundle,encoder,item,arm,calibration,online=online)

    training_ms=(time.perf_counter_ns()-started)/1e6
    if arm["family"]=="gdm50-null-margin-control":
        tr=json.loads((directory/"training.json").read_text()); tr.update({"gdm51_family":receipt["gdm51_family"]}); write_json(directory/"training.json",tr)

    base=[r for r in public["test"] if r["task"]==task]; bref={r["id"]:refs["test"][r["id"]] for r in base}; reg_rows=list(base); reg_refs=dict(bref)
    for mod in (g46,g47,g48,g49,g50):
        rr,rf=mod.fresh_holdout(task); reg_rows+=rr; reg_refs.update(rf)
    hold_rows,hold_refs=fresh_holdout(task)
    regression=g49.evaluate(reg_rows,reg_refs,lambda x:infer(x,False),compositor,directory/"predictions-regression.jsonl")
    holdout=g49.evaluate(hold_rows,hold_refs,lambda x:infer(x,False),compositor,directory/"predictions-holdout.jsonl")

    bench=hold_rows[:12]
    for row in bench[:2]: validate_emission(row,infer(row,True))
    samples=[]; before=encoder.calls
    for _ in range(2):
        for row in bench:
            t=time.perf_counter_ns(); pred=infer(row,True); validate_emission(row,pred); samples.append((time.perf_counter_ns()-t)/1e6)
    calls=encoder.calls-before
    write_json(directory/"timings.json",{"generation_ms":samples,"warmup_requests":min(2,len(bench)),"encoder_calls_measured":calls,"scope":"single-request fresh clause query encoding + cached catalog/NONE vectors + listwise capability + ambiguity discrimination + generation/validation; excludes Rover/backend"})
    gc.collect(); rss_after=current_rss_kib()
    files=["config.json","training.json","calibration.json","initial.pt","selected.pt","predictions-regression.jsonl","predictions-holdout.jsonl","timings.json"]
    summary={
        "format":FORMAT,"evidence_kind":"trained-feature-model","config":config,"source_commit":os.environ.get("GITHUB_SHA","unrecorded"),"run_id":os.environ.get("GITHUB_RUN_ID"),"seed":seed,"encoder":encoder.meta,"training":receipt,"training_and_selection_ms":training_ms,
        "dataset_sha256":data_report["dataset_sha256"],"dataset_scope":data_report["scope"],
        "calibration_only_sha256":sha({"rows":cal_rows,"references":cal_refs}),"calibration_scope":"disjoint Project/Audit calibration-only cases; never used for optimizer updates or checkpoint selection",
        "secondary_holdout_sha256":sha({"rows":hold_rows,"references":hold_refs}),"secondary_holdout_scope":"new GDM51 Incident/Contract synthetic holdout; never used for optimizer/checkpoint/calibration selection",
        "regression_scope":"corrected public test plus inspected GDM46-GDM50 holdouts","regression_examples":len(regression),"secondary_holdout_examples":len(holdout),
        "regression_metrics":aggregate(regression),"secondary_holdout_metrics":aggregate(holdout),"regression_clause_metrics":g49.clause_metrics(regression),"secondary_holdout_clause_metrics":g49.clause_metrics(holdout),
        "generation_latency":timing(samples),"latency_scope":"fresh request-clause encoding + learned feature-space scoring + deterministic generation/validation; catalog/NONE embeddings cached",
        "memory":{"rss_before_training_kib":rss_before,"rss_after_evaluation_kib":rss_after,"peak_worker_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"scope":"current process RSS plus worker high-water; worker reuses frozen backbone across arms"},
        "file_hashes":{n:file_hash(directory/n) for n in files},
    }
    write_json(directory/"summary.json",summary)
    print(json.dumps({"event":"completed_arm","config":config["id"],"holdout":summary["secondary_holdout_metrics"][task],"clauses":summary["secondary_holdout_clause_metrics"],"p50_ms":summary["generation_latency"]["p50_ms"]}),flush=True)
    return directory.name

def main():
    p=argparse.ArgumentParser(); p.add_argument("--backbone",default="distilbert",choices=["distilbert","hash"]); p.add_argument("--task",required=True,choices=["operation","schema"]); p.add_argument("--arms",default=",".join(ARMS)); p.add_argument("--seeds",default=SEEDS_DEFAULT); p.add_argument("--epochs",type=int,default=12); p.add_argument("--output",default="artifacts/results"); p.add_argument("--smoke",action="store_true"); a=p.parse_args()
    if os.environ.get("GITHUB_ACTIONS")!="true": raise RuntimeError("Project execution belongs in GitHub Actions")
    torch.set_num_threads(2); torch.set_num_interop_threads(1); root=Path(a.output); root.mkdir(parents=True,exist_ok=True); public,refs,report=corrected_dataset(root/"dataset"); arms=[x for x in a.arms.split(",") if x]; unknown=set(arms)-set(ARMS)
    if unknown: raise ValueError(sorted(unknown))
    seeds=[int(x) for x in a.seeds.split(",") if x]
    if a.smoke: arms=["ambiguity-linear"]; seeds=[5101]; a.epochs=2
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
