"""GDM65 operation-only structured schema-relation channel."""
from __future__ import annotations
import hashlib,json,re,torch
from torch.nn import functional as F
from experiments.followup50 import run as g50
from experiments.followup51 import run as g51
from experiments.followup52 import run as g52
from experiments.followup55 import run as g55
from experiments.followup56 import run as g56
from experiments.followup59 import run as g59
from experiments.followup63 import run as g63
from experiments.followup64 import run as g64

FORMAT="gdm65-measured-v1"; SEEDS_DEFAULT="6501,6502"
CANONICAL_GDM64_SOURCE="6fe0f89630324fa18c18175c1d7c64ba614a1cff"
CANONICAL_GDM64_AUDIT_RUN=36007882683
CANONICAL_GDM64_PROMOTION="fec580e4d41ad03eedd8ba182cfcbcbb3c313080"
TOPK=5; BASE_FEATURES=11; SEMANTIC_BLOCKS=10; RELATION_FEATURES=20
MODES=("structured-zero-control","structured-top1-relation","structured-set-relation","structured-type-shape-relation","structured-hybrid-relation")
REPRESENTATION_MODES=MODES
ARMS={m:{**g64._BASE,"structured":m} for m in MODES}
CALIBRATION_AMBIGUITIES=[("role","assessment participant without choosing author or moderator"),("lifecycle-time","catalog time without choosing first or latest"),("representation","assessment author identity without choosing name or identifier"),("object-vs-supplier","catalog name without choosing title or supplier")]
HOLDOUT_AMBIGUITIES=[("role","annotation participant without choosing writer or moderator"),("lifecycle-time","manuscript time without choosing first or latest"),("representation","annotation writer identity without choosing name or identifier"),("object-vs-supplier","manuscript name without choosing title or supplier")]
CALIBRATION_LANGUAGE=["catalog public title","catalog initial timestamp","catalog latest timestamp","assessment integer rating","assessment author names","assessment moderator names","catalog supplier company","assessment author identifiers"]
HOLDOUT_LANGUAGE=["manuscript public heading","manuscript original timestamp","manuscript latest timestamp","annotation integer rating","annotation writer names","annotation moderator names","manuscript supplier company","annotation writer identifiers"]
UNSUPPORTED=["physical shelf coordinate","display color code","shipping region code","storage box number"]

def ambiguity_feature_dim(d): return BASE_FEATURES+SEMANTIC_BLOCKS*int(d)+RELATION_FEATURES
def _path(o):
 p=tuple(str(x) for x in (o.get("path") or [])[1:])
 if not p: raise ValueError("real GraphQL path required")
 return p
def _parent(o): return _path(o)[:-1]
def _leaf(o): return _path(o)[-1]
def _raw_type(o):
 t=o.get("types") or []
 if not t: raise ValueError("output type required")
 return str(t[-1])
def _named(o): return re.sub(r"[\[\]!]","",_raw_type(o))
def _shape(o): return re.sub(r"[_A-Za-z][_0-9A-Za-z]*","T",_raw_type(o),count=1)
def _same_parent(a,b): return float(_parent(a)==_parent(b))
def _same_leaf(a,b): return float(_leaf(a)==_leaf(b))
def _same_named_type(a,b): return float(_named(a)==_named(b))
def _same_type_shape(a,b): return float(_shape(a)==_shape(b))
def _lcp(a,b):
 pa,pb=_path(a),_path(b); n=0
 for x,y in zip(pa,pb):
  if x!=y: break
  n+=1
 return n/max(len(pa),len(pb),1)
def _depth(a,b):
 da,db=len(_path(a)),len(_path(b)); return 1-min(abs(da-db)/max(da,db,1),1)
def _mean(opts,i,fn):
 js=[j for j in range(len(opts)) if j!=i]
 return 0.0 if not js else sum(fn(opts[i],opts[j]) for j in js)/len(js)
def _relation_channel(options,mode):
 active=list(options[:TOPK]); top=active[0] if active else None; out=[]
 for i,o in enumerate(active):
  if mode=="structured-zero-control": b=[0.,0.,0.,0.]
  elif mode=="structured-top1-relation": b=[_same_parent(o,top),_same_leaf(o,top),_same_named_type(o,top),_lcp(o,top)]
  elif mode=="structured-set-relation": b=[_mean(active,i,_same_parent),_mean(active,i,_same_leaf),_mean(active,i,_same_named_type),_mean(active,i,_lcp)]
  elif mode=="structured-type-shape-relation": b=[_same_parent(o,top),_same_leaf(o,top),_same_named_type(o,top),_same_type_shape(o,top)]
  elif mode=="structured-hybrid-relation": b=[_same_parent(o,top),_same_leaf(o,top),_mean(active,i,_lcp),_mean(active,i,_depth)]
  else: raise ValueError(mode)
  if any(x<0 or x>1 for x in b): raise AssertionError(b)
  out.extend(float(x) for x in b)
 out += [0.0]*((TOPK-len(active))*4)
 if len(out)!=20: raise AssertionError(len(out))
 return out
def _relation_hash(mode):
 x={"mode":mode,"topk":5,"primitives":["same_parent","same_leaf","same_named_type","same_type_shape","lcp_ratio","depth_similarity"],"source":"provided-schema-only"}
 return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def clause_state(bundle,encoder,item,clause,structured=False,online=False):
 mode=structured
 if mode not in MODES: return g64.g60._ORIGINAL_CLAUSE_STATE(bundle,encoder,item,clause,structured=structured,online=online)
 x,opts,raw=g50.features_with_none(encoder,item,clause,online=online); logits=bundle.capability(x); probs=torch.softmax(logits,-1)
 order=torch.argsort(logits[:-1],descending=True,stable=True); a=int(order[0]); b=int(order[1]) if len(order)>1 else a; n=len(opts)-1; d=encoder.dim; q=x[a,:d]
 c1=x[a,d:2*d]; c2=x[b,d:2*d]; p1=opts[a]["path"]; p2=opts[b]["path"]
 base=[float(logits[n]-logits[a]),float(logits[a]-logits[b]),float(probs[a]),float(probs[b]),float(probs[n]),float(raw[a]),float(raw[b]),float(F.cosine_similarity(c1[None],c2[None])[0]),float(p1[:-1]==p2[:-1]),float(len(p1)==len(p2)),float((p1[-1] in {"name","id"})==(p2[-1] in {"name","id"}))]
 idx=[int(i) for i in order[:min(TOPK,len(order))]]; rows=torch.stack([x[i,d:2*d] for i in idx]); product,delta=g59._ranked_blocks(q,rows,"ranked-request-only"); rel=_relation_channel([opts[i] for i in idx],mode)
 feats=base+product.float().tolist()+delta.float().tolist()+rel; expected=ambiguity_feature_dim(d)
 if len(feats)!=expected: raise AssertionError((len(feats),expected))
 return {"clause":clause,"best_real_id":opts[a]["id"],"second_real_id":opts[b]["id"],"none_margin":float(logits[n]-logits[a]),"real_margin":float(logits[a]-logits[b]),"best_real_prob":float(probs[a]),"second_real_prob":float(probs[b]),"none_prob":float(probs[n]),"raw_best":float(raw[a]),"raw_second":float(raw[b]),"ambiguity_features":feats,"ambiguity_representation_mode":mode,"ambiguity_feature_dim":expected,"ambiguity_topk_effective":len(idx),"ambiguity_real_candidate_count":len(order),"ambiguity_rank_preserving":True,"ambiguity_relation_features":rel}

def _receipt(bundle,mode,d):
 return {"gdm65_family":"explicit-structured-schema-relation-channel","gdm65_representation_mode":mode,"gdm65_requested_topk":5,"gdm65_base_feature_dim":11+10*d,"gdm65_relation_feature_count":20,"gdm65_ambiguity_feature_dim":ambiguity_feature_dim(d),"gdm65_base_semantics":"exact canonical ranked top5 q*ci + abs(q-ci)","gdm65_relation_contract_hash":_relation_hash(mode),"gdm65_relation_request_independent":True,"gdm65_missing_rank_relation_zero_padding":True,"gdm65_ambiguity_parameter_count":sum(p.numel() for p in bundle.ambiguity.parameters()),"canonical_gdm64_source_commit":CANONICAL_GDM64_SOURCE,"canonical_gdm64_audit_run":CANONICAL_GDM64_AUDIT_RUN,"canonical_gdm64_promotion_commit":CANONICAL_GDM64_PROMOTION}
def train_model(bundle,encoder,arm,task,public,refs,seed,epochs,directory):
 if task!="operation": raise RuntimeError("operation only")
 examples=g56.ambiguity_curriculum_examples(task,public,refs,"family-balanced"); old=g51.ambiguity_training_examples; g51.ambiguity_training_examples=lambda *_:list(examples)
 try: r=g64.g60._ORIGINAL_TRAIN_MODEL(bundle,encoder,arm,task,public,refs,seed,epochs,directory)
 finally: g51.ambiguity_training_examples=old
 r.update(g56._curriculum_receipt(examples,"family-balanced")); r.update(_receipt(bundle,arm["structured"],encoder.dim)); return r
def calibration_set(task):
 if task!="operation": raise RuntimeError("operation only")
 rows,refs=g50._dataset(task,[("Catalog","catalog"),("Ledgerbook","ledgerbook")],CALIBRATION_LANGUAGE,UNSUPPORTED,[p for _,p in CALIBRATION_AMBIGUITIES],"gdm65-calibration")
 return g55._tag_ambiguity_families(rows,refs,CALIBRATION_AMBIGUITIES)
def fresh_holdout(task):
 if task!="operation": raise RuntimeError("operation only")
 rows,refs=g50._dataset(task,[("Manuscript","manuscript"),("Dossier","dossier")],HOLDOUT_LANGUAGE,UNSUPPORTED,[p for _,p in HOLDOUT_AMBIGUITIES],"gdm65-holdout")
 return g55._tag_ambiguity_families(rows,refs,HOLDOUT_AMBIGUITIES)
def _postprocess(root):
 for sp in root.rglob("summary.json"):
  d=sp.parent; cp=d/"config.json"; tp=d/"training.json"; c=json.loads(cp.read_text()); t=json.loads(tp.read_text()); s=json.loads(sp.read_text()); mode=ARMS[c["arm"]]["structured"]; t["selected_calibration"]=json.loads((d/"calibration.json").read_text())["selected"]
  c.update({"forward_scope":"operation-generation-only","schema_generation":"suspended-frozen","ambiguity_curriculum":"family-balanced","ambiguity_representation_mode":mode,"ambiguity_feature_dim":t["gdm65_ambiguity_feature_dim"],"ambiguity_requested_topk":5,"ambiguity_rank_preserving":True,"ambiguity_relation_feature_count":20,"ambiguity_relation_contract_hash":t["gdm65_relation_contract_hash"]})
  cp.write_text(json.dumps(c,indent=2,sort_keys=True)+"\n"); tp.write_text(json.dumps(t,indent=2,sort_keys=True)+"\n"); s["config"]=c; s["training"]=t; s["calibration_scope"]="Catalog/Ledgerbook calibration-only"; s["secondary_holdout_scope"]="Manuscript/Dossier fresh operation holdout"; s["regression_scope"]="corrected public operation test plus inspected GDM46-GDM64 operation holdouts"; s["secondary_holdout_ambiguity_family_metrics"]=g55._ambiguity_family_metrics(d/"predictions-holdout.jsonl"); s["file_hashes"]["config.json"]=g52.file_hash(cp); s["file_hashes"]["training.json"]=g52.file_hash(tp); sp.write_text(json.dumps(s,indent=2,sort_keys=True)+"\n")
def main():
 saved={k:getattr(g64,k) for k in ("FORMAT","SEEDS_DEFAULT","ARMS","REPRESENTATION_MODES","calibration_set","fresh_holdout","train_model","clause_state","_postprocess")}; orig63=g63.fresh_holdout; orig64=saved["fresh_holdout"]
 def prior(task):
  a,b=orig63(task); c,d=orig64(task); return a+c,{**b,**d}
 g63.fresh_holdout=prior; g64.FORMAT=FORMAT; g64.SEEDS_DEFAULT=SEEDS_DEFAULT; g64.ARMS=ARMS; g64.REPRESENTATION_MODES=MODES; g64.calibration_set=calibration_set; g64.fresh_holdout=fresh_holdout; g64.train_model=train_model; g64.clause_state=clause_state; g64._postprocess=_postprocess
 try: g64.main()
 finally:
  g63.fresh_holdout=orig63
  for k,v in saved.items(): setattr(g64,k,v)
if __name__=="__main__": main()
