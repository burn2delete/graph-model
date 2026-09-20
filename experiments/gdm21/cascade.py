import json,os,time,resource,random
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModel
SEED=21001;random.seed(SEED);torch.set_num_threads(4)
CAPS={
 "created":"Product.created: when the product was first registered",
 "updated":"Product.updated: when the product was most recently changed",
 "name":"Product.name: public product display name",
 "rating":"Product.rating: numeric score attached to the product",
 "supplier":"Product.supplier: business supplying the product",
 "reviews":"Product.reviews: customer reviews of the product",
 "author":"Review.author: person who wrote the review",
 "author_name":"User.name: public name of the review author",
}
Q={
 "created":"moment this merchandise was initially persisted","updated":"timestamp of the newest edit","name":"customer-facing merchandise label","rating":"numeric evaluation attached to the item",
 "supplier":"upstream commercial provider furnishing the item","reviews":"consumer assessment entries","author":"person responsible for writing the assessment","author_name":"human-readable identity of the feedback writer",
}
KEYS=list(CAPS)
retr_id=os.getenv("GDM21_RETRIEVER","xthor/Qwen3-Embedding-0.6B-GraphQL")
dec_id=os.getenv("GDM21_DECIDER","answerdotai/ModernBERT-base")
def load(mid):
 t=AutoTokenizer.from_pretrained(mid,trust_remote_code=True);m=AutoModel.from_pretrained(mid,trust_remote_code=True).eval();return t,m
def embed(tok,model,texts):
 with torch.no_grad():
  b=tok(texts,padding=True,truncation=True,max_length=128,return_tensors="pt");o=model(**b).last_hidden_state;mask=b["attention_mask"].unsqueeze(-1);e=(o*mask).sum(1)/mask.sum(1).clamp_min(1);return torch.nn.functional.normalize(e,dim=-1)
rt,rm=load(retr_id);dt,dm=load(dec_id)
corpus=[CAPS[k] for k in KEYS];ce=embed(rt,rm,corpus)
rows=[(f"Return the {Q[KEYS[i%len(KEYS)]]}.",KEYS[i%len(KEYS)]) for i in range(80)]
# Warm both models.
embed(rt,rm,[rows[0][0]]);embed(dt,dm,[rows[0][0]+" "+corpus[0]])
results=[]
for k in [1,2,4,8]:
 start=time.perf_counter();retr_ok=final_ok=0
 for q,target in rows:
  qe=embed(rt,rm,[q]);scores=(qe@ce.T)[0];idx=torch.topk(scores,min(k,len(KEYS))).indices.tolist();cands=[KEYS[i] for i in idx];retr_ok+=target in cands
  texts=[f"Request: {q} Candidate: {CAPS[c]}" for c in cands];de=embed(dt,dm,texts);q2=embed(dt,dm,[f"Request: {q}"]);sel=int((q2@de.T)[0].argmax());final_ok+=cands[sel]==target
 elapsed=time.perf_counter()-start
 results.append({"k":k,"retrieval_recall":retr_ok/len(rows),"final_accuracy":final_ok/len(rows),"seconds":elapsed,"ms_per_world":1000*elapsed/len(rows),"worlds_per_second":len(rows)/elapsed})
out={"retriever":retr_id,"decider":dec_id,"worlds":len(rows),"results":results,"max_rss_kb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"retriever_parameters":sum(p.numel() for p in rm.parameters()),"decider_parameters":sum(p.numel() for p in dm.parameters())}
Path("artifacts").mkdir(exist_ok=True);Path("artifacts/gdm21.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
