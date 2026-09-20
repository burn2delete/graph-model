import json,os,time,resource,random
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModel
SEED=21002;random.seed(SEED);torch.set_num_threads(4)
CAPS={"created":"Product.created: when the product was first registered","updated":"Product.updated: when the product was most recently changed","name":"Product.name: public product display name","rating":"Product.rating: numeric score attached to the product","supplier":"Product.supplier: business supplying the product","reviews":"Product.reviews: customer reviews of the product","author":"Review.author: person who wrote the review","author_name":"User.name: public name of the review author"}
Q={"created":"moment this merchandise was initially persisted","updated":"timestamp of the newest edit","name":"customer-facing merchandise label","rating":"numeric evaluation attached to the item","supplier":"upstream commercial provider furnishing the item","reviews":"consumer assessment entries","author":"person responsible for writing the assessment","author_name":"human-readable identity of the feedback writer"}
KEYS=list(CAPS);rows=[(f"Return the {Q[KEYS[i%8]]}.",KEYS[i%8]) for i in range(80)]
def load(mid):
 t=AutoTokenizer.from_pretrained(mid,trust_remote_code=True);m=AutoModel.from_pretrained(mid,trust_remote_code=True).eval();return t,m
def embed(t,m,texts):
 with torch.no_grad():
  b=t(texts,padding=True,truncation=True,max_length=128,return_tensors="pt");o=m(**b).last_hidden_state;mask=b["attention_mask"].unsqueeze(-1);return torch.nn.functional.normalize((o*mask).sum(1)/mask.sum(1).clamp_min(1),dim=-1)
rt,rm=load(os.getenv("RETRIEVER","xthor/Qwen3-Embedding-0.6B-GraphQL"));dt,dm=load(os.getenv("DECIDER","answerdotai/ModernBERT-base"))
corpus=embed(rt,rm,[CAPS[k] for k in KEYS])
# Cache query retrieval embeddings once; this is the intended production pattern.
start=time.perf_counter();queries=embed(rt,rm,[q for q,_ in rows]);retrieval_batch_ms=1000*(time.perf_counter()-start)/len(rows)
results=[]
for k in [1,2,4]:
 top=torch.topk(queries@corpus.T,k,dim=1).indices.tolist();retr=sum(t in [KEYS[i] for i in ids] for ids,(_,t) in zip(top,rows))
 start=time.perf_counter();correct=0
 for ids,(q,target) in zip(top,rows):
  cs=[KEYS[i] for i in ids]
  if k==1:pred=cs[0]
  else:
   # batch all candidate pairs for a decision in one encoder pass
   ce=embed(dt,dm,[f"Request: {q} Candidate: {CAPS[c]}" for c in cs]);qe=embed(dt,dm,[f"Request: {q}"]);pred=cs[int((qe@ce.T)[0].argmax())]
  correct+=pred==target
 decision_ms=1000*(time.perf_counter()-start)/len(rows)
 results.append({"k":k,"retrieval_recall":retr/len(rows),"final_accuracy":correct/len(rows),"retrieval_batch_ms_per_world":retrieval_batch_ms,"decision_ms_per_world":decision_ms,"total_ms_per_world":retrieval_batch_ms+decision_ms})
out={"results":results,"max_rss_kb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
Path("artifacts").mkdir(exist_ok=True);Path("artifacts/gdm21-cached.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
