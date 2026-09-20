import json,os,time,resource,random
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModel
SEED=20002
random.seed(SEED);torch.set_num_threads(4)
CAPS={"created":"when the product was first registered","updated":"when the product was most recently changed","name":"public product display name","rating":"numeric score attached to the product"}
TEST={"created":"moment this merchandise was initially persisted","updated":"timestamp of the newest edit","name":"customer-facing merchandise label","rating":"numeric evaluation attached to the item"}
model_id=os.getenv("GDM20_QWEN","xthor/Qwen3-Embedding-0.6B-GraphQL")
tok=AutoTokenizer.from_pretrained(model_id,trust_remote_code=True);model=AutoModel.from_pretrained(model_id,trust_remote_code=True).eval()
def pool(o,mask):
 h=o.last_hidden_state;m=mask.unsqueeze(-1);return (h*m).sum(1)/m.sum(1).clamp_min(1)
def emb(texts):
 with torch.no_grad():
  b=tok(texts,padding=True,truncation=True,max_length=128,return_tensors="pt");return torch.nn.functional.normalize(pool(model(**b),b["attention_mask"]),dim=-1)
# Evaluate retrieval role: query -> candidate GraphQL coordinate/gloss.
corpus=[f"Product.{k}: {v}" for k,v in CAPS.items()];ce=emb(corpus)
rows=[]
for i in range(80):
 k=list(CAPS)[i%4];rows.append((f"Return the {TEST[k]}.",k))
# warmup
emb([rows[0][0]])
t=time.perf_counter();correct=0
for q,k in rows:
 qe=emb([q]);idx=int((qe@ce.T).argmax());correct+=list(CAPS)[idx]==k
elapsed=time.perf_counter()-t
out={"model":model_id,"role":"retriever","correct":correct,"worlds":len(rows),"seconds":elapsed,"ms_per_world":1000*elapsed/len(rows),"worlds_per_second":len(rows)/elapsed,"max_rss_kb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"parameters":sum(p.numel() for p in model.parameters())}
Path("artifacts").mkdir(exist_ok=True);Path("artifacts/qwen-graphql.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
