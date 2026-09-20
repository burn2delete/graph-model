import json,os,random,time,resource
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer,AutoModel
SEED=22001;random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(4)
CAPS={"created":"Product.created: when the product was first registered","updated":"Product.updated: when the product was most recently changed","name":"Product.name: public product display name","rating":"Product.rating: numeric score attached to the product","supplier":"Product.supplier: business supplying the product","reviews":"Product.reviews: customer reviews of the product","author":"Review.author: person who wrote the review","author_name":"User.name: public name of the review author"}
TRAIN={"created":"initial registration time","updated":"latest modification time","name":"public product name","rating":"product score","supplier":"company supplying the item","reviews":"customer reviews","author":"person who wrote the review","author_name":"reviewer's public name"}
TEST={"created":"moment this merchandise was initially persisted","updated":"timestamp of the newest edit","name":"customer-facing merchandise label","rating":"numeric evaluation attached to the item","supplier":"upstream commercial provider furnishing the item","reviews":"consumer assessment entries","author":"person responsible for writing the assessment","author_name":"human-readable identity of the feedback writer"}
K=list(CAPS)
def load(mid):
 t=AutoTokenizer.from_pretrained(mid,trust_remote_code=True);m=AutoModel.from_pretrained(mid,trust_remote_code=True).eval();return t,m
def emb(t,m,texts,grad=False):
 b=t(texts,padding=True,truncation=True,max_length=128,return_tensors="pt");ctx=torch.enable_grad() if grad else torch.no_grad()
 with ctx:
  o=m(**b).last_hidden_state;mask=b["attention_mask"].unsqueeze(-1);return (o*mask).sum(1)/mask.sum(1).clamp_min(1)
rt,rm=load(os.getenv("RETRIEVER","xthor/Qwen3-Embedding-0.6B-GraphQL"));dt,dm=load(os.getenv("DECIDER","answerdotai/ModernBERT-base"))
for p in dm.parameters():p.requires_grad=False
head=nn.Linear(dm.config.hidden_size,1);opt=torch.optim.AdamW(head.parameters(),lr=2e-3)
corpus=torch.nn.functional.normalize(emb(rt,rm,[CAPS[k] for k in K]),dim=-1)
def retrieve(q,k=2):
 qe=torch.nn.functional.normalize(emb(rt,rm,[q]),dim=-1);return torch.topk((qe@corpus.T)[0],k).indices.tolist()
# Mine top-2 hard negatives from retriever on training wording.
pairs=[]
for epoch_copy in range(8):
 for target in K:
  q=f"Return the {TRAIN[target]}.";ids=retrieve(q,2);cands=[K[i] for i in ids]
  if target not in cands:cands=[target,cands[0]]
  random.Random(SEED+epoch_copy+K.index(target)).shuffle(cands);pairs.append((q,cands,target))
# Train only the tiny reranker head; encoder stays frozen.
for ep in range(12):
 random.shuffle(pairs)
 for q,cands,target in pairs:
  e=emb(dt,dm,[f"Request: {q} Candidate: {CAPS[c]}" for c in cands]);z=head(e).squeeze(-1);y=cands.index(target);loss=F.cross_entropy(z[None,:],torch.tensor([y]));opt.zero_grad();loss.backward();opt.step()
# Test top-2 cascade on held-out wording and measure stages.
rows=[(f"Return the {TEST[K[i%8]]}.",K[i%8]) for i in range(80)]
start=time.perf_counter();retrieved=[];rec=0
for q,target in rows:
 ids=retrieve(q,2);cs=[K[i] for i in ids];retrieved.append(cs);rec+=target in cs
retr_ms=1000*(time.perf_counter()-start)/len(rows)
start=time.perf_counter();correct=0
for (q,target),cs in zip(rows,retrieved):
 e=emb(dt,dm,[f"Request: {q} Candidate: {CAPS[c]}" for c in cs]);pred=cs[int(head(e).squeeze(-1).argmax())];correct+=pred==target
rerank_ms=1000*(time.perf_counter()-start)/len(rows)
out={"retrieval_recall_at_2":rec/80,"final_accuracy":correct/80,"retrieval_ms_per_world":retr_ms,"rerank_ms_per_world":rerank_ms,"total_ms_per_world":retr_ms+rerank_ms,"max_rss_kb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"train_pairs":len(pairs),"trainable_parameters":sum(p.numel() for p in head.parameters())}
Path("artifacts").mkdir(exist_ok=True);Path("artifacts/gdm22.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
