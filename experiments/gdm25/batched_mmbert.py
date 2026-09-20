import json,os,random,time,resource
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer,AutoModel
SEED=25001;random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(4)
CAPS={"created":"Product.created: when the product was first registered","updated":"Product.updated: when the product was most recently changed","name":"Product.name: public product display name","rating":"Product.rating: numeric score attached to the product","supplier":"Product.supplier: business supplying the product","reviews":"Product.reviews: customer reviews of the product","author":"Review.author: person who wrote the review","author_name":"User.name: public name of the review author"}
TRAIN={"created":["initial registration time","original product creation moment"],"updated":["latest modification time","most recent product edit"],"name":["public product name","display label for product"],"rating":["product score","numeric rating of product"],"supplier":["company supplying the item","business providing product"],"reviews":["customer reviews","consumer feedback entries"],"author":["person who wrote the review","review writer"],"author_name":["reviewer's public name","display name of review writer"]}
TEST={"created":"moment this merchandise was initially persisted","updated":"timestamp of the newest edit","name":"customer-facing merchandise label","rating":"numeric evaluation attached to the item","supplier":"upstream commercial provider furnishing the item","reviews":"consumer assessment entries","author":"person responsible for writing the assessment","author_name":"human-readable identity of the feedback writer"}
K=list(CAPS)
def load(mid):
 t=AutoTokenizer.from_pretrained(mid,trust_remote_code=True);m=AutoModel.from_pretrained(mid,trust_remote_code=True).eval();return t,m
def emb(t,m,texts):
 with torch.no_grad():
  b=t(texts,padding=True,truncation=True,max_length=128,return_tensors="pt");o=m(**b).last_hidden_state;mask=b["attention_mask"].unsqueeze(-1);return (o*mask).sum(1)/mask.sum(1).clamp_min(1)
rt,rm=load(os.getenv("RETRIEVER","xthor/Qwen3-Embedding-0.6B-GraphQL"));dt,dm=load(os.getenv("DECIDER","jhu-clsp/mmBERT-base"))
for p in list(rm.parameters())+list(dm.parameters()):p.requires_grad=False
# Precompute immutable corpus vectors.
corpus=F.normalize(emb(rt,rm,[CAPS[k] for k in K]),dim=-1)
# Batch all training requests once.
train_q=[];train_target=[]
for rep in range(16):
 for target in K:
  for phrase in TRAIN[target]:train_q.append(f"Return the {phrase}.");train_target.append(target)
rq=F.normalize(emb(rt,rm,train_q),dim=-1);top=torch.topk(rq@corpus.T,2,dim=1).indices.tolist()
# Cache mmBERT candidate-pair features once: training the head should not re-run the encoder.
feature_rows=[]
for n,(q,target,ix) in enumerate(zip(train_q,train_target,top)):
 cs=[K[i] for i in ix]
 if target not in cs:cs=[target,cs[0]]
 random.Random(SEED+n).shuffle(cs)
 e=emb(dt,dm,[f"Request: {q} Candidate: {CAPS[c]}" for c in cs]);feature_rows.append((e,cs.index(target)))
d=dm.config.hidden_size;head=nn.Sequential(nn.Linear(d,128),nn.GELU(),nn.Linear(128,1));opt=torch.optim.AdamW(head.parameters(),lr=1e-3)
for ep in range(16):
 random.shuffle(feature_rows)
 for e,y in feature_rows:
  z=head(e).squeeze(-1);loss=F.cross_entropy(z[None,:],torch.tensor([y]));opt.zero_grad();loss.backward();opt.step()
rows=[(f"Return the {TEST[K[i%8]]}.",K[i%8]) for i in range(80)]
# Production-style batch retrieval.
start=time.perf_counter();qvec=F.normalize(emb(rt,rm,[q for q,_ in rows]),dim=-1);ids=torch.topk(qvec@corpus.T,2,dim=1).indices.tolist();retr_ms=1000*(time.perf_counter()-start)/80
rec=sum(t in [K[i] for i in ix] for ix,(_,t) in zip(ids,rows))
# Production-style batch rerank: one mmBERT forward pass for all 160 candidate pairs.
texts=[];mapping=[]
for wi,(ix,(q,target)) in enumerate(zip(ids,rows)):
 cs=[K[i] for i in ix]
 for ci,c in enumerate(cs):texts.append(f"Request: {q} Candidate: {CAPS[c]}");mapping.append((wi,ci,c))
start=time.perf_counter();features=emb(dt,dm,texts);scores=head(features).squeeze(-1);rerank_ms=1000*(time.perf_counter()-start)/80
by=[[] for _ in rows]
for score,(wi,ci,c) in zip(scores.tolist(),mapping):by[wi].append((score,c))
correct=sum(max(v)[1]==target for v,(_,target) in zip(by,rows))
out={"retrieval_recall_at_2":rec/80,"final_accuracy":correct/80,"retrieval_ms_per_world":retr_ms,"rerank_ms_per_world":rerank_ms,"total_ms_per_world":retr_ms+rerank_ms,"worlds_per_second":1000/(retr_ms+rerank_ms),"train_pairs":len(feature_rows),"trainable_parameters":sum(p.numel() for p in head.parameters()),"max_rss_kb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
Path("artifacts").mkdir(exist_ok=True);Path("artifacts/gdm25.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
