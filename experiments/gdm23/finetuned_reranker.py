import json,os,random,time,resource
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer,AutoModel
SEED=23001;random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(4)
CAPS={"created":"Product.created: when the product was first registered","updated":"Product.updated: when the product was most recently changed","name":"Product.name: public product display name","rating":"Product.rating: numeric score attached to the product","supplier":"Product.supplier: business supplying the product","reviews":"Product.reviews: customer reviews of the product","author":"Review.author: person who wrote the review","author_name":"User.name: public name of the review author"}
TRAIN={"created":["initial registration time","original product creation moment"],"updated":["latest modification time","most recent product edit"],"name":["public product name","display label for product"],"rating":["product score","numeric rating of product"],"supplier":["company supplying the item","business providing product"],"reviews":["customer reviews","consumer feedback entries"],"author":["person who wrote the review","review writer"],"author_name":["reviewer's public name","display name of review writer"]}
TEST={"created":"moment this merchandise was initially persisted","updated":"timestamp of the newest edit","name":"customer-facing merchandise label","rating":"numeric evaluation attached to the item","supplier":"upstream commercial provider furnishing the item","reviews":"consumer assessment entries","author":"person responsible for writing the assessment","author_name":"human-readable identity of the feedback writer"}
K=list(CAPS)
def load(mid):
 t=AutoTokenizer.from_pretrained(mid,trust_remote_code=True);m=AutoModel.from_pretrained(mid,trust_remote_code=True);return t,m
def emb(t,m,texts,grad=False):
 b=t(texts,padding=True,truncation=True,max_length=128,return_tensors="pt");ctx=torch.enable_grad() if grad else torch.no_grad()
 with ctx:
  o=m(**b).last_hidden_state;mask=b["attention_mask"].unsqueeze(-1);return (o*mask).sum(1)/mask.sum(1).clamp_min(1)
rt,rm=load(os.getenv("RETRIEVER","xthor/Qwen3-Embedding-0.6B-GraphQL"));rm.eval()
for p in rm.parameters():p.requires_grad=False
dt,dm=load(os.getenv("DECIDER","answerdotai/ModernBERT-base"))
# Fine-tune only top encoder block + head to keep update cost bounded.
for p in dm.parameters():p.requires_grad=False
blocks=None
for attr in ["layers","layer"]:
 obj=getattr(getattr(dm,"encoder",None),attr,None)
 if obj is not None:blocks=obj;break
if blocks is not None:
 for p in blocks[-1].parameters():p.requires_grad=True
head=nn.Linear(dm.config.hidden_size,1)
corpus=F.normalize(emb(rt,rm,[CAPS[k] for k in K]),dim=-1)
def retrieve_batch(qs,k=2):
 q=F.normalize(emb(rt,rm,qs),dim=-1);return torch.topk(q@corpus.T,k,dim=1).indices.tolist()
# Mine hard negatives in one batched retrieval pass.
train=[]
for rep in range(12):
 qs=[];targets=[]
 for target in K:
  for phrase in TRAIN[target]:qs.append(f"Return the {phrase}.");targets.append(target)
 ids=retrieve_batch(qs,2)
 for q,target,ix in zip(qs,targets,ids):
  cs=[K[i] for i in ix]
  if target not in cs:cs=[target,cs[0]]
  random.Random(SEED+rep+len(train)).shuffle(cs);train.append((q,cs,target))
opt=torch.optim.AdamW([p for p in list(dm.parameters())+list(head.parameters()) if p.requires_grad],lr=2e-5)
dm.train()
for ep in range(5):
 random.shuffle(train)
 for q,cs,target in train:
  e=emb(dt,dm,[f"Request: {q} Candidate: {CAPS[c]}" for c in cs],True);z=head(e).squeeze(-1);loss=F.cross_entropy(z[None,:],torch.tensor([cs.index(target)]));opt.zero_grad();loss.backward();opt.step()
dm.eval()
rows=[(f"Return the {TEST[K[i%8]]}.",K[i%8]) for i in range(80)]
start=time.perf_counter();ids=retrieve_batch([q for q,_ in rows],2);retr_ms=1000*(time.perf_counter()-start)/80
rec=sum(t in [K[i] for i in ix] for ix,(_,t) in zip(ids,rows))
start=time.perf_counter();correct=0
for ix,(q,target) in zip(ids,rows):
 cs=[K[i] for i in ix];e=emb(dt,dm,[f"Request: {q} Candidate: {CAPS[c]}" for c in cs]);pred=cs[int(head(e).squeeze(-1).argmax())];correct+=pred==target
rerank_ms=1000*(time.perf_counter()-start)/80
out={"retrieval_recall_at_2":rec/80,"final_accuracy":correct/80,"retrieval_ms_per_world":retr_ms,"rerank_ms_per_world":rerank_ms,"total_ms_per_world":retr_ms+rerank_ms,"max_rss_kb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"train_pairs":len(train),"trainable_parameters":sum(p.numel() for p in list(dm.parameters())+list(head.parameters()) if p.requires_grad)}
Path("artifacts").mkdir(exist_ok=True);Path("artifacts/gdm23.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
