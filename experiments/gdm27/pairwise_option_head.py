import json,os,random
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer,AutoModel
SEED=27001;random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(4)
CAPS={"created":"Product.created: when the product was first registered","updated":"Product.updated: when the product was most recently changed","name":"Product.name: public product display name","rating":"Product.rating: numeric score attached to the product","supplier":"Product.supplier: business supplying the product","reviews":"Product.reviews: customer reviews of the product","author":"Review.author: person who wrote the review","author_name":"User.name: public name of the review author"}
TRAIN={"created":["initial registration time","original product creation moment"],"updated":["latest modification time","most recent product edit"],"name":["public product name","display label for product"],"rating":["product score","numeric product rating"],"supplier":["company supplying the item","business providing product"],"reviews":["customer reviews","consumer feedback entries"],"author":["person who wrote the review","review writer"],"author_name":["reviewer's public name","display name of review writer"]}
TEST={"created":"moment this merchandise was initially persisted","updated":"timestamp of the newest edit","name":"customer-facing merchandise label","rating":"numeric evaluation attached to the item","supplier":"upstream commercial provider furnishing the item","reviews":"consumer assessment entries","author":"person responsible for writing the assessment","author_name":"human-readable identity of the feedback writer"}
K=list(CAPS)
def load(mid):
 t=AutoTokenizer.from_pretrained(mid,trust_remote_code=True);m=AutoModel.from_pretrained(mid,trust_remote_code=True).eval();return t,m
def emb(t,m,texts):
 with torch.no_grad():
  b=t(texts,padding=True,truncation=True,max_length=128,return_tensors="pt");o=m(**b).last_hidden_state;mask=b["attention_mask"].unsqueeze(-1);return (o*mask).sum(1)/mask.sum(1).clamp_min(1)
rt,rm=load("xthor/Qwen3-Embedding-0.6B-GraphQL");dt,dm=load("jhu-clsp/mmBERT-base")
for p in list(rm.parameters())+list(dm.parameters()):p.requires_grad=False
corpus=F.normalize(emb(rt,rm,[CAPS[k] for k in K]),dim=-1)
def retrieve(qs):return torch.topk(F.normalize(emb(rt,rm,qs),dim=-1)@corpus.T,2,dim=1).indices.tolist()
# Balanced pairwise curriculum: every capability is trained against every competing capability,
# while inference is still restricted to Qwen's top-2.
pairs=[]
for target in K:
 for neg in K:
  if neg==target:continue
  for phrase in TRAIN[target]:
   for order in range(2):
    cs=[target,neg] if order==0 else [neg,target]
    pairs.append((f"Return the {phrase}.",cs,target))
# Cache frozen mmBERT features.
features=[]
for q,cs,target in pairs:
 e=emb(dt,dm,[f"Request: {q} Candidate: {CAPS[c]}" for c in cs]);features.append((e,cs.index(target)))
d=dm.config.hidden_size
# Option interaction head sees candidate embedding plus pairwise difference.
head=nn.Sequential(nn.Linear(d*2,256),nn.GELU(),nn.Linear(256,1));opt=torch.optim.AdamW(head.parameters(),lr=8e-4)
for ep in range(20):
 random.shuffle(features)
 for e,y in features:
  other=torch.flip(e,[0]);x=torch.cat([e,e-other],dim=-1);z=head(x).squeeze(-1);loss=F.cross_entropy(z[None,:],torch.tensor([y]));opt.zero_grad();loss.backward();opt.step()
rows=[(f"Return the {TEST[K[i%8]]}.",K[i%8]) for i in range(80)];ids=retrieve([q for q,_ in rows]);rec=correct=0;fails=[]
for ix,(q,target) in zip(ids,rows):
 cs=[K[i] for i in ix];rec+=target in cs;e=emb(dt,dm,[f"Request: {q} Candidate: {CAPS[c]}" for c in cs]);x=torch.cat([e,e-torch.flip(e,[0])],dim=-1);pred=cs[int(head(x).squeeze(-1).argmax())];correct+=pred==target
 if pred!=target:fails.append({"target":target,"predicted":pred,"candidates":cs})
out={"retrieval_recall_at_2":rec/80,"final_accuracy":correct/80,"curriculum_pairs":len(pairs),"trainable_parameters":sum(p.numel() for p in head.parameters()),"failures":fails}
Path("artifacts").mkdir(exist_ok=True);Path("artifacts/gdm27.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
