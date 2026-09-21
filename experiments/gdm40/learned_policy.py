import json,os,random
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer,AutoModel
SEED=40001;random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(4)
C={
"Product.name":{"kind":"scalar","ret":"String","text":"public product display name"},
"Product.created":{"kind":"scalar","ret":"String","text":"when product was first registered"},
"Product.updated":{"kind":"scalar","ret":"String","text":"when product was most recently changed"},
"Product.reviews":{"kind":"relationship","ret":"Review","text":"customer reviews of product"},
"Review.rating":{"kind":"scalar","ret":"Int","text":"numeric score attached to review"},
"Review.author":{"kind":"relationship","ret":"User","text":"person who wrote review"},
"User.name":{"kind":"scalar","ret":"String","text":"public name of review author"},
}
TRAIN={
"Product.name":["public product name"],"Product.created":["initial registration time"],"Product.updated":["latest revision time"],
"Product.reviews":["customer reviews"],"Review.rating":["review score"],"Review.author":["review writer"],"User.name":["reviewer's public name"]}
TEST=[
("customer-facing merchandise label","Product.name"),("moment this merchandise was initially persisted","Product.created"),
("timestamp of the newest edit","Product.updated"),("consumer assessment entries","Product.reviews"),
("numeric evaluation attached to feedback","Review.rating"),("person responsible for writing the assessment","Review.author"),
("human-readable identity of the feedback writer","User.name")]
def desc(k):
 x=C[k];p,f=k.split(".");return f"coordinate {k}; parent {p}; field {f}; capability {x['kind']}; returns {x['ret']}; meaning {x['text']}"
tok=AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base",trust_remote_code=True);m=AutoModel.from_pretrained("jhu-clsp/mmBERT-base",trust_remote_code=True).eval()
for p in m.parameters():p.requires_grad=False
def emb(xs):
 with torch.no_grad():
  b=tok(xs,padding=True,truncation=True,max_length=128,return_tensors="pt");o=m(**b).last_hidden_state;mask=b["attention_mask"].unsqueeze(-1);return (o*mask).sum(1)/mask.sum(1).clamp_min(1)
keys=list(C);features=[]
for t in keys:
 for n in keys:
  if t==n:continue
  for q in TRAIN[t]:
   for ids in ([t,n],[n,t]):
    e=emb([f"Request: {q}. Candidate: {desc(i)}" for i in ids]);features.append((e,ids.index(t)))
d=m.config.hidden_size;head=nn.Sequential(nn.Linear(d*2,256),nn.GELU(),nn.Linear(256,1));opt=torch.optim.AdamW(head.parameters(),lr=7e-4)
for _ in range(20):
 random.shuffle(features)
 for e,y in features:
  x=torch.cat([e,e-torch.flip(e,[0])],-1);z=head(x).squeeze(-1);loss=F.cross_entropy(z[None,:],torch.tensor([y]));opt.zero_grad();loss.backward();opt.step()
# Real learned policy: rank all typed coordinates; no expected-target fallback.
preds=[]
for q,target in TEST:
 # tournament ranking from pairwise wins
 wins={k:0 for k in keys}
 for i,a in enumerate(keys):
  for b in keys[i+1:]:
   ids=[a,b];e=emb([f"Request: {q}. Candidate: {desc(x)}" for x in ids]);x=torch.cat([e,e-torch.flip(e,[0])],-1);p=ids[int(head(x).squeeze(-1).argmax())];wins[p]+=1
 pred=max(wins,key=wins.get);preds.append({"request":q,"target":target,"predicted":pred,"correct":pred==target})
out={"cases":len(preds),"correct":sum(x["correct"] for x in preds),"accuracy":sum(x["correct"] for x in preds)/len(preds),"predictions":preds}
Path("artifacts").mkdir(exist_ok=True);Path("artifacts/gdm40.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
