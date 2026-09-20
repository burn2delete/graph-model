import json,random
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer,AutoModel
SEED=29001;random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(4)
# Mini Federation catalog generated from SDL-like metadata.
CAT=[
 {"id":"Product.created","kind":"scalar","ret":"String","owner":"products","key":False,"gloss":"when the product was first registered"},
 {"id":"Product.updated","kind":"scalar","ret":"String","owner":"products","key":False,"gloss":"when the product was most recently changed"},
 {"id":"Product.name","kind":"scalar","ret":"String","owner":"products","key":False,"gloss":"public product display name"},
 {"id":"Product.supplier","kind":"relationship","ret":"Supplier","owner":"products","key":False,"gloss":"business supplying the product"},
 {"id":"Product.reviews","kind":"relationship","ret":"Review","owner":"reviews","key":False,"gloss":"customer reviews of the product"},
 {"id":"Review.rating","kind":"scalar","ret":"Int","owner":"reviews","key":False,"gloss":"numeric score attached to a review"},
 {"id":"Review.author","kind":"relationship","ret":"User","owner":"reviews","key":False,"gloss":"person who wrote the review"},
 {"id":"User.name","kind":"scalar","ret":"String","owner":"users","key":False,"gloss":"public name of the user"},
]
REQ=[
 ("Return the moment this merchandise was initially persisted.","Product.created"),
 ("Return the timestamp of the newest edit.","Product.updated"),
 ("Return the customer-facing merchandise label.","Product.name"),
 ("Return the upstream commercial provider furnishing the item.","Product.supplier"),
 ("Return the consumer assessment entries.","Product.reviews"),
 ("Return the numeric evaluation attached to feedback.","Review.rating"),
 ("Return the person responsible for writing the assessment.","Review.author"),
 ("Return the human-readable identity of the feedback writer.","User.name"),
]
def text(x):return f"coordinate {x['id']}; capability kind {x['kind']}; returns {x['ret']}; owned by subgraph {x['owner']}; meaning {x['gloss']}"
tok=AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base",trust_remote_code=True);m=AutoModel.from_pretrained("jhu-clsp/mmBERT-base",trust_remote_code=True).eval()
for p in m.parameters():p.requires_grad=False
def emb(xs):
 with torch.no_grad():
  b=tok(xs,padding=True,truncation=True,max_length=160,return_tensors="pt");o=m(**b).last_hidden_state;mask=b["attention_mask"].unsqueeze(-1);return (o*mask).sum(1)/mask.sum(1).clamp_min(1)
# Train pairwise head using descriptions distinct from held-out requests.
trainphr={x["id"]:x["gloss"] for x in CAT};features=[]
for target in CAT:
 for neg in CAT:
  if target["id"]==neg["id"]:continue
  for cs in ([target,neg],[neg,target]):
   q=f"Expose {trainphr[target['id']]}.";e=emb([f"Request: {q} Candidate: {text(c)}" for c in cs]);features.append((e,[c["id"] for c in cs].index(target["id"])))
d=m.config.hidden_size;head=nn.Sequential(nn.Linear(d*2,256),nn.GELU(),nn.Linear(256,1));opt=torch.optim.AdamW(head.parameters(),lr=8e-4)
for ep in range(20):
 random.shuffle(features)
 for e,y in features:
  x=torch.cat([e,e-torch.flip(e,[0])],-1);z=head(x).squeeze(-1);loss=F.cross_entropy(z[None,:],torch.tensor([y]));opt.zero_grad();loss.backward();opt.step()
# Simulate retriever top-2 with hardest structurally adjacent candidate; typed reranker must resolve it.
hard={"Product.created":"Product.updated","Product.updated":"Product.created","Product.name":"Product.supplier","Product.supplier":"Product.name","Product.reviews":"Review.rating","Review.rating":"Product.reviews","Review.author":"User.name","User.name":"Review.author"}
byid={x["id"]:x for x in CAT};correct=0;details=[]
for rep in range(10):
 for q,target in REQ:
  ids=[target,hard[target]];random.Random(SEED+rep+len(details)).shuffle(ids);cs=[byid[i] for i in ids];e=emb([f"Request: {q} Candidate: {text(c)}" for c in cs]);x=torch.cat([e,e-torch.flip(e,[0])],-1);pred=ids[int(head(x).squeeze(-1).argmax())];correct+=pred==target;details.append({"target":target,"predicted":pred})
out={"worlds":len(details),"correct":correct,"accuracy":correct/len(details),"catalog_entries":len(CAT),"trainable_parameters":sum(p.numel() for p in head.parameters()),"failures":[x for x in details if x["target"]!=x["predicted"]]}
Path("artifacts").mkdir(exist_ok=True);Path("artifacts/gdm29.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
