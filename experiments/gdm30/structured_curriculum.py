import json,random
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer,AutoModel
SEED=30001;random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(4)
CAT=[
 {"id":"Product.created","kind":"scalar","ret":"String","owner":"products","gloss":"when the product was first registered"},
 {"id":"Product.updated","kind":"scalar","ret":"String","owner":"products","gloss":"when the product was most recently changed"},
 {"id":"Product.name","kind":"scalar","ret":"String","owner":"products","gloss":"public product display name"},
 {"id":"Product.supplier","kind":"relationship","ret":"Supplier","owner":"products","gloss":"business supplying the product"},
 {"id":"Product.reviews","kind":"relationship","ret":"Review","owner":"reviews","gloss":"customer reviews of the product"},
 {"id":"Review.rating","kind":"scalar","ret":"Int","owner":"reviews","gloss":"numeric score attached to a review"},
 {"id":"Review.author","kind":"relationship","ret":"User","owner":"reviews","gloss":"person who wrote the review"},
 {"id":"User.name","kind":"scalar","ret":"String","owner":"users","gloss":"public name of the review author"},
]
REQ=[("Return the moment this merchandise was initially persisted.","Product.created"),("Return the timestamp of the newest edit.","Product.updated"),("Return the customer-facing merchandise label.","Product.name"),("Return the upstream commercial provider furnishing the item.","Product.supplier"),("Return the consumer assessment entries.","Product.reviews"),("Return the numeric evaluation attached to feedback.","Review.rating"),("Return the person responsible for writing the assessment.","Review.author"),("Return the human-readable identity of the feedback writer.","User.name")]
by={x["id"]:x for x in CAT}
def text(x):return f"coordinate {x['id']}; capability kind {x['kind']}; returns {x['ret']}; owned by subgraph {x['owner']}; meaning {x['gloss']}"
tok=AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base",trust_remote_code=True);m=AutoModel.from_pretrained("jhu-clsp/mmBERT-base",trust_remote_code=True).eval()
for p in m.parameters():p.requires_grad=False
def emb(xs):
 with torch.no_grad():
  b=tok(xs,padding=True,truncation=True,max_length=160,return_tensors="pt");o=m(**b).last_hidden_state;mask=b["attention_mask"].unsqueeze(-1);return (o*mask).sum(1)/mask.sum(1).clamp_min(1)
# Structured curriculum includes both a semantic gloss and a type-aware paraphrase.
trainphr={
"Product.created":["when the product was first registered","creation timestamp scalar"],
"Product.updated":["when the product was most recently changed","revision timestamp scalar"],
"Product.name":["public product display name","string label of the product"],
"Product.supplier":["business supplying the product","relationship from product to supplier"],
"Product.reviews":["customer reviews of the product","relationship from product to reviews"],
"Review.rating":["numeric score attached to a review","integer scalar rating on a review"],
"Review.author":["person who wrote the review","relationship from review to user author"],
"User.name":["public name of the review author","string scalar name on the user"],
}
features=[]
for target in CAT:
 for neg in CAT:
  if target["id"]==neg["id"]:continue
  for phrase in trainphr[target["id"]]:
   for ids in ([target["id"],neg["id"]],[neg["id"],target["id"]]):
    cs=[by[i] for i in ids];e=emb([f"Request: Expose {phrase}. Candidate: {text(c)}" for c in cs]);features.append((e,ids.index(target["id"])))
d=m.config.hidden_size;head=nn.Sequential(nn.Linear(d*2,384),nn.GELU(),nn.Dropout(.1),nn.Linear(384,1));opt=torch.optim.AdamW(head.parameters(),lr=5e-4,weight_decay=.01)
# checkpoint by held-out TRAINING-family validation, never test wording.
val=[("Expose the product display name.","Product.name","Product.supplier"),("Expose the review author's name.","User.name","Review.author"),("Expose the review author relationship.","Review.author","User.name"),("Expose the review rating.","Review.rating","Product.reviews")]
best=None;bestacc=-1
for ep in range(30):
 random.shuffle(features)
 for e,y in features:
  x=torch.cat([e,e-torch.flip(e,[0])],-1);z=head(x).squeeze(-1);loss=F.cross_entropy(z[None,:],torch.tensor([y]));opt.zero_grad();loss.backward();opt.step()
 acc=0
 for q,t,n in val:
  ids=[t,n];e=emb([f"Request: {q} Candidate: {text(by[i])}" for i in ids]);x=torch.cat([e,e-torch.flip(e,[0])],-1);acc+=ids[int(head(x).squeeze(-1).argmax())]==t
 if acc>bestacc:bestacc=acc;best={k:v.detach().clone() for k,v in head.state_dict().items()}
head.load_state_dict(best)
hard={"Product.created":"Product.updated","Product.updated":"Product.created","Product.name":"Product.supplier","Product.supplier":"Product.name","Product.reviews":"Review.rating","Review.rating":"Product.reviews","Review.author":"User.name","User.name":"Review.author"}
correct=0;fails=[]
for rep in range(10):
 for q,target in REQ:
  ids=[target,hard[target]];random.Random(SEED+rep+len(fails)).shuffle(ids);e=emb([f"Request: {q} Candidate: {text(by[i])}" for i in ids]);x=torch.cat([e,e-torch.flip(e,[0])],-1);pred=ids[int(head(x).squeeze(-1).argmax())];correct+=pred==target
  if pred!=target:fails.append({"target":target,"predicted":pred})
out={"worlds":80,"accuracy":correct/80,"validation_correct":bestacc,"train_pairs":len(features),"trainable_parameters":sum(p.numel() for p in head.parameters()),"failures":fails}
Path("artifacts").mkdir(exist_ok=True);Path("artifacts/gdm30.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
