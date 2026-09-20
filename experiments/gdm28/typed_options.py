import json,random
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer,AutoModel
SEED=28001;random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(4)
CAPS={
 "created":{"coord":"Product.created","kind":"scalar","return":"String","gloss":"when the product was first registered"},
 "updated":{"coord":"Product.updated","kind":"scalar","return":"String","gloss":"when the product was most recently changed"},
 "name":{"coord":"Product.name","kind":"scalar","return":"String","gloss":"public product display name"},
 "rating":{"coord":"Product.rating","kind":"scalar","return":"Int","gloss":"numeric score attached to the product"},
 "supplier":{"coord":"Product.supplier","kind":"relationship","return":"Supplier","gloss":"business supplying the product"},
 "reviews":{"coord":"Product.reviews","kind":"relationship","return":"Review","gloss":"customer reviews of the product"},
 "author":{"coord":"Review.author","kind":"relationship","return":"User","gloss":"person who wrote the review"},
 "author_name":{"coord":"User.name","kind":"scalar","return":"String","gloss":"public name of the review author"},
}
TRAIN={"created":["initial registration time"],"updated":["latest modification time"],"name":["public product name"],"rating":["numeric product rating"],"supplier":["company supplying the item"],"reviews":["customer reviews"],"author":["person who wrote the review"],"author_name":["reviewer's public name"]}
TEST={"created":"moment this merchandise was initially persisted","updated":"timestamp of the newest edit","name":"customer-facing merchandise label","rating":"numeric evaluation attached to the item","supplier":"upstream commercial provider furnishing the item","reviews":"consumer assessment entries","author":"person responsible for writing the assessment","author_name":"human-readable identity of the feedback writer"}
K=list(CAPS)
tok=AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base",trust_remote_code=True);model=AutoModel.from_pretrained("jhu-clsp/mmBERT-base",trust_remote_code=True).eval()
for p in model.parameters():p.requires_grad=False
def text(c):
 x=CAPS[c];return f"coordinate {x['coord']}; capability kind {x['kind']}; returns {x['return']}; meaning {x['gloss']}"
def emb(texts):
 with torch.no_grad():
  b=tok(texts,padding=True,truncation=True,max_length=128,return_tensors="pt");o=model(**b).last_hidden_state;m=b["attention_mask"].unsqueeze(-1);return (o*m).sum(1)/m.sum(1).clamp_min(1)
# Full balanced typed pair curriculum, both orders.
features=[]
for target in K:
 for neg in K:
  if target==neg:continue
  for phrase in TRAIN[target]:
   for cs in ([target,neg],[neg,target]):
    e=emb([f"Request: Return the {phrase}. Candidate: {text(c)}" for c in cs]);features.append((e,cs.index(target)))
d=model.config.hidden_size;head=nn.Sequential(nn.Linear(d*2,256),nn.GELU(),nn.Linear(256,1));opt=torch.optim.AdamW(head.parameters(),lr=8e-4)
for ep in range(24):
 random.shuffle(features)
 for e,y in features:
  x=torch.cat([e,e-torch.flip(e,[0])],-1);z=head(x).squeeze(-1);loss=F.cross_entropy(z[None,:],torch.tensor([y]));opt.zero_grad();loss.backward();opt.step()
# Test specifically over semantically plausible top-2 pairs, including author vs author_name.
pairs={"created":"updated","updated":"created","name":"supplier","rating":"reviews","supplier":"name","reviews":"rating","author":"author_name","author_name":"author"}
correct=0;fails=[]
for i in range(80):
 target=K[i%8];cs=[target,pairs[target]];random.Random(SEED+i).shuffle(cs);q=f"Return the {TEST[target]}.";e=emb([f"Request: {q} Candidate: {text(c)}" for c in cs]);x=torch.cat([e,e-torch.flip(e,[0])],-1);pred=cs[int(head(x).squeeze(-1).argmax())];correct+=pred==target
 if pred!=target:fails.append({"target":target,"predicted":pred})
out={"final_accuracy":correct/80,"train_pairs":len(features),"trainable_parameters":sum(p.numel() for p in head.parameters()),"failures":fails}
Path("artifacts").mkdir(exist_ok=True);Path("artifacts/gdm28.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
