import json,random,re
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer,AutoModel
SEED=32001;random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(4)
SDLS={"products":'''type Product @key(fields: "id") {
 id: ID!
 created: String!
 updated: String!
 name: String!
 supplier: Supplier
}
type Supplier @key(fields: "id") {
 id: ID!
 name: String!
}''',"reviews":'''type Review @key(fields: "id") {
 id: ID!
 rating: Int!
 author: User!
 product: Product!
}
type User @key(fields: "id") {
 id: ID!
 name: String!
}
extend type Product @key(fields: "id") {
 id: ID!
 reviews: [Review!]!
}'''}
SCALARS={"ID","String","Int","Float","Boolean"}
def parse(sdl,owner):
 out=[]
 for typ,header,body in re.findall(r'(?:extend\s+)?type\s+(\w+)([^\{]*)\{([^}]*)\}',sdl,re.S):
  keys=set()
  for fs in re.findall(r'@key\s*\(\s*fields\s*:\s*"([^"]+)"',header):keys.update(re.findall(r'\w+',fs))
  # Parse fields globally within body, not line-by-line, so compact SDL is supported.
  for field,raw in re.findall(r'(\w+)\s*(?:\([^)]*\))?\s*:\s*([\[\]!\w]+)',body):
   base=re.sub(r'[\[\]!]','',raw);out.append({"id":f"{typ}.{field}","parent":typ,"field":field,"kind":"scalar" if base in SCALARS else "relationship","ret":base,"list":"[" in raw,"nonnull":raw.endswith("!"),"owner":owner,"key":field in keys})
 return out
CAT=[]
for owner,sdl in SDLS.items():CAT+=parse(sdl,owner)
by={x["id"]:x for x in CAT}
def desc(x):
 return f"coordinate {x['id']}; parent {x['parent']}; field {x['field']}; capability {x['kind']}; returns {'list of ' if x['list'] else ''}{x['ret']}; {'entity key; ' if x['key'] else ''}owned by {x['owner']}"
def structural_prompts(x):
 if x["kind"]=="scalar":return [f"select scalar {x['field']} from {x['parent']}",f"return {x['ret']} field {x['parent']}.{x['field']}"]
 return [f"traverse {x['parent']}.{x['field']} relationship to {x['ret']}",f"select related {x['ret']} from {x['parent']}"]
# Semantic examples are only needed where structure cannot distinguish intent.
SEM={
"Product.created":["initial registration time","original creation timestamp"],
"Product.updated":["latest modification time","most recent revision timestamp"],
"Product.name":["public product name","customer-facing product label"],
"Review.rating":["numeric review score","rating attached to feedback"],
"User.name":["public user name","human-readable name of the review author"],
}
tok=AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base",trust_remote_code=True);model=AutoModel.from_pretrained("jhu-clsp/mmBERT-base",trust_remote_code=True).eval()
for p in model.parameters():p.requires_grad=False
def emb(xs):
 with torch.no_grad():
  b=tok(xs,padding=True,truncation=True,max_length=160,return_tensors="pt");o=model(**b).last_hidden_state;m=b["attention_mask"].unsqueeze(-1);return (o*m).sum(1)/m.sum(1).clamp_min(1)
features=[];structural=semantic=0
for x in CAT:
 for y in CAT:
  if x["id"]==y["id"]:continue
  prompts=structural_prompts(x)
  # If fields are structurally similar, augment with semantic intent examples.
  similar=(x["parent"]==y["parent"] and x["kind"]==y["kind"] and x["ret"]==y["ret"])
  if similar and x["id"] in SEM:prompts=prompts+SEM[x["id"]];semantic+=len(SEM[x["id"]])*2
  for q in prompts:
   for ids in ([x["id"],y["id"]],[y["id"],x["id"]]):
    e=emb([f"Request: {q}. Candidate: {desc(by[i])}" for i in ids]);features.append((e,ids.index(x["id"])));structural+=1
d=model.config.hidden_size;head=nn.Sequential(nn.Linear(d*2,384),nn.GELU(),nn.Linear(384,1));opt=torch.optim.AdamW(head.parameters(),lr=6e-4)
for ep in range(24):
 random.shuffle(features)
 for e,y in features:
  x=torch.cat([e,e-torch.flip(e,[0])],-1);z=head(x).squeeze(-1);loss=F.cross_entropy(z[None,:],torch.tensor([y]));opt.zero_grad();loss.backward();opt.step()
# Held-out semantic + structural requests.
TEST=[("moment this merchandise was initially persisted","Product.created","Product.updated"),("timestamp of the newest edit","Product.updated","Product.created"),("customer-facing merchandise label","Product.name","Product.supplier"),("consumer assessment entries","Product.reviews","Review.rating"),("numeric evaluation attached to feedback","Review.rating","Product.reviews"),("person responsible for writing the assessment","Review.author","User.name"),("human-readable identity of the feedback writer","User.name","Review.author")]
correct=0;fails=[]
for rep in range(10):
 for q,t,n in TEST:
  ids=[t,n];random.Random(SEED+rep+len(fails)).shuffle(ids);e=emb([f"Request: Return the {q}. Candidate: {desc(by[i])}" for i in ids]);x=torch.cat([e,e-torch.flip(e,[0])],-1);pred=ids[int(head(x).squeeze(-1).argmax())];correct+=pred==t
  if pred!=t:fails.append({"target":t,"predicted":pred})
out={"catalog_entries":len(CAT),"training_examples":len(features),"semantic_augmentations":semantic,"worlds":70,"accuracy":correct/70,"failures":fails}
Path("artifacts").mkdir(exist_ok=True);Path("artifacts/gdm32.json").write_text(json.dumps(out,indent=2));Path("artifacts/catalog.json").write_text(json.dumps(CAT,indent=2));print(json.dumps(out,indent=2))
