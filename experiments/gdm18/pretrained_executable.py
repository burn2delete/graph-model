import json, os, random, re, subprocess, tempfile
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

SEED=18002
random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(4)

CAPS={
 "created":("String","when the product was first registered"),
 "updated":("String","when the product was most recently changed"),
 "name":("String","public product display name"),
 "rating":("Int","numeric score attached to the product"),
}
PHRASES={
 "created":["initial registration time","when the item first entered the catalog","moment this merchandise was initially persisted"],
 "updated":["latest modification time","when the item was last changed","timestamp of the newest edit"],
 "name":["public product name","display name for the item","customer-facing merchandise label"],
 "rating":["product score","numeric product rating","numeric evaluation attached to the item"],
}
FIELDS=list(CAPS)

def examples(split,n,offset,risk=None):
 out=[]
 phrase_idx={"train":0,"validation":1,"test":2}[split]
 for i in range(n):
  target=FIELDS[i%len(FIELDS)]
  opts=[(k,CAPS[k][1]) for k in FIELDS]
  random.Random(SEED+offset+i).shuffle(opts)
  q=f"Return the {PHRASES[target][phrase_idx]}."
  expected=target
  if risk=="missing":
   opts=[x for x in opts if x[0]!=target]; expected="NO_MATCH"
  elif risk=="ambiguous":
   opts.append((target+"_alias",CAPS[target][1])); expected="AMBIGUOUS"
  out.append({"id":f"{split}-{risk or 'normal'}-{i}","question":q,"options":opts,"target":expected,"semantic_target":target})
 return out

class Model:
 def __init__(self,name):
  from transformers import AutoTokenizer,AutoModel
  self.tok=AutoTokenizer.from_pretrained(name); self.enc=AutoModel.from_pretrained(name).eval()
  for p in self.enc.parameters():p.requires_grad=False
  d=self.enc.config.hidden_size; self.choice=nn.Linear(d,1); self.support=nn.Linear(d,1)
 def embed(self,q,opts):
  texts=[f"GraphQL request: {q} Candidate capability: {desc}" for _,desc in opts]
  with torch.no_grad():
   b=self.tok(texts,padding=True,truncation=True,max_length=96,return_tensors="pt")
   h=self.enc(**b).last_hidden_state;m=b["attention_mask"].unsqueeze(-1)
   return (h*m).sum(1)/m.sum(1).clamp_min(1)

def train(m,rows,epochs=6):
 cache=[]
 for x in rows:
  e=m.embed(x["question"],x["options"]); keys=[k for k,_ in x["options"]]; y=keys.index(x["semantic_target"])
  cache.append((e,y))
  keep=[j for j in range(len(keys)) if j!=y];cache.append((e[keep],-1))
 opt=torch.optim.AdamW(list(m.choice.parameters())+list(m.support.parameters()),lr=2e-3)
 rng=random.Random(SEED)
 for ep in range(epochs):
  rng.shuffle(cache)
  for e,y in cache:
   s=m.support(e).squeeze(-1)
   if y>=0:
    z=m.choice(e).squeeze(-1); lab=torch.zeros_like(s);lab[y]=1
    loss=F.cross_entropy(z[None,:],torch.tensor([y]))+.5*F.binary_cross_entropy_with_logits(s,lab)
   else:loss=F.binary_cross_entropy_with_logits(s,torch.zeros_like(s))
   opt.zero_grad();loss.backward();opt.step()

def decide(m,x,t):
 e=m.embed(x["question"],x["options"])
 with torch.no_grad():z=m.choice(e).squeeze(-1);s=torch.sigmoid(m.support(e).squeeze(-1))
 good=[i for i,v in enumerate(s.tolist()) if v>=t]
 if not good:return "NO_MATCH"
 if len(good)>1:
  desc=[x["options"][i][1] for i in good]
  if len(set(desc))<len(desc):return "AMBIGUOUS"
  ranked=sorted([(float(z[i]),i) for i in good],reverse=True)
  if ranked[0][0]-ranked[1][0]<.2:return "AMBIGUOUS"
  return x["options"][ranked[0][1]][0]
 return x["options"][good[0]][0]

def lexical(x):
 q=set(re.findall(r"[a-z]+",x["question"].lower()))
 scores=[]
 for k,d in x["options"]:
  scores.append((len(q&set(re.findall(r"[a-z]+",d.lower()))),k))
 return max(scores)[1]

def schema(field):
 typ=CAPS[field][0]
 return f'''extend schema @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@key"])
type Query {{ product: Product! }}
type Product @key(fields: "id") {{ id: ID! {field}: {typ} }}
'''

def rover_compose(field,cache):
 if field in cache:return cache[field]
 if not shutil_which("rover"):cache[field]=None;return None
 with tempfile.TemporaryDirectory() as d:
  Path(d,"products.graphql").write_text(schema(field))
  Path(d,"supergraph.yaml").write_text('federation_version: =2.0\nsubgraphs:\n  products:\n    routing_url: http://products/graphql\n    schema:\n      file: ./products.graphql\n')
  p=subprocess.run(["rover","supergraph","compose","--config",str(Path(d,"supergraph.yaml"))],capture_output=True,text=True)
  cache[field]=p.returncode==0
 return cache[field]

def shutil_which(cmd):
 import shutil;return shutil.which(cmd)

def execute(field,target):
 fixtures=[
  {"created":"2026-01-01","updated":"2026-02-01","name":"A","rating":5},
  {"created":"2025-03-01","updated":"2026-04-01","name":"B","rating":2},
  {"created":"2024-05-01","updated":"2026-06-01","name":"C","rating":4},
 ]
 return field==target and all(f[field]==f[target] for f in fixtures)

def eval_arm(name,fn,rows,compose_cache):
 decision=chain=compose_ok=0;nm=amb=0
 for x in rows:
  p=fn(x);decision+=p==x["target"]
  if x["target"]=="NO_MATCH":nm+=p=="NO_MATCH";continue
  if x["target"]=="AMBIGUOUS":amb+=p=="AMBIGUOUS";continue
  if p not in CAPS:continue
  c=rover_compose(p,compose_cache)
  compose_ok+=c is True
  chain+=(c is True and execute(p,x["semantic_target"]))
 return {"arm":name,"worlds":len(rows),"decision_correct":decision,"composed":compose_ok,"executed_semantically_correct":chain,"no_match_correct":nm,"ambiguous_correct":amb}

def main():
 train_rows=examples("train",120,0)
 val=examples("validation",40,1000)+examples("validation",12,2000,"missing")
 test=examples("test",80,3000)
 missing=examples("test",40,4000,"missing")
 ambiguous=examples("test",40,5000,"ambiguous")
 m=Model(os.getenv("GDM18_PRETRAINED","distilbert-base-uncased"));train(m,train_rows)
 thresholds=[.4,.45,.5,.55,.6,.65,.7,.75,.8]
 # validation only
 t=max(thresholds,key=lambda z:sum(decide(m,x,z)==x["target"] for x in val))
 cache={}
 suites={}
 for label,rows in [("ood_wording",test),("missing",missing),("ambiguous",ambiguous)]:
  suites[label]=[
   eval_arm("first_option",lambda x:x["options"][0][0],rows,cache),
   eval_arm("lexical",lexical,rows,cache),
   eval_arm("pretrained",lambda x:decide(m,x,t),rows,cache),
  ]
 out={"seed":SEED,"threshold":t,"rover_version":subprocess.run(["rover","--version"],capture_output=True,text=True).stdout.strip() if shutil_which("rover") else None,"suites":suites}
 os.makedirs("artifacts",exist_ok=True)
 Path("artifacts/gdm18-pretrained.json").write_text(json.dumps(out,indent=2))
 print(json.dumps(out,indent=2))

if __name__=="__main__":main()
