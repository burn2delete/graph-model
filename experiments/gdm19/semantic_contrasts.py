import json, os, random, re
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

SEED=19001
random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(4)
CAPS={
 "created":"when the product was first registered",
 "updated":"when the product was most recently changed",
 "name":"public product display name",
 "rating":"numeric score attached to the product",
}
PH={
 "created":["initial registration time","when the item first entered the catalog","moment this merchandise was initially persisted"],
 "updated":["latest modification time","when the item was last changed","timestamp of the newest edit"],
 "name":["public product name","display name for the item","customer-facing merchandise label"],
 "rating":["product score","numeric product rating","numeric evaluation attached to the item"],
}
KEYS=list(CAPS)

def rows(split,n,offset,risk=None):
 idx={"train":0,"validation":1,"test":2}[split];out=[]
 for i in range(n):
  t=KEYS[i%4];opts=[(k,CAPS[k]) for k in KEYS];random.Random(SEED+offset+i).shuffle(opts)
  target=t
  if risk=="missing":opts=[x for x in opts if x[0]!=t];target="NO_MATCH"
  if risk=="ambiguous":opts.append((t+"_alias",CAPS[t]));target="AMBIGUOUS"
  out.append({"q":f"Return the {PH[t][idx]}.","options":opts,"target":target,"semantic":t})
 return out

class Semantic(nn.Module):
 def __init__(self,name,finetune=False):
  super().__init__()
  from transformers import AutoTokenizer,AutoModel
  self.tok=AutoTokenizer.from_pretrained(name);self.enc=AutoModel.from_pretrained(name)
  if not finetune:
   self.enc.eval()
   for p in self.enc.parameters():p.requires_grad=False
  d=self.enc.config.hidden_size;self.choice=nn.Linear(d,1);self.support=nn.Linear(d,1);self.finetune=finetune
 def embed(self,q,ops,grad):
  texts=[f"Request: {q} Candidate: {d}" for _,d in ops]
  b=self.tok(texts,padding=True,truncation=True,max_length=96,return_tensors="pt")
  ctx=torch.enable_grad() if grad else torch.no_grad()
  with ctx:
   h=self.enc(**b).last_hidden_state;m=b["attention_mask"].unsqueeze(-1);e=(h*m).sum(1)/m.sum(1).clamp_min(1)
  return e

class RandomSemantic(nn.Module):
 def __init__(self,d=256):
  super().__init__();self.proj=nn.Linear(256,d);self.choice=nn.Linear(d,1);self.support=nn.Linear(d,1)
 def embed(self,q,ops,grad):
  xs=[]
  for _,desc in ops:
   v=torch.zeros(256)
   for tok in re.findall(r"[a-z]+",(q+" "+desc).lower()):v[hash(tok)%256]+=1
   xs.append(v)
  return torch.stack(xs)
 
def train(m,train_rows,epochs=5,lr=2e-4):
 params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=lr)
 rng=random.Random(SEED)
 for ep in range(epochs):
  rng.shuffle(train_rows)
  for x in train_rows:
   e=m.embed(x["q"],x["options"],True);keys=[k for k,_ in x["options"]];y=keys.index(x["semantic"])
   z=m.choice(e).squeeze(-1);s=m.support(e).squeeze(-1);lab=torch.zeros_like(s);lab[y]=1
   loss=F.cross_entropy(z[None,:],torch.tensor([y]))+.5*F.binary_cross_entropy_with_logits(s,lab)
   opt.zero_grad();loss.backward();opt.step()

def decide(m,x,t=.5):
 e=m.embed(x["q"],x["options"],False);z=m.choice(e).squeeze(-1);s=torch.sigmoid(m.support(e).squeeze(-1))
 good=[i for i,v in enumerate(s.tolist()) if v>=t]
 if not good:return "NO_MATCH"
 if len(good)>1:
  desc=[x["options"][i][1] for i in good]
  if len(set(desc))<len(desc):return "AMBIGUOUS"
  ranked=sorted([(float(z[i]),i) for i in good],reverse=True)
  if ranked[0][0]-ranked[1][0]<.2:return "AMBIGUOUS"
  return x["options"][ranked[0][1]][0]
 return x["options"][good[0]][0]

def evaluate(m,rs,t):
 return {"correct":sum(decide(m,x,t)==x["target"] for x in rs),"worlds":len(rs)}

def run_arm(name,m,trainset,val,test,missing,amb,lr):
 train(m,trainset,5,lr)
 ts=[.3,.4,.5,.6,.7,.8]
 t=max(ts,key=lambda z:evaluate(m,val,z)["correct"])
 return {"arm":name,"threshold":t,"ood":evaluate(m,test,t),"missing":evaluate(m,missing,t),"ambiguous":evaluate(m,amb,t)}

def main():
 tr=rows("train",120,0);val=rows("validation",40,1000)+rows("validation",12,1500,"missing")
 test=rows("test",80,2000);missing=rows("test",40,3000,"missing");amb=rows("test",40,4000,"ambiguous")
 name=os.getenv("GDM19_MODEL","distilbert-base-uncased")
 arms=[]
 arms.append(run_arm("random_control",RandomSemantic(),tr.copy(),val,test,missing,amb,8e-4))
 arms.append(run_arm("frozen_pretrained",Semantic(name,False),tr.copy(),val,test,missing,amb,2e-3))
 arms.append(run_arm("finetuned_pretrained",Semantic(name,True),tr.copy(),val,test,missing,amb,2e-5))
 out={"model":name,"seed":SEED,"arms":arms}
 os.makedirs("artifacts",exist_ok=True);Path("artifacts/gdm19.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=="__main__":main()
