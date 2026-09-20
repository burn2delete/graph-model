import json, os, re, random, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
SEED=20001
random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(4)
CAPS={"created":"when the product was first registered","updated":"when the product was most recently changed","name":"public product display name","rating":"numeric score attached to the product"}
P={"created":["initial registration time","when item entered catalog","moment merchandise was initially persisted"],"updated":["latest modification time","when item was last changed","timestamp of newest edit"],"name":["public product name","display name for item","customer-facing merchandise label"],"rating":["product score","numeric product rating","numeric evaluation attached to item"]}
K=list(CAPS)
def rows(split,n,off):
 idx={"train":0,"validation":1,"test":2}[split];out=[]
 for i in range(n):
  t=K[i%4];ops=list(CAPS.items());random.Random(SEED+off+i).shuffle(ops)
  out.append({"q":f"Return the {P[t][idx]}.","ops":ops,"target":t})
 return out
class Arm(nn.Module):
 def __init__(self,name,finetune):
  super().__init__();from transformers import AutoTokenizer,AutoModel
  self.tok=AutoTokenizer.from_pretrained(name,trust_remote_code=True);self.enc=AutoModel.from_pretrained(name,trust_remote_code=True)
  if not finetune:
   self.enc.eval()
   for p in self.enc.parameters():p.requires_grad=False
  d=self.enc.config.hidden_size;self.head=nn.Linear(d,1)
 def embed(self,x,grad):
  texts=[f"Request: {x['q']} Candidate: {d}" for _,d in x["ops"]];b=self.tok(texts,padding=True,truncation=True,max_length=96,return_tensors="pt")
  ctx=torch.enable_grad() if grad else torch.no_grad()
  with ctx:
   o=self.enc(**b).last_hidden_state;m=b["attention_mask"].unsqueeze(-1);return (o*m).sum(1)/m.sum(1).clamp_min(1)
def train(m,rs,epochs,lr):
 opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=lr);rng=random.Random(SEED)
 for _ in range(epochs):
  rng.shuffle(rs)
  for x in rs:
   e=m.embed(x,True);y=[k for k,_ in x["ops"]].index(x["target"]);z=m.head(e).squeeze(-1);loss=F.cross_entropy(z[None,:],torch.tensor([y]));opt.zero_grad();loss.backward();opt.step()
def eval(m,rs):
 t=time.perf_counter();c=0
 for x in rs:
  e=m.embed(x,False);p=x["ops"][int(m.head(e).squeeze(-1).argmax())][0];c+=p==x["target"]
 return {"correct":c,"worlds":len(rs),"seconds":time.perf_counter()-t}
def main():
 tr=rows("train",80,0);test=rows("test",80,2000)
 models=json.loads(os.environ["GDM20_MODELS"]);out=[]
 for spec in models:
  for ft in ([False,True] if spec.get("finetune",True) else [False]):
   try:
    m=Arm(spec["id"],ft);train(m,tr.copy(),2 if ft else 4,2e-5 if ft else 2e-3);r=eval(m,test);r.update({"model":spec["name"],"id":spec["id"],"finetuned":ft,"parameters":sum(p.numel() for p in m.parameters())});out.append(r);del m
   except Exception as e:out.append({"model":spec["name"],"id":spec["id"],"finetuned":ft,"error":repr(e)})
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/gdm20.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=="__main__":main()
