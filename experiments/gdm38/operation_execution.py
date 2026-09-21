import json,re
from pathlib import Path
# GDM38 operation validation + fixture execution.
SCHEMA={
"Query":{"product":"Product"},
"Product":{"id":"ID","name":"String","created":"String","reviews":"Review"},
"Review":{"id":"ID","rating":"Int"},
}
OPS=[
("catalog","query Generated { product { name created } }",{"product":{"id":"p1","name":"Widget","created":"2026-01-01","reviews":[{"id":"r1","rating":5}]}}),
("reviews","query Generated { product { reviews { rating } } }",{"product":{"id":"p2","name":"Gadget","created":"2025-01-01","reviews":[{"id":"r2","rating":2},{"id":"r3","rating":4}]}}),
]
TOK=re.compile(r'[A-Za-z_][A-Za-z0-9_]*|[{}]')
def parse(src):
 ts=TOK.findall(src);i=0
 if ts and ts[0]=="query":i+=1
 if i<len(ts) and ts[i] not in "{}":i+=1
 def sel():
  nonlocal i
  assert ts[i]=="{";i+=1;out=[]
  while ts[i]!="}":
   f=ts[i];i+=1;children=[]
   if ts[i]=="{":children=sel()
   out.append((f,children))
  i+=1;return out
 return sel()
def validate(selection,typ):
 for f,ch in selection:
  if f not in SCHEMA[typ]:return False
  ret=SCHEMA[typ][f]
  if ch:
   if ret not in SCHEMA or not validate(ch,ret):return False
  elif ret in SCHEMA:return False
 return True
def execute(selection,obj):
 out={}
 for f,ch in selection:
  v=obj[f]
  if ch:
   if isinstance(v,list):out[f]=[execute(ch,x) for x in v]
   else:out[f]=execute(ch,v)
  else:out[f]=v
 return out
results=[]
for name,op,fixture in OPS:
 tree=parse(op);valid=validate(tree,"Query");resp=execute(tree,fixture) if valid else None
 results.append({"id":name,"operation":op,"valid":valid,"response":resp})
out={"cases":len(results),"valid":sum(x["valid"] for x in results),"executed":sum(x["response"] is not None for x in results),"results":results}
Path("artifacts").mkdir(exist_ok=True);Path("artifacts/gdm38.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if out["valid"]!=len(results) or out["executed"]!=len(results):raise SystemExit(1)
