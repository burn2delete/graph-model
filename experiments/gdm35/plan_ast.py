import json,collections
from pathlib import Path
# GDM35: emit an Apollo-style executable plan AST with explicit entity transitions.
COORDS=[
{"c":"Query.product","p":"Query","f":"product","r":"Product","k":"rel","o":"products","key":None},
{"c":"Product.id","p":"Product","f":"id","r":"ID","k":"scalar","o":"products","key":"id"},
{"c":"Product.reviews","p":"Product","f":"reviews","r":"Review","k":"rel","o":"reviews","key":None},
{"c":"Review.id","p":"Review","f":"id","r":"ID","k":"scalar","o":"reviews","key":"id"},
{"c":"Review.author","p":"Review","f":"author","r":"User","k":"rel","o":"reviews","key":None},
{"c":"Review.rating","p":"Review","f":"rating","r":"Int","k":"scalar","o":"reviews","key":None},
{"c":"User.id","p":"User","f":"id","r":"ID","k":"scalar","o":"reviews","key":"id"},
{"c":"User.name","p":"User","f":"name","r":"String","k":"scalar","o":"reviews","key":None},
]
by=collections.defaultdict(list)
for x in COORDS:by[x["p"]].append(x)
keys={"Product":"id","Review":"id","User":"id"}
def path(target):
 q=collections.deque([("Query",[])]);seen=set()
 while q:
  typ,p=q.popleft()
  if (typ,tuple(x["c"] for x in p)) in seen:continue
  seen.add((typ,tuple(x["c"] for x in p)))
  for x in by[typ]:
   np=p+[x]
   if x["c"]==target:return np
   if x["k"]=="rel":q.append((x["r"],np))
def operation(p):
 s=p[-1]["f"]
 for x in reversed(p[:-1]):s=x["f"]+" { "+s+" }"
 return "query Generated { "+s+" }"
def planner(p):
 # Group contiguous coordinates by service, inserting entity representation fetches on ownership changes.
 groups=[];cur=None
 for x in p:
  if x["o"]!=cur:
   if groups:
    entity=x["p"];key=keys.get(entity)
    groups.append({"kind":"Flatten","path":entity,"node":{"kind":"Fetch","service":x["o"],"requires":["__typename"]+([key] if key else []),"entity":entity,"coordinates":[]}})
    cur=x["o"];groups[-1]["node"]["coordinates"].append(x["c"])
   else:
    groups.append({"kind":"Fetch","service":x["o"],"coordinates":[x["c"]]});cur=x["o"]
  else:
   g=groups[-1]["node"] if groups[-1]["kind"]=="Flatten" else groups[-1];g["coordinates"].append(x["c"])
 return {"kind":"Sequence","nodes":groups}
targets=["Review.rating","User.name"];res=[]
for t in targets:
 p=path(t);res.append({"target":t,"operation":operation(p),"plan":planner(p)})
out={"resolved":len(res),"results":res}
Path("artifacts").mkdir(exist_ok=True);Path("artifacts/gdm35.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
