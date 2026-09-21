import json,collections
from pathlib import Path
# GDM34: recursive operation and ownership-aware planning over Graph IR.
IR={
"coordinates":[
{"coordinate":"Query.product","parent":"Query","field":"product","return_type":"Product","kind":"relationship","owners":["products"]},
{"coordinate":"Product.id","parent":"Product","field":"id","return_type":"ID","kind":"scalar","owners":["products","reviews"]},
{"coordinate":"Product.name","parent":"Product","field":"name","return_type":"String","kind":"scalar","owners":["products"]},
{"coordinate":"Product.supplier","parent":"Product","field":"supplier","return_type":"Supplier","kind":"relationship","owners":["products"]},
{"coordinate":"Product.reviews","parent":"Product","field":"reviews","return_type":"Review","kind":"relationship","owners":["reviews"]},
{"coordinate":"Review.rating","parent":"Review","field":"rating","return_type":"Int","kind":"scalar","owners":["reviews"]},
{"coordinate":"Review.author","parent":"Review","field":"author","return_type":"User","kind":"relationship","owners":["reviews"]},
{"coordinate":"User.name","parent":"User","field":"name","return_type":"String","kind":"scalar","owners":["reviews"]},
{"coordinate":"Supplier.name","parent":"Supplier","field":"name","return_type":"String","kind":"scalar","owners":["products"]},
]}
byparent=collections.defaultdict(list)
for x in IR["coordinates"]:byparent[x["parent"]].append(x)
def find_path(root,target):
 q=collections.deque([(root,[])]);seen=set()
 while q:
  typ,path=q.popleft()
  if typ in seen:continue
  seen.add(typ)
  for x in byparent[typ]:
   np=path+[x]
   if x["coordinate"]==target:return np
   if x["kind"]=="relationship":q.append((x["return_type"],np))
 return None
def operation(path):
 # Build nested selection from path after Query root.
 fields=[x["field"] for x in path]
 s=fields[-1]
 for f in reversed(fields[:-1]):s=f+" { "+s+" }"
 return "query Generated { "+s+" }"
def plan(path):
 steps=[];current=None
 for x in path:
  owner=x["owners"][0]
  if current!=owner:
   steps.append({"kind":"Fetch","service":owner,"at":x["parent"]})
   current=owner
  steps[-1].setdefault("coordinates",[]).append(x["coordinate"])
 return steps
targets=["Product.name","Supplier.name","Review.rating","User.name"]
results=[]
for target in targets:
 p=find_path("Query",target);results.append({"target":target,"path":[x["coordinate"] for x in p],"operation":operation(p),"plan":plan(p)})
out={"targets":len(targets),"resolved":sum(x["path"] is not None for x in results),"results":results}
Path("artifacts").mkdir(exist_ok=True);Path("artifacts/gdm34.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
