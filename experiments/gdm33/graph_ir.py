import json,re
from pathlib import Path
# GDM33: turn Federation SDL into a reusable graph-model intermediate representation.
SDLS={"products":'''type Query { product: Product! }
type Product @key(fields: "id") { id: ID! created: String! updated: String! name: String! supplier: Supplier }
type Supplier @key(fields: "id") { id: ID! name: String! }''',"reviews":'''type Review @key(fields: "id") { id: ID! rating: Int! author: User! product: Product! }
type User @key(fields: "id") { id: ID! name: String! }
extend type Product @key(fields: "id") { id: ID! reviews: [Review!]! }'''}
SCALARS={"ID","String","Int","Float","Boolean"}
def parse(sdl,owner):
 out=[]
 for typ,header,body in re.findall(r'(?:extend\s+)?type\s+(\w+)([^\{]*)\{([^}]*)\}',sdl,re.S):
  key=set()
  for fs in re.findall(r'@key\s*\(\s*fields\s*:\s*"([^"]+)"',header):key.update(re.findall(r'\w+',fs))
  for field,raw in re.findall(r'(\w+)\s*(?:\([^)]*\))?\s*:\s*([\[\]!\w]+)',body):
   base=re.sub(r'[\[\]!]','',raw);out.append({"coordinate":f"{typ}.{field}","parent":typ,"field":field,"return_type":base,"kind":"scalar" if base in SCALARS else "relationship","list":"[" in raw,"non_null":raw.endswith("!"),"owner":owner,"key":field in key})
 return out
entries=[]
for o,s in SDLS.items():entries+=parse(s,o)
# Merge repeated Federation coordinates while preserving owner declarations.
merged={}
for x in entries:
 c=x["coordinate"]
 if c not in merged:merged[c]={**x,"owners":[x["owner"]]}
 elif x["owner"] not in merged[c]["owners"]:merged[c]["owners"].append(x["owner"])
nodes=sorted(merged.values(),key=lambda x:x["coordinate"])
edges=[]
for x in nodes:
 if x["kind"]=="relationship":edges.append({"from":x["parent"],"field":x["field"],"to":x["return_type"],"coordinate":x["coordinate"],"owners":x["owners"]})
# Candidate descriptors are generated mechanically from IR.
def descriptor(x):
 return {"coordinate":x["coordinate"],"text":f"coordinate {x['coordinate']}; parent {x['parent']}; field {x['field']}; {x['kind']}; returns {'list of ' if x['list'] else ''}{x['return_type']}; owned by {', '.join(x['owners'])}; {'entity key' if x['key'] else 'non-key'}"}
descriptors=[descriptor(x) for x in nodes]
out={"version":1,"subgraphs":list(SDLS),"coordinates":nodes,"edges":edges,"candidate_descriptors":descriptors}
Path("artifacts").mkdir(exist_ok=True);Path("artifacts/graph-ir.json").write_text(json.dumps(out,indent=2));print(json.dumps({"coordinates":len(nodes),"relationships":len(edges),"subgraphs":len(SDLS)},indent=2))
