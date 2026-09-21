import json,re,random
from pathlib import Path
# GDM31: synthesize a typed decision curriculum directly from Federation SDL.
SDLS={
"products":'''extend schema @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@key"])
type Query { product: Product! }
type Product @key(fields: "id") { id: ID! created: String! updated: String! name: String! supplier: Supplier }
type Supplier @key(fields: "id") { id: ID! name: String! }''',
"reviews":'''extend schema @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@key"])
type Review @key(fields: "id") { id: ID! rating: Int! author: User! product: Product! }
type User @key(fields: "id") { id: ID! name: String! }
extend type Product @key(fields: "id") { id: ID! reviews: [Review!]! }'''
}
SCALARS={"ID","String","Int","Float","Boolean"}
def parse(sdl,owner):
 out=[];pat=re.compile(r'(?:extend\s+)?type\s+(\w+)([^\{]*)\{([^}]*)\}',re.S)
 for typ,header,body in pat.findall(sdl):
  keys=set(re.findall(r'@key\s*\(\s*fields\s*:\s*"([^"]+)"',header))
  keyfields=set()
  for k in keys:keyfields.update(re.findall(r'\w+',k))
  for line in body.splitlines():
   line=line.strip()
   if not line:continue
   m=re.match(r'(\w+)\s*(?:\([^)]*\))?\s*:\s*([\[\]!\w]+)',line)
   if not m:continue
   field,raw=m.groups();base=re.sub(r'[\[\]!]','',raw);kind="scalar" if base in SCALARS else "relationship"
   out.append({"id":f"{typ}.{field}","parent":typ,"field":field,"kind":kind,"return":base,"list":"[" in raw,"non_null":raw.endswith("!"),"owner":owner,"key":field in keyfields})
 return out
catalog=[]
for owner,sdl in SDLS.items():catalog+=parse(sdl,owner)
# Deterministic structural descriptions, no hand-authored business gloss.
def describe(x):
 card="collection" if x["list"] else "single"
 role="entity key" if x["key"] else x["kind"]
 return f"coordinate {x['id']}; parent type {x['parent']}; field {x['field']}; {role}; returns {card} {x['return']}; owned by {x['owner']}"
def prompts(x):
 if x["kind"]=="scalar":
  return [f"select scalar field {x['field']} on {x['parent']}",f"return the {x['field']} value from {x['parent']}",f"choose {x['return']} scalar {x['parent']}.{x['field']}"]
 return [f"traverse relationship {x['field']} from {x['parent']} to {x['return']}",f"select related {x['return']} through {x['parent']}.{x['field']}",f"follow the {x['field']} edge on {x['parent']}"]
# Pair every coordinate with structurally plausible negatives and serialize a reusable curriculum.
rows=[]
for x in catalog:
 for y in catalog:
  if x["id"]==y["id"]:continue
  hard=(x["parent"]==y["parent"] or x["field"]==y["field"] or x["return"]==y["return"] or x["kind"]!=y["kind"])
  if not hard:continue
  for q in prompts(x):
   for order in ([x,y],[y,x]):
    rows.append({"question":q,"options":[describe(z) for z in order],"option_ids":[z["id"] for z in order],"target":x["id"]})
out={"subgraphs":list(SDLS),"catalog_entries":len(catalog),"curriculum_rows":len(rows),"catalog":catalog,"examples":rows[:12]}
Path("artifacts").mkdir(exist_ok=True);Path("artifacts/catalog.json").write_text(json.dumps(catalog,indent=2));Path("artifacts/curriculum.jsonl").write_text("\n".join(json.dumps(x) for x in rows)+"\n");Path("artifacts/gdm31.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
