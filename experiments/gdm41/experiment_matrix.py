import json,math,os,random,time
from pathlib import Path
SEED=41001; random.seed(SEED)

# GDM41: a parallel experiment matrix for the two now-separated tasks.
# Fast synthetic harness: each arm isolates one architectural hypothesis before expensive backbone runs.

OP_CASES=[
 {"q":"return each review author's display name","path":["Product.reviews","Review.author","User.name"]},
 {"q":"return review scores","path":["Product.reviews","Review.rating"]},
 {"q":"return product creation time","path":["Product.created"]},
 {"q":"return product display name","path":["Product.name"]},
]
SCHEMA_CASES=[
 {"q":"products have a public display name","actions":["ADD_TYPE Product","ADD_FIELD Product.name:String!"]},
 {"q":"products expose their original registration time","actions":["ADD_TYPE Product","ADD_FIELD Product.created:String!"]},
 {"q":"products have reviews with numeric ratings","actions":["ADD_TYPE Review","ADD_REL Product.reviews:[Review!]!","ADD_FIELD Review.rating:Int!"]},
 {"q":"reviews have authors whose public names are exposed","actions":["ADD_TYPE User","ADD_REL Review.author:User!","ADD_FIELD User.name:String!"]},
]
THEORIES=[
 ("op_path_state","operation","score legal next actions using current parent type + path state"),
 ("op_terminal_only","operation","predict terminal coordinate then derive path deterministically"),
 ("op_retrieval_top2","operation","retrieve top-2 legal outgoing actions before classification"),
 ("op_retrieval_top4","operation","retrieve top-4 legal outgoing actions before classification"),
 ("op_pairwise","operation","pairwise typed option comparison"),
 ("op_independent","operation","independent candidate scoring"),
 ("op_shared_encoder","operation","shared frozen encoder with operation-specific head"),
 ("op_task_adapter","operation","operation-specific adapter"),
 ("schema_action_state","schema","score schema mutation actions against partial Graph IR"),
 ("schema_capability_set","schema","predict required capability set then deterministically realize SDL"),
 ("schema_pairwise","schema","pairwise typed schema-action comparison"),
 ("schema_multilabel","schema","independent multilabel capability prediction"),
 ("schema_shared_encoder","schema","shared frozen encoder with schema-specific head"),
 ("schema_task_adapter","schema","schema-specific adapter"),
 ("schema_structural_plus_semantic","schema","separate structural supervision from semantic supervision"),
 ("schema_semantic_only","schema","semantic supervision without generated structural curriculum"),
]
# Proxy expectations are intentionally hypotheses, not model results.
# They prioritize experiments that match task constraints and minimize learned surface area.
def proxy(name,task,theory):
 quality=.55; speed=1.0; risk=.25
 if "path_state" in name: quality+=.25; speed+=.15; risk-=.08
 if "terminal_only" in name: quality+=.18; speed+=.25; risk+=.03
 if "top2" in name: quality+=.22; speed-=.08; risk-=.05
 if "top4" in name: quality+=.24; speed-=.18; risk-=.04
 if "pairwise" in name: quality+=.17; speed-=.12; risk-=.04
 if "independent" in name: quality+=.08; speed+=.08
 if "task_adapter" in name: quality+=.16; speed-=.03; risk-=.04
 if "shared_encoder" in name: quality+=.10; speed+=.12
 if "action_state" in name: quality+=.24; speed-=.08; risk-=.08
 if "capability_set" in name: quality+=.20; speed+=.08; risk-=.05
 if "multilabel" in name: quality+=.13; speed+=.10
 if "structural_plus_semantic" in name: quality+=.27; speed-=.05; risk-=.09
 if "semantic_only" in name: quality+=.10; speed+=.08; risk+=.06
 return {"hypothesis_quality":round(min(1,quality),3),"relative_speed":round(speed,3),"semantic_risk":round(max(0,risk),3)}
rows=[]
for name,task,theory in THEORIES:
 r={"arm":name,"task":task,"theory":theory,**proxy(name,task,theory)}
 r["priority_score"]=round(.65*r["hypothesis_quality"]+.25*min(1.2,r["relative_speed"])/1.2+.10*(1-r["semantic_risk"]),3)
 rows.append(r)
rows.sort(key=lambda x:x["priority_score"],reverse=True)
out={"seed":SEED,"operation_cases":len(OP_CASES),"schema_cases":len(SCHEMA_CASES),"arms":rows}
Path("artifacts").mkdir(exist_ok=True);Path("artifacts/gdm41-matrix.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
