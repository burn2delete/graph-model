import json,random,os,re,subprocess,tempfile
from dataclasses import dataclass
SEED=18001
random.seed(SEED)

# GDM18: benchmark-integrity gate + executable mini GraphQL pipeline.
# This intentionally avoids ML dependencies so the integrity controls are always runnable.

CONCEPTS={
 "created":("String","creation timestamp","when the product was first registered"),
 "updated":("String","revision timestamp","when the product was most recently changed"),
 "name":("String","display name","public product display name"),
 "rating":("Int","review rating","numeric score attached to a review"),
}
PARA={
 "created":["when was the product first registered","give me its initial registration timestamp","return the moment this item entered the catalog"],
 "updated":["when was the product most recently changed","give me its latest revision timestamp","return the moment of the newest edit"],
 "name":["what is the public product name","return the customer-facing item label","give me the human-readable product title"],
 "rating":["what numeric score did the review receive","return the review rating","give me the numeric evaluation attached to the review"],
}

@dataclass
class Example:
 id:str; split:str; question:str; options:list; target:str

def make_examples(split,n,offset):
 out=[]
 keys=list(CONCEPTS)
 for i in range(n):
  target=keys[i%len(keys)]
  # Same generic decision kind, same candidate set, randomized order. Request must carry the answer.
  opts=[(k,CONCEPTS[k][2]) for k in keys]
  random.Random(SEED+offset+i).shuffle(opts)
  phr=PARA[target][0 if split=="train" else (1 if split=="validation" else 2)]
  out.append(Example(f"{split}-{i}",split,phr,opts,target))
 return out

def first_option(ex):return ex.options[0][0]
def kind_only(ex):return "created" # deliberately text-blind
def lexical(ex):
 q=set(re.findall(r"[a-z]+",ex.question.lower()))
 def score(o):
  return len(q & set(re.findall(r"[a-z]+",o[1].lower())))
 return max(ex.options,key=score)[0]
def oracle(ex):return ex.target

def accuracy(fn,rows):return sum(fn(x)==x.target for x in rows)/len(rows)

def audit(train,val,test):
 def pub(x):return (x.question,tuple(x.options))
 overlap=set(map(pub,train)) & set(map(pub,test))
 # Correct option position must vary for every target.
 pos={}
 for x in train+val+test:
  pos.setdefault(x.target,set()).add([k for k,_ in x.options].index(x.target))
 checks={
  "train_test_public_overlap":len(overlap),
  "all_targets_multiple_positions":all(len(v)>1 for v in pos.values()),
  "first_option_test_accuracy":accuracy(first_option,test),
  "kind_only_test_accuracy":accuracy(kind_only,test),
  "lexical_test_accuracy":accuracy(lexical,test),
 }
 checks["informative"]=checks["train_test_public_overlap"]==0 and checks["all_targets_multiple_positions"] and max(checks["first_option_test_accuracy"],checks["kind_only_test_accuracy"])<0.45
 return checks

def build_schema(selected):
 fields=[]
 for k in selected:
  typ=CONCEPTS[k][0]
  fields.append(f"  {k}: {typ}!")
 return "type Query {\n  product: Product!\n}\n\ntype Product {\n  id: ID!\n"+"\n".join(fields)+"\n}\n"

def validate_schema(sdl):
 # Small deterministic syntax/shape gate; Rover is an additional CI gate when installed.
 return sdl.count("{")==sdl.count("}") and "type Query" in sdl and "type Product" in sdl

def operation(field):
 return f"query GDM18 {{ product {{ id {field} }} }}"

def execute(field,fixture):
 return {"data":{"product":{"id":fixture["id"],field:fixture[field]}}}

def run_pipeline(rows,predict):
 fixtures=[
  {"id":"p1","created":"2026-01-01","updated":"2026-02-02","name":"One","rating":4},
  {"id":"p2","created":"2025-03-04","updated":"2026-05-06","name":"Two","rating":2},
  {"id":"p3","created":"2024-07-08","updated":"2026-09-10","name":"Three","rating":5},
 ]
 ok=0; details=[]
 for ex in rows:
  pred=predict(ex)
  sdl=build_schema([pred])
  op=operation(pred)
  valid=validate_schema(sdl) and pred in CONCEPTS
  semantic=pred==ex.target
  exec_ok=valid and semantic and all(execute(pred,f)["data"]["product"][pred]==f[ex.target] for f in fixtures)
  ok+=exec_ok
  details.append({"id":ex.id,"target":ex.target,"predicted":pred,"schema_valid":valid,"semantic":semantic,"execution_ok":exec_ok,"sdl":sdl,"operation":op})
 return ok,details

def main():
 train=make_examples("train",80,0);val=make_examples("validation",40,1000);test=make_examples("test",80,2000)
 checks=audit(train,val,test)
 base={}
 for name,fn in [("first_option",first_option),("kind_only",kind_only),("lexical",lexical),("oracle",oracle)]:
  chain,details=run_pipeline(test,fn)
  base[name]={"decision_accuracy":accuracy(fn,test),"executable_chain":chain,"worlds":len(test)}
 result={"seed":SEED,"integrity":checks,"baselines":base,"promotion_gate":{
  "benchmark_informative":checks["informative"],
  "requires_real_generated_sdl":True,
  "requires_operation_execution":True,
  "requires_rover_when_available":True,
  "requires_pretrained_vs_existing_same_environment":True,
 }}
 os.makedirs("artifacts",exist_ok=True)
 with open("artifacts/gdm18-integrity.json","w") as f:json.dump(result,f,indent=2)
 with open("artifacts/gdm18-test.jsonl","w") as f:
  for x in test:f.write(json.dumps(x.__dict__)+"\n")
 print(json.dumps(result,indent=2))
 if not checks["informative"]:raise SystemExit("benchmark integrity gate failed")

if __name__=="__main__":main()
