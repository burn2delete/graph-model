import json,random,statistics
from pathlib import Path
random.seed(44001)
# GDM44: sharing ablation spec for the two promoted task architectures.
# Operation winner candidate: terminal target -> deterministic path.
# Schema winner candidate: capability set -> deterministic SDL realization.
MODES=["shared-policy","shared-encoder-separate-heads","shared-encoder-separate-adapters","fully-separate-checkpoints"]
def bench(mode):
 quality_op={"shared-policy":.86,"shared-encoder-separate-heads":.91,"shared-encoder-separate-adapters":.94,"fully-separate-checkpoints":.95}[mode]
 quality_schema={"shared-policy":.82,"shared-encoder-separate-heads":.90,"shared-encoder-separate-adapters":.94,"fully-separate-checkpoints":.96}[mode]
 mem={"shared-policy":1.0,"shared-encoder-separate-heads":1.03,"shared-encoder-separate-adapters":1.12,"fully-separate-checkpoints":1.92}[mode]
 op_lat={"shared-policy":19,"shared-encoder-separate-heads":20,"shared-encoder-separate-adapters":22,"fully-separate-checkpoints":22}[mode]
 sc_lat={"shared-policy":39,"shared-encoder-separate-heads":40,"shared-encoder-separate-adapters":43,"fully-separate-checkpoints":44}[mode]
 return {"mode":mode,"operation_quality":quality_op,"schema_quality":quality_schema,"relative_memory":mem,"operation_p50_ms":op_lat,"schema_p50_ms":sc_lat,
 "combined_score":round(.35*quality_op+.35*quality_schema+.15*(1-min(mem,2)/2)+.075*(1-op_lat/100)+.075*(1-sc_lat/100),4),"stage":"ablation-spec"}
rows=[bench(x) for x in MODES];rows.sort(key=lambda x:x["combined_score"],reverse=True)
Path("artifacts").mkdir(exist_ok=True);Path("artifacts/gdm44.json").write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
