import json
from pathlib import Path
# GDM37: complete deterministic schema/operation validation gate.
cases=[
{"id":"catalog","subgraphs":{"products":'''extend schema @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@key"])
type Query { product: Product! }
type Product @key(fields: "id") { id: ID! name: String! created: String! }
'''},"operations":["query Generated { product { name created } }"]},
{"id":"reviews","subgraphs":{"products":'''extend schema @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@key"])
type Query { product: Product! }
type Product @key(fields: "id") { id: ID! }
''',"reviews":'''extend schema @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@key"])
type Review @key(fields: "id") { id: ID! rating: Int! }
extend type Product @key(fields: "id") { id: ID! @external reviews: [Review!]! }
'''},"operations":["query Generated { product { reviews { rating } } }"]}]
# Write artifacts and Rover configs. CI composes them for real.
root=Path("artifacts");root.mkdir(exist_ok=True)
manifest=[]
for c in cases:
 d=root/c["id"];d.mkdir(exist_ok=True)
 config=["federation_version: =2.11.0","subgraphs:"]
 for owner,sdl in c["subgraphs"].items():
  (d/f"{owner}.graphql").write_text(sdl)
  config += [f"  {owner}:","    routing_url: http://"+owner+"/graphql","    schema:",f"      file: {owner}.graphql"]
 (d/"supergraph.yaml").write_text("\n".join(config)+"\n")
 (d/"operations.graphql").write_text("\n".join(c["operations"])+"\n")
 manifest.append({"id":c["id"],"subgraphs":list(c["subgraphs"]),"operations":c["operations"]})
(root/"manifest.json").write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
