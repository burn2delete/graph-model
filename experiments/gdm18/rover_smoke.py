import os,subprocess,tempfile
from pathlib import Path
s='''extend schema @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@key"])
type Query { product: Product! }
type Product @key(fields: "id") { id: ID! created: String! }
'''
with tempfile.TemporaryDirectory() as d:
 p=Path(d);(p/"products.graphql").write_text(s)
 (p/"supergraph.yaml").write_text('subgraphs:\n  products:\n    routing_url: http://products/graphql\n    schema:\n      file: products.graphql\nfederation_version: =2.11.0\n')
 print((p/"supergraph.yaml").read_text())
 r=subprocess.run(["rover","supergraph","compose","--config","supergraph.yaml"],cwd=d,capture_output=True,text=True)
 print("returncode",r.returncode);print("STDOUT",r.stdout);print("STDERR",r.stderr)
