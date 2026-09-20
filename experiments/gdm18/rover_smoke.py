import os,subprocess,tempfile
from pathlib import Path
s='''extend schema @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@key"])
type Query { product: Product! }
type Product @key(fields: "id") { id: ID! created: String! }
'''
with tempfile.TemporaryDirectory() as d:
 p=Path(d);(p/"products.graphql").write_text(s)
 (p/"supergraph.yaml").write_text('federation_version: =2.0\nsubgraphs:\n  products:\n    routing_url: http://products/graphql\n    schema:\n      file: products.graphql\n')
 r=subprocess.run(["rover","supergraph","compose","--config",str(p/"supergraph.yaml")],cwd=d,capture_output=True,text=True)
 print("returncode",r.returncode);print("STDOUT",r.stdout);print("STDERR",r.stderr)
