import unittest
from experiments.followup65 import run

class GDM65Contracts(unittest.TestCase):
 def opts(self):
  return [{"path":["root","reviews","author","name"],"types":["X","Y","String!"]},{"path":["root","reviews","author","id"],"types":["X","Y","ID!"]},{"path":["root","reviews","moderator","name"],"types":["X","Y","String!"]}]
 def test_matrix_dimensions(self):
  self.assertEqual(len(run.ARMS),5);self.assertEqual(len(run.ARMS)*2,10);self.assertEqual(run.ambiguity_feature_dim(128),1311);self.assertEqual(run.ambiguity_feature_dim(768),7711);self.assertEqual(run.RELATION_FEATURES,20)
 def test_zero_padding_bounds(self):
  o=self.opts()[:2];self.assertEqual(run._relation_channel(o,"structured-zero-control"),[0.0]*20)
  for m in run.MODES:
   x=run._relation_channel(o,m);self.assertEqual(len(x),20);self.assertEqual(x[8:],[0.0]*12);self.assertTrue(all(0<=v<=1 for v in x))
 def test_relations(self):
  a,i,m=self.opts();self.assertEqual((run._same_parent(a,i),run._same_leaf(a,i),run._same_named_type(a,i)),(1.0,0.0,0.0));self.assertEqual((run._same_parent(a,m),run._same_leaf(a,m),run._same_named_type(a,m)),(0.0,1.0,1.0))
 def test_provenance(self):
  self.assertEqual(run.CANONICAL_GDM64_SOURCE,"6fe0f89630324fa18c18175c1d7c64ba614a1cff");self.assertEqual(run.CANONICAL_GDM64_AUDIT_RUN,36007882683);self.assertEqual(run.CANONICAL_GDM64_PROMOTION,"fec580e4d41ad03eedd8ba182cfcbcbb3c313080")
if __name__=="__main__":unittest.main()
