import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_ritz import solve,actions

class RitzSolverTests(unittest.TestCase):
    def test_dual_matches_dense_canonical(self):
        rng=np.random.default_rng(61)
        F=rng.normal(size=(12,30));G=rng.normal(size=F.shape)
        c,r=solve(F,G,1e-8)
        S=F.T@F/len(F);A=(F.T@G+G.T@F)/(2*len(F))
        vals,V=eigh(S);keep=vals>1e-8*vals[-1];X=V[:,keep]/np.sqrt(vals[keep])
        exact=eigh(X.T@A@X,eigvals_only=True)[0]
        self.assertAlmostEqual(r['energy'],exact,places=11)
        self.assertAlmostEqual(c@S@c,1,places=12)
        ac,sc=actions(F,G,c)
        np.testing.assert_allclose(ac,A@c,atol=1e-13)
        np.testing.assert_allclose(sc,S@c,atol=1e-13)
        self.assertEqual(r['empirical_nullity'],18)
