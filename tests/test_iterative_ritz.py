import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.linalg import eigh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from iterative_ritz import solve_iterative
class IterativeRitzTests(unittest.TestCase):
    def test_matches_dense_generalized_oracle(self):
        rng=np.random.default_rng(731)
        F=rng.normal(size=(180,32));G=rng.normal(size=F.shape)
        S=F.T@F/len(F);A=(F.T@G+G.T@F)/(2*len(F))
        E=eigh(A,S,eigvals_only=True)[0]
        c,r=solve_iterative(F,G,0.,maxiter=160)
        self.assertAlmostEqual(r['energy'],E,places=8)
        self.assertLess(r['relative_generalized_residual'],1e-6)
        self.assertAlmostEqual(c@S@c,1,places=11)
