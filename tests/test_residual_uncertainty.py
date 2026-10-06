import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_ritz import holdout
class ResidualUncertaintyTests(unittest.TestCase):
    def test_constant_operator_has_zero_sampling_residual(self):
        rng=np.random.default_rng(619);F=rng.normal(size=(256,8));G=3.5*F;c=rng.normal(size=8)
        r=holdout(F,G,c,16,training_energy=3.5)
        self.assertAlmostEqual(r['energy'],3.5,places=12)
        self.assertLess(r['energy_SE'],1e-12)
        self.assertLess(r['residual_norm_confidence_ball_95pct'][1],1e-12)
