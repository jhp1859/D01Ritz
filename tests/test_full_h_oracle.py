import json
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from full_h_oracle import full_h_matrix

class FullHOracleTests(unittest.TestCase):
    def test_archived_sw400_energy_and_residual(self):
        root=Path(__file__).resolve().parents[1]
        z=np.load(root/'reference/fermi_sw400/state_M400.npz');ints=np.load(root/'inputs/integrals_sw.npz')
        H=full_h_matrix(z['alpha_bits'],z['beta_bits'],ints['h1'].real,ints['eri'].real)
        c=z['coefficients'].real;E=float(c@H@c)
        ref=json.loads((root/'reference/fermi_sw400/growth.json').read_text())['E_var']
        self.assertAlmostEqual(E,ref,places=9)
        # Archived Davidson used energy-change OR residual stopping. Its
        # measured residual is 1.2813e-6; energy agrees with exact diagonalization.
        self.assertLess(np.linalg.norm(H@c-E*c),2e-6)
        self.assertAlmostEqual(float(np.linalg.eigvalsh(H)[0]),E,places=9)
