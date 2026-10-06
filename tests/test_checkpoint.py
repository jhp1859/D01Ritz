import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

class CheckpointTests(unittest.TestCase):
    def test_interrupted_restart_equals_uninterrupted(self):
        root=Path(__file__).resolve().parents[1]
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        with tempfile.TemporaryDirectory() as tmp:
            base=[sys.executable,str(root/'scripts/run_chain.py'),'--m','10','--seed','7119','--samples','32','--burn','20','--stride','1','--stage','benchmark','--out']
            resumed=Path(tmp)/'resumed';full=Path(tmp)/'full'
            proc=subprocess.Popen(base+[str(resumed)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env)
            line=proc.stdout.readline()
            self.assertIn('samples=16',line)
            proc.terminate();proc.communicate(timeout=30)
            subprocess.run(base+[str(resumed)],check=True,stdout=subprocess.DEVNULL,env=env)
            subprocess.run(base+[str(full)],check=True,stdout=subprocess.DEVNULL,env=env)
            for name in ['F.npy','G.npy','states.npy']:
                np.testing.assert_array_equal(np.load(resumed/name),np.load(full/name))
