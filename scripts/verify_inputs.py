#!/usr/bin/env python3
"""Verify the true 6x6 bare-Fermi seed and archived SW400 precedent."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}

def main():
    manifest = json.loads((ROOT/'inputs/MANIFEST.json').read_text())
    assert manifest['status'] == 'FERMI_SEED_ONLY_TARGET_LADDER_NOT_GENERATED'
    for name, expected in manifest['files'].items():
        actual = sha256(ROOT/name)
        assert actual == expected, (name, actual, expected)
    orbitals = load(ROOT/'inputs/fermi_orbitals.npz')
    seed = load(ROOT/'inputs/state_M1_fermi.npz')
    example = load(ROOT/'reference/fermi_sw400/state_M400.npz')
    c = np.asarray(orbitals['C'], float)
    occ = np.asarray(orbitals['occ'], int)
    bits = np.uint64(sum(1 << int(i) for i in occ))
    assert c.shape == (36,36) and len(occ) == 16
    assert np.max(abs(c.T@c-np.eye(36))) < 2e-12
    assert np.array_equal(seed['alpha_bits'], np.array([bits], np.uint64))
    assert np.array_equal(seed['beta_bits'], np.array([bits], np.uint64))
    assert np.array_equal(seed['coefficients'], np.array([1.0]))
    assert np.array_equal(seed['C'], c)
    assert int(seed['reference_bits']) == int(bits)
    assert len(example['coefficients']) == 400
    assert int(example['reference_bits']) == int(bits)
    assert np.max(abs(example['C']-c)) < 1e-14
    assert len(np.where((example['alpha_bits']==bits)&(example['beta_bits']==bits))[0]) == 1
    assert all(bin(int(v)).count('1') == 16 for v in example['alpha_bits'])
    assert all(bin(int(v)).count('1') == 16 for v in example['beta_bits'])
    with np.load(ROOT/'inputs/integrals_sw.npz', allow_pickle=False) as z:
        assert z['h1'].shape == (36,36) and z['eri'].shape == (36,)*4
        assert np.max(abs(c.T @ np.asarray(orbitals['K'],float) @ c-z['h1'])) < 3e-14
    print('PASS: bare-Fermi seed, SW400 Fermi provenance, original orbitals and integrals')
    print('TARGET_SUPPORTS_ABSENT: M=10,100,1000,10000,100000 must be Fermi-grown')

if __name__ == '__main__':
    main()
