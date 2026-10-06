#!/usr/bin/env python3
"""Verify the generated bare-Fermi ladder without regenerating or changing it."""
from pathlib import Path
import hashlib,json
import numpy as np
from d01_sampling import sha
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'supports/MANIFEST.json').read_text())
seed=np.load(root/'inputs/state_M1_fermi.npz')
assert manifest['origin']=='single_bare_Fermi'
assert manifest['seed_sha256']==sha(root/'inputs/state_M1_fermi.npz')
previous=root/'inputs/state_M1_fermi.npz'
for m in [100,1000,10000,100000]:
    if str(m) not in manifest['states']:break
    rec=manifest['states'][str(m)];path=root/f'supports/state_M{m}.npz';z=np.load(path)
    assert sha(path)==rec['sha256'];assert sha(previous)==rec['parent_sha256']
    assert np.array_equal(z['C'],seed['C'])
    a=z['alpha_bits'];b=z['beta_bits'];c=z['coefficients']
    assert len(c)==m and len(set(zip(map(int,a),map(int,b))))==m
    assert all(bin(int(v)).count('1')==16 for array in [a,b] for v in array)
    assert abs(c@c-1)<1e-7
    assert rec['determinant_list_sha256']==hashlib.sha256(np.column_stack([a,b]).astype('<u8').tobytes()).hexdigest()
    assert (int(seed['alpha_bits'][0]),int(seed['beta_bits'][0])) in set(zip(map(int,a),map(int,b)))
    print('PASS',m,rec['sha256']);previous=path
if '10' in manifest['states']:
    a=np.load(root/'supports/state_M10.npz');b=np.load(root/'supports/state_M100.npz')
    idx=np.argsort(-abs(b['coefficients']),kind='stable')[:10]
    assert np.array_equal(idx,a['source_indices'])
    for key in ['alpha_bits','beta_bits']:assert np.array_equal(a[key],b[key][idx])
    assert sha(root/'supports/state_M10.npz')==manifest['states']['10']['sha256']
    print('PASS 10: top-ten subset of Fermi-grown M100, not independent growth')
