"""Independent creation/annihilation oracle, including the 72-mode 6x6 case."""
import sys
import unittest
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from d01_sampling import Amplitudes, initial, lattice, proposal
from d01_operator import doublons, heff01_connections


def oracle_hops(state,h):
    n=len(h);mask=int(state[0])|(int(state[1])<<n)
    for spin in range(2):
        for i,j in zip(*np.nonzero(h)):
            src=int(j)+spin*n;dst=int(i)+spin*n
            if not (mask>>src)&1:continue
            sign=(-1)**sum((mask>>k)&1 for k in range(src))
            mid=mask^(1<<src)
            if (mid>>dst)&1:continue
            sign*=(-1)**sum((mid>>k)&1 for k in range(dst))
            end=mid|(1<<dst)
            yield (end&((1<<n)-1),end>>n),sign*h[i,j]


def oracle(state,h):
    out=defaultdict(float);d=doublons(state);out[state]+=8*d
    for mid,v in oracle_hops(state,h):
        if doublons(mid)<=1:out[mid]+=v
        elif d==1 and doublons(mid)==2:
            for end,w in oracle_hops(mid,h):
                if doublons(end)==1:out[end]-=v*w/8
    return {k:v for k,v in out.items() if abs(v)>1e-14}


class SamplingTests(unittest.TestCase):
    def test_independent_signed_6x6(self):
        rng=np.random.default_rng(812)
        h=lattice(6)
        for _ in range(3):
            d0=initial(rng)
            d1=next(s for s,v in oracle_hops(d0,h) if doublons(s)==1)
            for state in [d0,d1]:
                self.assertEqual(oracle(state,h),heff01_connections(state,h))

    def test_updates_all_6x6_neighbors(self):
        z=np.load(Path(__file__).resolve().parents[1]/'reference/fermi_sw400/state_M400.npz')
        e=Amplitudes(z['C'],z['alpha_bits'][:10],z['beta_bits'][:10])
        rng=np.random.default_rng(91);h=lattice(6);state=initial(rng)
        d1=next(s for s,v in oracle_hops(state,h) if doublons(s)==1)
        for state in [state,d1]:
            p,ctx=e.prepare(state)
            np.testing.assert_allclose(p,e.direct(state),rtol=1e-7,atol=1e-22)
            for target in oracle(state,h):
                np.testing.assert_allclose(e.updated(target,ctx),e.direct(target),rtol=1e-6,atol=1e-22)

    def test_small_amplitudes_and_proposals(self):
        rng=np.random.default_rng(413)
        C=np.linalg.qr(rng.normal(size=(4,4)))[0]
        e=Amplitudes(C,[3,5,9],[6,10,12]);state=(3,6)
        _,ctx=e.prepare(state)
        for a in [3,5,9,6,10,12]:
            for b in [3,5,9,6,10,12]:
                np.testing.assert_allclose(e.updated((a,b),ctx),e.direct((a,b)),atol=1e-13)
        seen=set()
        for _ in range(1000):
            state=proposal(state,rng,n=4);seen.add(doublons(state))
            self.assertLessEqual(doublons(state),1)
            self.assertEqual(bin(int(state[0])).count('1'),2)
        self.assertEqual(seen,{0,1})
