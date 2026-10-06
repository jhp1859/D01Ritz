"""Frozen-support D01 amplitudes and reversible site Metropolis sampler.

No spin projection. Unique spin determinants are cached, not new generators.
"""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from d01_operator import doublons, heff01_connections


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def occ(bits, n):
    return np.array([i for i in range(n) if (int(bits) >> i) & 1], dtype=int)


def lattice(side):
    h = np.zeros((side*side, side*side))
    for x in range(side):
        for y in range(side):
            i = x*side+y
            for dx,dy in [(1,0),(0,1)]:
                if x+dx < side and y+dy < side:
                    j=(x+dx)*side+y+dy; h[i,j]=h[j,i]=-1
    return h


class Amplitudes:
    def __init__(self, C, alpha, beta):
        assert np.max(np.abs(np.asarray(C).imag)) < 1e-13
        self.C = np.asarray(C).real
        self.n = len(C)
        self.rows=[]; self.owner=[]
        for bits in [alpha,beta]:
            unique, owner=np.unique(bits,return_inverse=True)
            orbitals=np.array([occ(b,self.n) for b in unique])
            self.rows.append(self.C[:,orbitals].transpose(1,0,2).copy())
            self.owner.append(owner.astype(np.int64))

    def direct(self,state):
        ds=[np.linalg.det(rows[:,occ(bits,self.n),:]) for rows,bits in zip(self.rows,state)]
        return ds[0][self.owner[0]]*ds[1][self.owner[1]]

    def prepare(self,state):
        ctx=[]
        for rows,bits in zip(self.rows,state):
            labels=occ(bits,self.n); mat=rows[:,labels,:]
            det=np.linalg.det(mat)
            # pinv handles exact standing-wave nodes in batches; unstable
            # members always use direct determinants for updates.
            inv=np.linalg.pinv(mat,rcond=1e-12)
            err=np.max(np.abs(mat@inv-np.eye(len(labels))),axis=(1,2))
            good=(err<1e-7)&(np.max(np.abs(inv),axis=(1,2))<1e10)
            ctx.append((labels,det,inv,good))
        return self.updated(state,ctx),ctx

    def updated(self,state,ctx):
        ds=[]
        for rows,bits,(old,det,inv,good) in zip(self.rows,state,ctx):
            new=occ(bits,self.n)
            removed=np.flatnonzero(~np.isin(old,new)); added=new[~np.isin(new,old)]
            if len(added)==0: ds.append(det); continue
            if len(added)>2:
                ds.append(np.linalg.det(rows[:,new,:])); continue
            order=old.copy();order[removed]=added
            sign=(-1)**sum(int(order[i]>order[j]) for i in range(len(order)) for j in range(i+1,len(order)))
            small=rows[:,added,:]@inv[:,:,removed]
            ratio=small[:,0,0] if len(added)==1 else small[:,0,0]*small[:,1,1]-small[:,0,1]*small[:,1,0]
            val=det*ratio*sign
            bad=(~good)|(~np.isfinite(val))|(np.abs(ratio)<1e-9)
            if np.any(bad): val[bad]=np.linalg.det(rows[bad][:,new,:])
            ds.append(val)
        return ds[0][self.owner[0]]*ds[1][self.owner[1]]


def initial(rng,n=36,k=16):
    p=rng.permutation(n)
    return sum(1<<int(i) for i in p[:k]),sum(1<<int(i) for i in p[k:2*k])


def proposal(state,rng,n=36):
    # Mixture of symmetric kernels. Fixed ordered site pair and spin flip
    # or exchange; invalid configurations are explicit self transitions.
    i,j=map(int,rng.choice(n,2,replace=False))
    a,b=map(int,state);mask=(1<<i)|(1<<j)
    if rng.random()<0.5:
        spin=int(rng.integers(2));bits=(a,b)[spin]
        if ((bits>>i)&1)==((bits>>j)&1): return state
        out=(a^mask,b) if spin==0 else (a,b^mask)
    else:
        if ((a>>i)&1)==((a>>j)&1) or ((b>>i)&1)==((b>>j)&1):return state
        out=(a^mask,b^mask)
    return out if doublons(out)<=1 else state


def local(evaluator,state,ctx,h):
    b=np.zeros(len(evaluator.owner[0]))
    for target,value in heff01_connections(state,h).items():
        b+=value.real*evaluator.updated(target,ctx)
    return b


def tau(x):
    x=np.asarray(x,float);x=x-x.mean()
    if len(x)<16 or x@x==0:return None
    f=np.fft.rfft(x,n=2*len(x));ac=np.fft.irfft(f*f.conj())[:len(x)];ac/=ac[0]
    t=0.5
    for lag in range(1,len(x)//2):
        if ac[lag]<=0:break
        t+=float(ac[lag])
        if lag>6*t:break
    return t
