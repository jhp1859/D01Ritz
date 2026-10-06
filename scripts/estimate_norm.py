#!/usr/bin/env python3
"""Independent uniform P0/P1 importance estimate of the absolute projected norm.

Uniform sector counts are exact; this does not use harmonic-mean estimates.
Precision may be poor, which must be retained in the coefficient metadata.
"""
import argparse
import math
import resource
import time
from pathlib import Path
import numpy as np
from d01_sampling import Amplitudes,dump,sha


def main():
    started=time.monotonic();cpu_started=time.process_time()
    p=argparse.ArgumentParser();p.add_argument('--m',type=int,required=True)
    p.add_argument('--result',type=Path,required=True);p.add_argument('--samples',type=int,default=4096)
    p.add_argument('--seed',type=int,default=306100);args=p.parse_args()
    root=Path(__file__).resolve().parents[1];z=np.load(root/f'supports/state_M{args.m}.npz')
    e=Amplitudes(z['C'],z['alpha_bits'],z['beta_bits']);c=np.load(args.result/'coefficients.npz')['coefficients_S_over_Z']
    n,k=36,16;counts=[math.comb(n,k)*math.comb(n-k,k),math.comb(n,k)*k*math.comb(n-k,k-1)]
    total=sum(counts);rng=np.random.default_rng(args.seed);v=[]
    for _ in range(args.samples):
        up=rng.choice(n,k,replace=False);other=np.setdiff1d(np.arange(n),up)
        d=int(rng.random()<counts[1]/total)
        down=rng.choice(other,k-d,replace=False)
        if d:down=np.append(down,rng.choice(up))
        state=(sum(1<<int(i) for i in up),sum(1<<int(i) for i in down))
        v.append(total*float(e.direct(state)@c)**2)
    v=np.asarray(v);norm=float(v.mean());se=float(v.std(ddof=1)/np.sqrt(len(v)))
    np.savez(args.result/'coefficients_physical_norm_estimate.npz',coefficients=c/np.sqrt(norm))
    dump(args.result/'normalization.json',{'seed':args.seed,'uniform_samples':args.samples,'sector_configuration_counts':counts,
        'norm_before_normalization':norm,'norm_SE':se,'relative_norm_SE':se/norm,
        'normalized_norm_estimate':1.,'normalized_norm_SE':se/norm,
        'warning':'Monte Carlo normalization, not exact; independent uniform sample precision is reported.',
        'wall_seconds':time.monotonic()-started,'CPU_seconds':time.process_time()-cpu_started,
        'peak_rss_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'support_sha256':sha(root/f'supports/state_M{args.m}.npz'),
        'coefficients_sha256':sha(args.result/'coefficients_physical_norm_estimate.npz')})
if __name__=='__main__':main()
