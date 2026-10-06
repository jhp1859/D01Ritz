#!/usr/bin/env python3
"""Restartable sample bank: F=phi/sqrt(w), G=Hphi/sqrt(w), no MxM arrays."""
import argparse
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from d01_sampling import *


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--m',type=int,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--samples',type=int,default=512);p.add_argument('--burn',type=int,default=4000)
    p.add_argument('--stride',type=int,default=20);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stage',choices=['train','holdout','benchmark'],required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1];source=root/f'supports/state_M{args.m}.npz'
    import json
    support_manifest=json.loads((root/'supports/MANIFEST.json').read_text())
    assert support_manifest['origin']=='single_bare_Fermi'
    assert support_manifest['seed_sha256']==sha(root/'inputs/state_M1_fermi.npz')
    assert sha(source)==support_manifest['states'][str(args.m)]['sha256']
    start=time.monotonic();cpu0=time.process_time()
    z=np.load(source);e=Amplitudes(z['C'],z['alpha_bits'],z['beta_bits']);h=lattice(6)
    rng=np.random.default_rng(args.seed);checkpoint=args.out/'checkpoint.json'
    options={'m':args.m,'seed':args.seed,'samples':args.samples,'burn':args.burn,'stride':args.stride,'stage':args.stage,'input_sha256':sha(source),'origin_seed_sha256':support_manifest['seed_sha256']}
    accepted=0;cross=0;proposed=0;done=0;prior_wall=0;prior_cpu=0
    if checkpoint.exists():
        import json
        cp=json.loads(checkpoint.read_text());assert cp['options']==options
        state=tuple(cp['state']);rng.bit_generator.state=cp['rng'];done=cp['done']
        accepted=cp['accepted'];cross=cp['cross'];proposed=cp['proposed']
        prior_wall=cp['wall_seconds'];prior_cpu=cp['cpu_seconds'];mode='r+'
    else:
        state=initial(rng);mode='w+'
    F=np.lib.format.open_memmap(args.out/'F.npy',mode=mode,dtype='float64',shape=(args.samples,args.m))
    G=np.lib.format.open_memmap(args.out/'G.npy',mode=mode,dtype='float64',shape=(args.samples,args.m))
    states=np.lib.format.open_memmap(args.out/'states.npy',mode=mode,dtype='uint64',shape=(args.samples,2))
    phi,ctx=e.prepare(state);w=float(phi@phi)
    def save(finished=False):
        F.flush();G.flush();states.flush()
        rec=dict(options=options,state=list(map(int,state)),rng=rng.bit_generator.state,done=done,accepted=accepted,cross=cross,proposed=proposed,
            wall_seconds=prior_wall+time.monotonic()-start,cpu_seconds=prior_cpu+time.process_time()-cpu0,
            peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            python=platform.python_version(),numpy=np.__version__,host=platform.node(),condor_ad=os.environ.get('_CONDOR_JOB_AD'),finished=finished)
        dump(checkpoint,rec)
        if finished:
            rec['files']={name:{'path':str((args.out/name).resolve()),'sha256':sha(args.out/name)} for name in ['F.npy','G.npy','states.npy']}
            d=np.array([doublons(s) for s in states]);rec['D1_fraction']=float(d.mean());rec['tau_D']=tau(d)
            c=z['coefficients'].real;num=(F@c)*(G@c);den=(F@c)**2
            rec['tau_starting_numerator']=tau(num);rec['tau_starting_denominator']=tau(den)
            dump(args.out/'manifest.json',rec)
    def step():
        nonlocal state,phi,ctx,w,accepted,cross,proposed
        candidate=proposal(state,rng);proposed+=1
        if candidate==state:return
        new=e.updated(candidate,ctx);nw=float(new@new)
        if nw>0 and rng.random()<min(1.,nw/w):
            cross+=int(doublons(candidate)!=doublons(state));accepted+=1;state=candidate
            phi,ctx=e.prepare(state);w=float(phi@phi)
    if not checkpoint.exists():
        for _ in range(args.burn):step()
        save()
    while done<args.samples:
        for _ in range(args.stride):step()
        b=local(e,state,ctx,h);F[done]=phi/np.sqrt(w);G[done]=b/np.sqrt(w);states[done]=state;done+=1
        if done%16==0:save();print(f'M={args.m} {args.stage} seed={args.seed} samples={done} seconds={time.monotonic()-start:.1f}',flush=True)
    save(True)

if __name__=='__main__':main()
