#!/usr/bin/env python3
"""Canonical Ritz using row banks. Large M uses the exact empirical S range.

The dual Gram implementation never constructs M x M arrays. For N<M its
rank cannot exceed N, a sampling limitation reported explicitly, not a pass.
"""
import argparse
from pathlib import Path
import json
import resource
import time
import numpy as np
from scipy.linalg import eigh
from d01_sampling import dump,sha,tau


def actions(F,G,c):
    n=len(F)
    return (F.T@(G@c)+G.T@(F@c))/(2*n),F.T@(F@c)/n


def solve(F,G,cutoff):
    n,m=F.shape
    if m<=n:
        S=F.T@F/n;values,V=eigh(S)
        keep=values>max(cutoff*values[-1],np.finfo(float).eps*m*values[-1])
        X=V[:,keep]/np.sqrt(values[keep])
    else:
        values,V=eigh(F@F.T/n)
        keep=values>max(cutoff*values[-1],np.finfo(float).eps*n*values[-1])
        X=F.T@V[:,keep]/(np.sqrt(n)*values[keep])
    FX=F@X;GX=G@X
    H=FX.T@GX/n;H=(H+H.T)/2
    en,Y=eigh(H,subset_by_index=[0,0]);c=X@Y[:,0]
    ac,sc=actions(F,G,c);res=ac-en[0]*sc
    return c,{'energy':float(en[0]),'retained_rank':int(keep.sum()),'empirical_nullity':int(m-keep.sum()),
        'overlap_retained_condition':float(values[-1]/values[keep][0]),
        'overlap_full_condition':float(values[-1]/values[0]) if m<=n and values[0]>0 else None,
        'cutoff':cutoff,'norm_S_over_Z':float(c@sc),
        'relative_generalized_residual':float(np.linalg.norm(res)/(np.linalg.norm(ac)+abs(en[0])*np.linalg.norm(sc))),
        'projected_residual':float(np.linalg.norm(X.T@res)),
        'solver':'primal canonical' if m<=n else 'dual canonical, exact on empirical overlap range'}


def holdout(F,G,c,block=32):
    # Independent non-overlapping blocks; jackknife accounts for ratio bias
    # to first order. Block-length sensitivity is reported separately.
    n=(len(F)//block)*block;F=F[:n];G=G[:n]
    f=F@c;g=G@c
    nums=(f*g).reshape(-1,block).mean(1);dens=(f*f).reshape(-1,block).mean(1)
    E=float(nums.mean()/dens.mean());B=len(nums)
    jk=(nums.sum()-nums)/(dens.sum()-dens)
    se=float(np.sqrt((B-1)/B*np.sum((jk-jk.mean())**2)))
    # Unsymmetrized action is unbiased and independent of the fitted vector.
    a=np.array([F[i:i+block].T@g[i:i+block]/block for i in range(0,n,block)])
    s=np.array([F[i:i+block].T@f[i:i+block]/block for i in range(0,n,block)])
    am=a.mean(0);sm=s.mean(0);scale=np.linalg.norm(am)+abs(E)*np.linalg.norm(sm)
    residual=float(np.linalg.norm(am-E*sm)/scale)
    rng=np.random.default_rng(83141);boots=[]
    for _ in range(128):
        weights=rng.multinomial(B,np.ones(B)/B)/B
        eb=float(weights@nums/(weights@dens));ab=weights@a;sb=weights@s
        boots.append(np.linalg.norm(ab-eb*sb)/(np.linalg.norm(ab)+abs(eb)*np.linalg.norm(sb)))
    return {'energy':E,'energy_SE':se,'relative_residual':residual,'residual_bootstrap_SD':float(np.std(boots,ddof=1)),
        'residual_bootstrap_95pct':list(map(float,np.percentile(boots,[2.5,97.5]))),
        'block_size':block,'blocks':B,'norm_S_over_Z':float(dens.mean()),
        'tau_numerator':tau(f*g),'tau_denominator':tau(f*f)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);t=time.monotonic();cpu=time.process_time()
    banks={};manifests=[]
    for stage in ['train','holdout']:
        dirs=sorted(args.root.glob(stage+'_*'))
        assert len(dirs)>=2
        for d in dirs:
            rec=json.loads((d/'manifest.json').read_text());assert rec['finished'];manifests.append(rec)
        banks[stage]=tuple(np.concatenate([np.load(d/(name+'.npy'),mmap_mode='r') for d in dirs]) for name in ['F','G'])
    F,G=banks['train'];HF,HG=banks['holdout'];results=[]
    for cutoff in [1e-8,1e-6,1e-4]:
        c,r=solve(F,G,cutoff);r['holdout']=holdout(HF,HG,c);results.append(r)
        if cutoff==1e-6:chosen=c;chosen_r=r
    np.savez(args.out/'coefficients.npz',coefficients_S_over_Z=chosen,coefficients_euclidean=chosen/np.linalg.norm(chosen))
    stability=[]
    for n in [len(F)//2,len(F)]:
        c,r=solve(F[:n],G[:n],1e-6);r['samples']=n;r['holdout']=holdout(HF,HG,c);stability.append(r)
    seed_results=[]
    for sl in [slice(0,len(F)//2),slice(len(F)//2,None)]:
        c,r=solve(F[sl],G[sl],1e-6);r['holdout']=holdout(HF,HG,c);seed_results.append(r)
    if F.shape[1]<=1000:
        A=F.T@G/len(F);herm=float(np.linalg.norm(A-A.T)/np.linalg.norm(A))
        a1=F[:len(F)//2].T@G[:len(F)//2]*2/len(F);a2=F[len(F)//2:].T@G[len(F)//2:]*2/len(F)
        matrix_difference=float(np.linalg.norm(a1-a2)/max(np.linalg.norm(a1),np.linalg.norm(a2)))
    else:
        R=np.random.default_rng(124).normal(size=(F.shape[1],8));AR=(F@R).T@(G@R)/len(F)
        herm=float(np.linalg.norm(AR-AR.T)/np.linalg.norm(AR));matrix_difference=None
    # Conservative outcome: report all diagnostics, never equate a fit with convergence.
    failures=[];h=chosen_r['holdout']
    if h['energy_SE']>0.1:failures.append('holdout_energy_precision')
    if h['residual_bootstrap_95pct'][1]>0.05:failures.append('holdout_residual')
    if chosen_r['projected_residual']>1e-7:failures.append('training_projected_residual')
    if abs(chosen_r['energy']-h['energy'])>max(.2,3*h['energy_SE']):failures.append('train_holdout_disagreement')
    if np.ptp([r['holdout']['energy'] for r in results])>.2:failures.append('overlap_cutoff_sensitivity')
    if np.ptp([r['holdout']['energy'] for r in stability])>.2:failures.append('sample_count_stability')
    if np.ptp([r['holdout']['energy'] for r in seed_results])>.2:failures.append('seed_stability')
    for rec in manifests:
        taus=[v for k,v in rec.items() if k.startswith('tau_') and v is not None]
        if not taus or rec['done']/(2*max(taus))<100:failures.append('insufficient_effective_samples')
        if rec['cross']<20:failures.append('insufficient_sector_crossings')
    if chosen_r['empirical_nullity']>0:failures.append('rank_deficient_empirical_metric')
    rec={'status':'unresolved' if failures else 'candidate_pass_requires_longer_chain_confirmation',
        'M':int(F.shape[1]),'train_samples':len(F),'holdout_samples':len(HF),'failures':sorted(set(failures)),
        'chosen':chosen_r,'training_fixed_vector_block_diagnostics':holdout(F,G,chosen),'cutoff_study':results,'sample_count_study':stability,'seed_study':seed_results,
        'block_length_study':[holdout(HF,HG,chosen,b) for b in [16,32,64,128]],
        'raw_training_hermiticity_relative_error':herm,'independent_train_matrix_relative_difference':matrix_difference,
        'normalization':'training c^T(S/Z)c=1, unknown Z; physical c^TSc is not estimated',
        'coefficients':{'path':str((args.out/'coefficients.npz').resolve()),'sha256':sha(args.out/'coefficients.npz')},
        'chains':manifests,'sampling_CPU_hours':sum(r['cpu_seconds'] for r in manifests)/3600,
        'sampling_wall_seconds_sum':sum(r['wall_seconds'] for r in manifests),
        'sampling_peak_rss_MiB':max(r['peak_rss_MiB'] for r in manifests),
        'analysis_seconds':time.monotonic()-t,'analysis_CPU_seconds':time.process_time()-cpu,
        'analysis_peak_rss_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    dump(args.out/'result.json',rec)
    print(json.dumps({k:v for k,v in rec.items() if k in ['status','M','failures','chosen']},indent=2))

if __name__=='__main__':main()
