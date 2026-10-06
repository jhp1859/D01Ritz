#!/usr/bin/env python3
"""Audit residual-vector uncertainty without refitting or overwriting coefficients."""
import argparse,json,time,resource
from pathlib import Path
import numpy as np
from analyze_ritz import holdout
from d01_sampling import dump,sha
p=argparse.ArgumentParser();p.add_argument('--m',type=int,required=True);p.add_argument('--round',required=True);args=p.parse_args()
root=Path(__file__).resolve().parents[1];out=root/f'reports/{args.round}/M{args.m}';bank=root/f'work/{args.round}/M{args.m}'
t=time.monotonic();cpu=time.process_time();r=json.loads((out/'result.json').read_text());c=np.load(out/'coefficients.npz')['coefficients_S_over_Z']
assert sha(out/'coefficients.npz')==r['coefficients']['sha256']
dirs=sorted(bank.glob('holdout_*'))
for d in dirs:
 rec=json.loads((d/'manifest.json').read_text());assert rec['finished'];assert rec['options']['origin_seed_sha256']==sha(root/'inputs/state_M1_fermi.npz')
 for name in ['F.npy','G.npy']:assert sha(d/name)==rec['files'][name]['sha256']
F,G=(np.concatenate([np.load(d/(name+'.npy'),mmap_mode='r') for d in dirs]) for name in ['F','G'])
rows=[holdout(F,G,c,b,training_energy=r['chosen']['energy']) for b in [16,32,64,128]]
upper=max(max(v['residual_bootstrap_95pct'][1],v['residual_norm_confidence_ball_95pct'][1]) for v in rows)
dump(out/'residual_uncertainty_audit.json',{'interpretation':'Centered residual-vector bootstrap ball; triangle-inequality norm interval. Raw norm-percentiles are not treated as an unbiased confidence interval near zero.',
 'coefficient_sha256':sha(out/'coefficients.npz'),'blocks':rows,'conservative_residual_upper_95pct':upper,'passes_residual_gate':upper<=.05,
 'wall_seconds':time.monotonic()-t,'CPU_seconds':time.process_time()-cpu,'peak_rss_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})
print(args.m,upper,flush=True)
