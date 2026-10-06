#!/usr/bin/env python3
import json,os
import numpy as np
from mc_common import *
if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
z=np.load(COMMON/'generator'/'state.npz',allow_pickle=False);rng=np.random.default_rng(81721);h=lattice_k(6)
rows=[]
for spin in ('free','singlet'):
 e=CompactSpinGeneratorAmplitudes(np.asarray(z['C']).real,z['alpha_bits'],z['beta_bits'],spin)
 ae=re=0.;checks=0;old_b_rel=0.
 def old_update(state,context):
  old=context['labels'].tolist();new=e.labels(state).tolist();oset,nset=set(old),set(new)
  pos=[i for i,q in enumerate(old) if q not in nset];added=sorted(q for q in new if q not in oset)
  if not added:return e._collect(context['dets'])
  temp=list(old)
  for p,q in zip(pos,added):temp[p]=q
  sign=-1. if sum(temp[i]>temp[j] for i in range(len(temp)) for j in range(i+1,len(temp)))&1 else 1.
  sm=np.matmul(e.rows[:,added,:],context['inverse'][:,:,pos])
  ratio=sm[:,0,0] if len(added)==1 else sm[:,0,0]*sm[:,1,1]-sm[:,0,1]*sm[:,1,0]
  nd=context['dets']*ratio*sign;bad=~context['good']
  if np.any(bad):nd[bad]=np.linalg.det(e.rows[bad][:,np.asarray(new,dtype=np.int16),:])
  return e._collect(nd)
 for _ in range(20):
  x=random_d0(rng);phi,ctx=e.prepare(x)
  neighbors=list(heff_connections(x,h));rng.shuffle(neighbors)
  for y in neighbors[:20]:
   a=e.from_context(y,ctx);b=e(y);err=float(np.max(abs(a-b)));scale=float(max(np.max(abs(a)),np.max(abs(b)),1e-300))
   ae=max(ae,err);re=max(re,err/scale);checks+=1
  bold=np.zeros(e.kgen);bnew=np.zeros(e.kgen)
  for y,v in heff_connections(x,h).items():bold+=v*old_update(y,ctx);bnew+=v*e.from_context(y,ctx)
  old_b_rel=max(old_b_rel,float(np.linalg.norm(bold-bnew)/max(np.linalg.norm(bnew),1e-300)))
 rows.append({'spin':spin,'checks':checks,'max_absolute_error':ae,'max_relative_error':re,
              'pre_fix_local_action_relative_impact':old_b_rel})
status='PASS' if max(r['max_relative_error'] for r in rows)<2e-8 else 'FAIL'
rec={'status':status,'rows':rows,'job':os.environ['SLURM_JOB_ID']};dump(MC/'validation'/'update_6x6.json',rec)
print(json.dumps(rec,indent=2))
if status!='PASS':raise SystemExit(2)
