#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from d01_sampling import Amplitudes,lattice,initial,dump
from d01_operator import heff01_connections,doublons,kinetic_hops
root=Path(__file__).resolve().parents[1];rng=np.random.default_rng(604151)
records=[]
for m in [10,100]:
 z=np.load(root/f'supports/state_M{m}.npz');e=Amplitudes(z['C'],z['alpha_bits'],z['beta_bits']);h=lattice(6)
 state=initial(rng);d1=next(q for q,v in kinetic_hops(state,h) if doublons(q)==1)
 for state in [state,d1]:
  phi,ctx=e.prepare(state);max_abs=0.;max_scaled=0.;count=0
  for target in heff01_connections(state,h):
   a=e.updated(target,ctx);b=e.direct(target);err=float(np.max(np.abs(a-b)))
   scale=max(float(np.max(np.abs(b))),1e-22);max_abs=max(max_abs,err);max_scaled=max(max_scaled,err/scale);count+=1
   np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-22)
  records.append({'M':m,'D':doublons(state),'neighbors':count,'max_absolute_error':max_abs,'max_scaled_error':max_scaled})
dump(root/'reports/target_amplitude_validation.json',{'status':'PASS','seed':604151,'records':records})
print(records)
