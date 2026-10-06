#!/usr/bin/env python3
"""Full-H heat-bath TrimCI from a single bare Fermi determinant; no MC selection."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time
import numpy as np
from d01_sampling import sha,dump
from full_h_oracle import full_h_matrix

ROOT=Path(__file__).resolve().parents[1]
SEED_SHA='e9b19b8051dd8ebecfdd249c316b030705bbb646000938fcbe512bf9a3760b5b'


def validate(z,seed,m):
    a=z['alpha_bits'];b=z['beta_bits'];c=z['coefficients']
    assert len(a)==len(b)==len(c)==m
    assert len(set(zip(map(int,a),map(int,b))))==m
    assert all(bin(int(v)).count('1')==16 for arr in [a,b] for v in arr)
    assert np.array_equal(z['C'],seed['C'])
    assert abs(c@c-1)<1e-7
    assert (int(seed['alpha_bits'][0]),int(seed['beta_bits'][0])) in set(zip(map(int,a),map(int,b)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--targets',type=int,nargs='+',default=[100,1000,10000,100000]);p.add_argument('--validate-sw400',action='store_true');args=p.parse_args()
    t=time.monotonic();cpu=time.process_time()
    sys.path.insert(0,str(ROOT/'work/runtime'))
    import trimci
    from trimci.TrimCI_runner.run_expansion import run_expansion,_build_config
    seed_path=ROOT/'inputs/state_M1_fermi.npz';assert sha(seed_path)==SEED_SHA
    seed=dict(np.load(seed_path));ints=np.load(ROOT/'inputs/integrals_sw.npz');h=ints['h1'].real;eri=ints['eri'].real
    phase=json.loads((ROOT/'reference/fermi_sw400/phase.json').read_text())
    cfg=_build_config(phase,str(ROOT/'work/config_probe'))
    for key,value in phase.items():
        if key!='davidson':
            assert hasattr(cfg,key),('unsupported setting',key)
            assert getattr(cfg,key)==value,(key,getattr(cfg,key),value)
    core=Path(trimci.trimci_core.__file__)
    provenance={'origin':'single_bare_Fermi','seed_sha256':SEED_SHA,'input_commit':'41bd5b5',
        'integrals_sha256':sha(ROOT/'inputs/integrals_sw.npz'),'native_core_path':str(core),'native_core_sha256':sha(core),
        'run_expansion_sha256':sha(ROOT/'work/runtime/trimci/TrimCI_runner/run_expansion.py'),
        'full_H_growth':True,'random_multistart':False,'site_MC_selection':False,'orbital_optimization':False,'spin_projection':False}
    supportdir=ROOT/'supports';supportdir.mkdir(exist_ok=True)
    manifest_path=supportdir/'MANIFEST.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(provenance,states={})
    assert manifest['seed_sha256']==SEED_SHA
    def grow(start,m,folder):
        folder.mkdir(parents=True,exist_ok=True);ph=dict(phase,max_n_dets=m)
        dump(folder/'phase.json',ph)
        result=run_expansion(h1=h,eri=eri,e_nuc=0.,alpha_init=start['alpha_bits'],beta_init=start['beta_bits'],coeffs_init=start['coefficients'],
            phases=[ph],checkpoint_dir=str(folder/'checkpoints'),output_prefix=str(folder/'native'),label=f'bare-Fermi-origin fixed SW full-H M{m}')
        state=dict(alpha_bits=np.asarray(result['alphas'],np.uint64),beta_bits=np.asarray(result['betas'],np.uint64),coefficients=np.asarray(result['coeffs'],float),C=seed['C'],variational_energy=float(result['energy_var']))
        validate(state,seed,m)
        if m<=400:
            H=full_h_matrix(state['alpha_bits'],state['beta_bits'],h,eri);c=state['coefficients'];E=float(c@H@c)
            residual=float(np.linalg.norm(H@c-E*c))
            assert abs(E-result['energy_var'])<1e-7,(E,result['energy_var'])
            assert residual<1e-6,residual
            dump(folder/'independent_full_H_check.json',{'energy':E,'absolute_residual':residual,'dense_lowest_energy':float(np.linalg.eigvalsh(H)[0])})
        return state,result
    if args.validate_sw400:
        folder=ROOT/'work/sw400_reproduction'
        if (folder/'validation.json').exists():
            assert json.loads((folder/'validation.json').read_text())['status']=='PASS'
        else:
            state,result=grow(seed,400,folder)
            np.savez(folder/'state_M400.npz',**state)
            ref=np.load(ROOT/'reference/fermi_sw400/state_M400.npz');a=set(zip(map(int,state['alpha_bits']),map(int,state['beta_bits'])));b=set(zip(map(int,ref['alpha_bits']),map(int,ref['beta_bits'])))
            Eref=float(json.loads((ROOT/'reference/fermi_sw400/growth.json').read_text())['E_var']);err=abs(state['variational_energy']-Eref)
            rec=dict(provenance,status='PASS' if err<1e-7 else 'FAILED',reference_energy=Eref,energy=state['variational_energy'],energy_error=err,support_exact_match=a==b,common_support=len(a&b),wall_seconds=time.monotonic()-t)
            dump(folder/'validation.json',rec)
            if rec['status']!='PASS':raise RuntimeError('SW400 growth reproduction failed; target growth forbidden')
    gate=ROOT/'work/sw400_reproduction/validation.json'
    assert gate.exists() and json.loads(gate.read_text())['status']=='PASS'
    current=seed;parent=seed_path
    for m in [100,1000,10000,100000]:
        dest=supportdir/f'state_M{m}.npz'
        if dest.exists():
            assert str(m) in manifest['states'] and sha(dest)==manifest['states'][str(m)]['sha256']
            current=dict(np.load(dest));validate(current,seed,m);parent=dest
            continue
        if m not in args.targets:break
        stage_t=time.monotonic();folder=ROOT/f'work/growth/M{m}'
        state,result=grow(current,m,folder)
        with dest.open('xb') as f:np.savez(f,**state)
        pairs=np.column_stack([state['alpha_bits'],state['beta_bits']]).astype('<u8')
        import hashlib
        manifest['states'][str(m)]={'path':str(dest),'sha256':sha(dest),'determinant_list_sha256':hashlib.sha256(pairs.tobytes()).hexdigest(),
            'parent_path':str(parent),'parent_sha256':sha(parent),'initial_determinants':len(current['coefficients']),
            'full_H_energy':state['variational_energy'],'phase':phase,'stage_wall_seconds':time.monotonic()-stage_t}
        dump(manifest_path,manifest)
        if m==100:
            order=np.argsort(-abs(state['coefficients']),kind='stable')[:10]
            small={k:v[order] for k,v in state.items() if k in ['alpha_bits','beta_bits','coefficients']};small['coefficients']/=np.linalg.norm(small['coefficients']);small['C']=seed['C'];small['source_indices']=order
            dst=supportdir/'state_M10.npz'
            with dst.open('xb') as f:np.savez(f,**small)
            manifest['states']['10']={'path':str(dst),'sha256':sha(dst),'parent_sha256':sha(dest),'definition':'top-ten magnitude subset of Fermi-grown M100; not independent growth','source_indices':order.tolist()}
            dump(manifest_path,manifest)
        current=state;parent=dest
    dump(ROOT/'reports/growth_status.json',dict(provenance,status='PASS',available_supports=list(manifest['states']),wall_seconds=time.monotonic()-t,CPU_hours=(time.process_time()-cpu)/3600,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))

if __name__=='__main__':main()
