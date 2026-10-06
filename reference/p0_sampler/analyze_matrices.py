#!/usr/bin/env python3
"""Freeze MC Ritz vectors and evaluate them on independent holdout chains."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from mc_common import *
from mc_common import _act


def support_hash(alpha, beta):
    pair = np.column_stack([np.asarray(alpha, dtype='<u8'), np.asarray(beta, dtype='<u8')])
    return hashlib.sha256(np.ascontiguousarray(pair).view(np.uint8)).hexdigest()


def load_stage(spin, stage):
    records=[]; sall=[];aall=[]
    for path in sorted((MC/'matrices'/spin).glob(f'{stage}_rep*.json')):
        meta=json.loads(path.read_text())
        if int(meta['replica']) >= 900: continue
        if meta['status']!='PASS' or meta['chains']<3: raise RuntimeError(('bad stage',path))
        data=np.load(path.with_suffix('.npz'),allow_pickle=False)
        S,A=np.asarray(data['S_blocks']),np.asarray(data['A_blocks'])
        per=len(S)//meta['chains']
        if per*meta['chains']!=len(S): raise RuntimeError('bad chain partition')
        records.append((meta,per));sall.append(S);aall.append(A)
    if not records: raise RuntimeError(('no production replicas',spin,stage))
    if stage == 'holdout' and len({p for _,p in records})!=1:
        raise RuntimeError('holdout replicas use different blocks/chain')
    meta={'status':'PASS','stage':stage,'replicas':[m['replica'] for m,_ in records],
          'chains':sum(m['chains'] for m,_ in records),
          'total_samples':sum(m['total_samples'] for m,_ in records),
          'wall_seconds':sum(m['wall_seconds'] for m,_ in records),
          'peak_RSS_GiB':max(m['peak_RSS_GiB'] for m,_ in records),
          'rows':[r for m,_ in records for r in m['rows']]}
    return np.concatenate(sall),np.concatenate(aall),meta,records[0][1]


def chain_jackknife_energy(Sb, Ab, c, nchains, per_chain):
    idx = np.arange(len(Sb)).reshape(nchains, per_chain)
    estimates = []
    for leave in range(nchains):
        keep = np.delete(idx, leave, axis=0).ravel()
        sm, am = Sb[keep].mean(0), Ab[keep].mean(0)
        estimates.append(float(c @ ((am+am.T)*0.5) @ c / (c @ ((sm+sm.T)*0.5) @ c)))
    estimates = np.asarray(estimates)
    full = float(c @ ((Ab.mean(0)+Ab.mean(0).T)*0.5) @ c /
                 (c @ ((Sb.mean(0)+Sb.mean(0).T)*0.5) @ c))
    mean_j = float(estimates.mean())
    se = float(np.sqrt((nchains-1)/nchains*np.sum((estimates-mean_j)**2)))
    return full, se, estimates


def spin_s2_free(alpha, beta, coeff):
    raised = {}
    for a, b, value in zip(alpha, beta, coeff):
        full = int(a) | (int(b) << N)
        for p in range(N):
            q1 = _act(full, N+p, False)
            q2 = None if q1 is None else _act(q1[0], p, True)
            if q2 is not None:
                raised[q2[0]] = raised.get(q2[0], 0.0) + float(value)*q1[1]*q2[1]
    return float(sum(x*x for x in raised.values()) / np.dot(coeff, coeff))


def training_replica_sensitivity(spin, pooled_c, pooled_s):
    records=[]
    pc=np.asarray(pooled_c,float);pc/=np.sqrt(pc@pooled_s@pc)
    for path in sorted((MC/'matrices'/spin).glob('train_rep*.json')):
        meta=json.loads(path.read_text())
        if int(meta['replica'])>=900: continue
        z=np.load(path.with_suffix('.npz'),allow_pickle=False)
        sr,ar=np.asarray(z['S_blocks']).mean(0),np.asarray(z['A_blocks']).mean(0)
        cr,sol=canonical_lowest(ar,sr,1e-6);cr=np.asarray(cr,float)
        cr/=np.sqrt(cr@pooled_s@cr)
        overlap=float(cr@pooled_s@pc)
        if overlap<0:cr*=-1;overlap=-overlap
        records.append({'replica':int(meta['replica']),'samples':meta['total_samples'],
                        'Ritz_energy':sol['energy'],'pooled_metric_overlap':overlap,
                        'pooled_metric_distance':float(np.sqrt(max(0.,2.-2.*overlap)))})
    return records


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('analysis must run on a compute node')
    if not (COMMON/'MANIFEST.json').is_file():
        raise RuntimeError('common inputs not frozen')
    t0 = time.monotonic()
    label = os.environ.get('MC_ANALYSIS_LABEL', 'refined').strip().lower()
    if not label.replace('_','').isalnum():
        raise RuntimeError('invalid MC_ANALYSIS_LABEL')
    gen = np.load(COMMON/'generator'/'state.npz', allow_pickle=False)
    alpha, beta = np.asarray(gen['alpha_bits'], np.uint64), np.asarray(gen['beta_bits'], np.uint64)
    C = np.asarray(gen['C'])
    shash = support_hash(alpha, beta)
    rows, coefficients = [], {}
    common_spaces = {s: np.load(COMMON/'spaces'/f'{s}.npz', allow_pickle=False)
                     for s in ('free', 'singlet')}

    for spin in ('free', 'singlet'):
        Str, Atr, mt, ptr = load_stage(spin, 'train')
        Sho, Aho, mh, pho = load_stage(spin, 'holdout')
        Sm, Am = Str.mean(0), Atr.mean(0)
        raw_s_anti = float(np.max(abs(Sm-Sm.T)))
        raw_a_anti = float(np.max(abs(Am-Am.T)))
        solves = {}
        c_primary = None
        for cut in (1e-7, 1e-6, 1e-5):
            c, rec = canonical_lowest(Am, Sm, cut)
            solves[f'{cut:.0e}'] = rec
            if cut == 1e-6:
                c_primary = np.asarray(c, float)
        c = c_primary
        Ehold, Esem, jk = chain_jackknife_energy(Sho, Aho, c, mh['chains'], pho)
        Sh, Ah = (Sho.mean(0)+Sho.mean(0).T)*0.5, (Aho.mean(0)+Aho.mean(0).T)*0.5
        residual = float(np.linalg.norm(Ah@c-Ehold*(Sh@c)) /
                         max(np.linalg.norm(Ah@c), 1e-300))

        space = common_spaces[spin]
        G, H = np.asarray(space['G']), np.asarray(space['H_full'])
        full_norm = float(c @ G @ c)
        evar = float(c @ H @ c / full_norm)
        euclid = c/np.linalg.norm(c)
        if euclid[np.argmax(abs(euclid))] < 0: euclid *= -1
        coefficients[spin] = euclid
        transform = np.asarray(space['generator_transform'])
        ndet_export = int(np.count_nonzero(abs(transform@euclid) > 2e-14))
        if spin == 'singlet':
            s2, ps = 0.0, 1.0
        else:
            s2 = spin_s2_free(alpha, beta, euclid)
            Gs = np.asarray(common_spaces['singlet']['G'])
            ps = float(euclid @ Gs @ euclid / (euclid @ G @ euclid))
        metric_values = np.linalg.eigvalsh((Sm+Sm.T)*0.5)
        replica_sensitivity=training_replica_sensitivity(spin,c,(Sm+Sm.T)*0.5)
        rows.append({
            'method': 'MC_SW2', 'spin': spin, 'K_gen': 100,
            'N_det_export': ndet_export, 'rank_S': solves['1e-06']['rank'],
            'E_var_full': evar, 'E_eff_train_noisy': solves['1e-06']['energy'],
            'E_eff_holdout': Ehold, 'E_eff_holdout_SE': Esem,
            'holdout_target_met': bool(Esem <= 1e-3),
            'holdout_generalized_residual_relative': residual,
            'S2': s2, 'p_S0': ps, 'full_norm_before_export_rescale': full_norm,
            'S_min_eigenvalue': float(metric_values[0]),
            'S_max_eigenvalue': float(metric_values[-1]),
            'S_raw_antihermiticity_max': raw_s_anti,
            'A_raw_antihermiticity_max': raw_a_anti,
            'regularization_sensitivity': solves,
            'training_replica_coefficient_sensitivity': replica_sensitivity,
            'holdout_chain_jackknife_values': jk.tolist(),
            'train_total_samples': mt['total_samples'],
            'holdout_total_samples': mh['total_samples'],
            'train_wall_seconds': mt['wall_seconds'],
            'holdout_wall_seconds': mh['wall_seconds'],
            'peak_RSS_GiB': max(mt['peak_RSS_GiB'], mh['peak_RSS_GiB']),
            'mean_acceptance_train': {k: float(np.mean([r['acceptance'][k] for r in mt['rows']]))
                                      for k in ('particle_hole','spin_exchange')},
            'max_tau_train': float(max(max(r['tau_num'],r['tau_den']) for r in mt['rows'])),
            'max_update_error': float(max(r['determinant_update_max_abs_error']
                                          for r in mt['rows']+mh['rows'])),
        })
        out = MC/f'results_{label}'/spin
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out/'matrices_and_coefficients.npz',
                            S_train_over_Z=Sm, A_train_over_Z=Am,
                            S_holdout_over_Z=Sho.mean(0), A_holdout_over_Z=Aho.mean(0),
                            S_trace_normalized=Sm/np.trace(Sm),
                            A_trace_normalized=Am/np.trace(Sm),
                            generator_coefficients=euclid,
                            holdout_chain_jackknife_energy=jk)
        dump(out/'summary.json', rows[-1])

    bfree = np.asarray(common_spaces['free']['baseline_generator_coefficients'], float)
    bs0 = np.asarray(common_spaces['singlet']['baseline_generator_coefficients'], float)
    bfree /= np.linalg.norm(bfree); bs0 /= np.linalg.norm(bs0)
    for v in (bfree, bs0):
        if v[np.argmax(abs(v))] < 0: v *= -1
    frozen = MC/f'TRIAL_COEFFICIENTS_{label.upper()}.npz'
    frozen.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(frozen, alpha_bits=alpha, beta_bits=beta, C=C,
                        support_sha256=np.asarray(shash),
                        mc_free_coefficients=coefficients['free'],
                        mc_singlet_coefficients=coefficients['singlet'],
                        baseline_free_coefficients=bfree,
                        baseline_singlet_coefficients=bs0)
    record = {'status': 'PASS' if all(r['holdout_target_met'] for r in rows) else 'UNRESOLVED',
              'run_key': RUN_KEY, 'support_sha256': shash,
              'trial_file': str(frozen), 'trial_file_sha256': sha(frozen),
              'rows': rows, 'wall_seconds': time.monotonic()-t0,
              'peak_RSS_GiB': rss_gib(), 'job': os.environ['SLURM_JOB_ID']}
    dump(MC/f'RESULTS_{label.upper()}.json', record)
    table(MC/f'RESULTS_{label.upper()}.csv', rows)
    print(json.dumps(record, indent=2), flush=True)
    if record['status'] != 'PASS':
        raise SystemExit(3)


if __name__ == '__main__':
    main()
