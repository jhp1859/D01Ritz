#!/usr/bin/env python3
"""Collect a provenance-checked summary; incomplete levels remain explicit."""
from pathlib import Path
import csv
import json
import subprocess
from d01_sampling import sha,dump
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'supports/MANIFEST.json').read_text())
rows=[];artifacts=[]
for m in [10,100,1000,10000,100000]:
    candidates=[ROOT/f'reports/{round_name}/M{m}' for round_name in ['round4','round3','round2','round1']]
    folder=next((p for p in candidates if (p/'result.json').exists()),None)
    state=manifest['states'][str(m)]
    row={'M':m,'support_sha256':state['sha256'],'support_full_H_energy':state.get('full_H_energy'),'status':'pending'}
    if folder:
        r=json.loads((folder/'result.json').read_text());q=r['chosen'];h=q['holdout']
        norm=json.loads((folder/'normalization.json').read_text()) if (folder/'normalization.json').exists() else {}
        audit=json.loads((folder/'residual_uncertainty_audit.json').read_text()) if (folder/'residual_uncertainty_audit.json').exists() else {}
        final_status=r['status'];final_failures=list(r['failures'])
        if audit and not audit['passes_residual_gate']:
            final_status='unresolved';final_failures.append('residual_vector_uncertainty_gate')
        row.update(round=folder.parent.name,status=final_status,training_energy=q['energy'],holdout_energy=h['energy'],
            holdout_energy_SE=h.get('conservative_energy_SE',h['energy_SE']),
            holdout_relative_residual=h['relative_residual'],holdout_residual_SD=h['residual_bootstrap_SD'],
            holdout_residual_95pct_upper=max(h.get('conservative_residual_95pct_upper',h['residual_bootstrap_95pct'][1]),audit.get('conservative_residual_upper_95pct',0)),
            residual_norm_interval_lower=min(v['residual_norm_confidence_ball_95pct'][0] for v in audit['blocks']) if audit else None,
            residual_at_training_energy=h.get('residual_at_training_energy'),
            overlap_rank=q['retained_rank'],overlap_retained_condition=q['overlap_retained_condition'],
            overlap_full_condition=q['overlap_full_condition'],regularization=q.get('overlap_regularization','canonical hard cutoff'),
            norm_training_S_over_Z=q['norm_S_over_Z'],physical_norm_relative_SE=norm.get('relative_norm_SE'),
            train_samples=r['train_samples'],holdout_samples=r['holdout_samples'],
            sampling_CPU_hours=r['sampling_CPU_hours'],sampling_wall_seconds_sum=r['sampling_wall_seconds_sum'],
            sampling_max_chain_wall_seconds=max(c['wall_seconds'] for c in r['chains']),
            peak_rss_MiB=max(r['sampling_peak_rss_MiB'],r['analysis_peak_rss_MiB'],norm.get('peak_rss_MiB',0),audit.get('peak_rss_MiB',0)),
            analysis_CPU_hours=r['analysis_CPU_seconds']/3600,analysis_wall_seconds=r['analysis_seconds'],
            norm_CPU_hours=norm.get('CPU_seconds',0)/3600,norm_wall_seconds=norm.get('wall_seconds'),
            residual_audit_CPU_hours=audit.get('CPU_seconds',0)/3600,
            coefficient_sha256=r['coefficients']['sha256'],physical_coefficient_sha256=norm.get('coefficients_sha256'),
            failures='; '.join(sorted(set(final_failures))),result_path=str(folder/'result.json'))
        for p in folder.iterdir():
            if p.is_file():artifacts.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
    chain_cost={}
    for manifest_file in (ROOT/'work').glob(f'round*/M{m}/*/manifest.json'):
        cm=json.loads(manifest_file.read_text());chain_cost[str(manifest_file.resolve())]=cm['cpu_seconds']
    row['all_corrected_sampling_CPU_hours']=sum(chain_cost.values())/3600
    rows.append(row)
allkeys=list(dict.fromkeys(k for r in rows for k in r))
with (ROOT/'reports/SUMMARY.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,allkeys);w.writeheader();w.writerows(rows)
# Bank files, checkpoints and scheduler logs stay on Ceph. Hash every artifact.
cluster_files=[]
for name in ['round1','round2','round3','round4','growth','sw400_reproduction','sw400_attempt_939909','node_smoke','validation','scheduler','runtime']:
    base=ROOT/'work'/name
    for p in sorted(base.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:
            cluster_files.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
dump(ROOT/'reports/RESULT_MANIFEST.json',{'input_commit':'41bd5b5','origin':'single_bare_Fermi',
    'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    'support_manifest_sha256':sha(ROOT/'supports/MANIFEST.json'),'input_manifest_sha256':sha(ROOT/'inputs/MANIFEST.json'),
    'invalid_old_run':'/mnt/ceph/home/park687/D01Ritz/invalid_input_record_20261006/manifest.json',
    'old_jobs_stopped':[939905,939906,939907],'AFQMC_run':False,
    'rows':rows,'small_result_files':artifacts,'cluster_files':cluster_files})
lines=['# Corrected bare-Fermi-origin D01 Ritz results','',
'Input commit `41bd5b5`; single M1 -> full-H M100 -> 1k -> 10k -> 100k. M10 is the top-ten subset of M100. The e961531 matched-random experiment is invalid and isolated. No old matrices, coefficients or checkpoints are reused.','',
'All support files are frozen in `supports/`; checksums and parentage are in its MANIFEST.json. SW400 is only a method check; the different native build and stricter production stopping criterion are documented in SW400_VALIDATION_POLICY.md.','',
'M100 round3 is a fresh independent holdout confirmation of the byte-identical round2 coefficient file. Round2 narrowly failed train/holdout agreement; that failure is preserved and its threshold was not relaxed.','',
'## Ritz estimates','',
'Unresolved rows are diagnostic coefficient fits, **not converged lowest Ritz energies**. Errors are conditional finite-chain estimates; see full block, seed, cutoff/ridge and sample-count diagnostics.','',
'| M | Training E | Independent holdout E ± SE | Holdout relative residual ± bootstrap SD | Status |',
'|---:|---:|---:|---:|---|']
for r in rows:
    if 'holdout_energy' not in r:lines.append(f"| {r['M']} | — | — | — | pending |");continue
    lines.append(f"| {r['M']:,} | {r['training_energy']:.6f} | {r['holdout_energy']:.6f} ± {r['holdout_energy_SE']:.6f} | {r['holdout_relative_residual']:.5f} ± {r['holdout_residual_SD']:.5f} | {r['status']} |")
lines+=['','M1000 remains limited by energy precision and sample/seed stability despite full empirical overlap rank. M10000 also fails the unregularized training residual and holdout residual gates; its ridge LOBPCG output is a failed diagnostic fit. M100000 has only 128 training rows and retained rank 120, so its diagnostic sample bank cannot identify a 100000-dimensional Ritz solution. More sampling and solver work are required; these levels are not declared solved.','',
'## Resources and diagnostics','',
'CPU hours below are sampling process CPU; full analysis and normalization CPU, active wall times, per-chain seeds, all support/coefficient hashes and failures are in SUMMARY.csv and RESULT_MANIFEST.json. Peak RSS is per process, not the sum of concurrent jobs. Queue waiting time is excluded.','',
'| M | Train / holdout samples | Sampling CPU h | Peak RSS MiB | Overlap rank | Failure gates |',
'|---:|---:|---:|---:|---:|---|']
for r in rows:
    if 'holdout_energy' in r:lines.append(f"| {r['M']:,} | {r['train_samples']:,} / {r['holdout_samples']:,} | {r['sampling_CPU_hours']:.3f} | {r['peak_rss_MiB']:.1f} | {r['overlap_rank']} | {r['failures']} |")
lines+=['','Coefficients are normalized to empirical training S/Z, with a separate physical-norm estimate and its uncertainty. Unknown guide Z is not treated as one. Large sample banks and exact RNG checkpoints remain on Ceph at the manifest paths; re-running a listed submission command resumes committed samples. No AFQMC was run.','']
(ROOT/'reports/RESULTS.md').write_text('\n'.join(lines))
print(json.dumps(rows,indent=2))
