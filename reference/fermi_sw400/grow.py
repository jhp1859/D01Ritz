"""Grow exactly one bare Fermi determinant to unrestricted SW TrimCI N=400."""
from __future__ import annotations
import json, os, shutil, sys, tempfile, time
from pathlib import Path
import numpy as np
from common import *

RUNTIME = HOME/'trimci_original_nooo_6x6_n16_1m_20260823/runtime'
EXPECTED_CORE = '5c2e54e52ba3a6f216a68f11374e2405a6b453902a2e72d4f416a541d7960c60'

def main():
    assert os.environ.get('SLURM_JOB_ID')
    work = Path(tempfile.mkdtemp(prefix='h6_fermi_sw400_', dir='/tmp'))
    record = dict(status='RUNNING', job=os.environ['SLURM_JOB_ID'], scratch=str(work),
        target_determinants=NDET, basis='fixed standing-wave', initial_determinants=1,
        initial_state='exact bare Fermi determinant', hamiltonian='full Hubbard H',
        doublon_filter=False, symmetry_filter=False, orbital_optimization=False,
        random_multistart=False)
    publish(ROOT/'trial/growth.json', record)
    start = time.monotonic()
    try:
        z = np.load(BASE/'orbitals.npz')
        C = np.asarray(z['C'], float); occ = np.asarray(z['occ'], int)
        old = np.load(SOURCE/'state.npz')
        assert np.max(abs(C-old['C'])) < 1e-14
        ints = np.load(SOURCE/'integrals.npz'); h1=ints['h1']; eri=ints['eri']
        assert np.max(abs(C.T@np.asarray(z['K'],float)@C-h1)) < 3e-14
        ref = np.uint64(sum(1 << int(i) for i in occ))
        alpha = np.array([ref], dtype=np.uint64)
        beta = np.array([ref], dtype=np.uint64)
        coeff = np.array([1.0])
        sys.path.insert(0, str(RUNTIME))
        import trimci
        from trimci.TrimCI_runner.run_expansion import run_expansion
        assert sha(trimci.trimci_core.__file__) == EXPECTED_CORE
        phase = dict(max_n_dets=NDET, growth_factor=2.0, max_expansion_rounds=100,
            screening_mode='hb', threshold=1e-2, threshold_decay=.5,
            strict_target_size=True, orbital_optimization=False,
            pt2_correction=False, dressed_energy=False, expansion_energy_tol=-1,
            dets_conv_ratio=-1, use_connection_cache=True, use_sparse_update=False,
            symmetric_sigma=True, davidson_block_size=1, backend='cpu', verbose=2,
            davidson=dict(max_iter=500, max_subspace=60, n_states=1,
                residual_tol=1e-9, energy_tol=1e-11, verbose=0))
        publish(ROOT/'trial/phase.json', phase)
        result = run_expansion(h1=h1, eri=eri, e_nuc=0., alpha_init=alpha,
            beta_init=beta, coeffs_init=coeff, phases=[phase],
            checkpoint_dir=str(work/'checkpoints'), output_prefix=str(work/'native'),
            label='6x6 bare-Fermi-start fixed-SW unrestricted TrimCI N400')
        a=np.asarray(result['alphas'],np.uint64); b=np.asarray(result['betas'],np.uint64)
        c=np.asarray(result['coeffs'],float); e=float(result['energy_var'])
        assert result['n_final']==NDET and len(c)==NDET
        assert np.unique(np.column_stack([a,b]),axis=0).shape[0]==NDET
        assert np.all([int(x).bit_count()==16 for x in a])
        assert np.all([int(x).bit_count()==16 for x in b])
        assert abs(np.dot(c,c)-1)<1e-7
        ref_index=np.where((a==ref)&(b==ref))[0]
        assert len(ref_index)==1, ref_index
        np.savez(ROOT/'trial/state.npz', alpha_bits=a, beta_bits=b,
            coefficients=c, C=C, variational_energy=e, reference_bits=ref)
        shutil.copyfile(work/'native_results.json',ROOT/'trial/growth_history.json')
        record.update(status='PASS', actual_determinants=len(c), E_var=e,
            fermi_reference_index=int(ref_index[0]), fermi_reference_coefficient=float(c[ref_index[0]]),
            coefficient_norm=float(np.linalg.norm(c)), state_sha256=sha(ROOT/'trial/state.npz'),
            native_core_sha256=EXPECTED_CORE, elapsed_seconds=time.monotonic()-start)
        publish(ROOT/'trial/growth.json',record)
        print(json.dumps(record,indent=2),flush=True)
    except BaseException as exc:
        record.update(status='FAILED',error=repr(exc),elapsed_seconds=time.monotonic()-start)
        publish(ROOT/'trial/growth.json',record); raise

if __name__=='__main__': main()
