# 6×6 bare-Fermi-origin D≤1 standing-wave Ritz handoff

**Correction (2026-10-06):** commit `e961531` bundled the wrong 6×6 support: it came from a matched random initial determinant bank, not a bare Fermi state. The incorrect M10–100k files have been removed from current `main`. Do not use that commit or its support files for this experiment.

System: 6×6 OBC Hubbard, t=1, U=8, Nup=Ndown=16. Follow the same two-stage construction as the 4×4 experiment: start with exactly one bare Fermi determinant and grow standing-wave TrimCI supports using the full Hubbard Hamiltonian; freeze each support; then reoptimize only its coefficients in the D≤1 effective Hamiltonian. Site-configuration sampling evaluates projected Ritz quantities. It does not select standing-wave determinants and is not AFQMC sampling.

## Correct inputs and what is missing

- `inputs/state_M1_fermi.npz` is the exact single bare Fermi determinant with coefficient one. `inputs/fermi_orbitals.npz` contains its original standing-wave orbitals and occupied indices.
- `inputs/integrals_sw.npz` contains the full-H transformed one- and two-body integrals. `scripts/verify_inputs.py` checks the kinetic transform against the archived Fermi orbitals.
- `reference/fermi_sw400/` is an independently verified example of growing 400 unrestricted standing-wave determinants from the single bare Fermi determinant. It includes the state, growth source, phase, history and status. M400 is a precedent, not a requested target support.
- The requested Fermi-origin M10, M100, M1000, M10000 and M100000 support files **do not yet exist here**. They must be grown and frozen before each projected Ritz calculation. Never use the old random-bank SW100 continuation or relabel M400 subsets as independently grown supports.

Run after cloning:

```bash
python3 scripts/verify_inputs.py
python3 -m unittest discover -s tests -v
```

Exact checksums are in `inputs/MANIFEST.json`. Preserve generated supports, coefficients, sampling chains and checkpoints with separate hashes and provenance.

## D≤1 Ritz stage

Use P=P0+P1 in the site basis and the Hermitian second-order operator `H_eff^(01)=PHP-P1 K P2 K P1/U`, with H=K+U Dhat. PHP already includes D=1 energy and D0↔D1 direct hopping. Do not add the old D0-only virtual D1 correction. `scripts/d01_operator.py` enumerates the signed local paths; `tests/test_d01_operator.py` checks exact small matrices.

For each fixed standing-wave support |D_I>, solve the lowest `A c=E S c`, with `S_IJ=<D_I|P|D_J>` and `A_IJ=<D_I|P H_eff^(01) P|D_J>`. Sample site configurations x with `q(x)∝w(x)=sum_I |<x|D_I>|²`, including D0↔D1 proposals. Use independent holdout samples and report statistical uncertainty. At M100k, avoid dense M×M matrices; one float64 100k×100k matrix alone is about 80 GB.

`reference/p0_sampler/` is a D0-only method precedent with old absolute paths and an int16 owner index invalid at M100k. It is not the D≤1 solver. The 6×6 D≤1 sampler and scalable generalized Ritz optimizer still require implementation and validation. No 6×6 D01RITZ or AFQMC result is claimed here.
