# 6×6 fixed-support D≤1 standing-wave Ritz handoff

This repository is a portable handoff for the 6×6 OBC Hubbard calculation at
`t=1`, `U=8`, `N_up=N_down=16`.  The requested fixed standing-wave determinant
budgets are `M=10, 100, 1,000, 10,000, 100,000`.  At each `M`, keep the
bundled determinant list and orbital matrix fixed; optimize only its linear
coefficients in the site `D≤1` effective Hamiltonian.

**Status:** the input ladder, checksums, signed local D≤1 SW2 operator and
small-system test are present.  The 6×6 D≤1 site sampler and the scalable
matrix-free generalized Ritz optimizer still require implementation and
validation.  This repository does not claim any completed 6×6 D01RITZ energy
or AFQMC result.  `REMOTE_TASK.md` is the work specification for the other
cluster.

## Frozen inputs

`inputs/state_M*.npz` contains `alpha_bits`, `beta_bits`, `coefficients`, and
the common `C` orbital matrix.  The `M=10` state is the **top-ten coefficient
subset of the archived M=100 support**; it is not an independently grown
TrimCI support.  Its `source_indices` record the exact selection.  The
M=100, 1k, 10k and 100k states are byte-for-byte copies of the archived
standing-wave TrimCI states.  The latter three came from sequential growth
starting at the archived M=100 state.  The `coefficients` are archived
full-H TrimCI coefficients and supply a starting vector only; they are not
D≤1 Ritz coefficients.

`inputs/integrals_sw.npz` contains the original transformed `h1`, `eri`, and
`C`.  Its 13.5 MB size is below GitHub's per-file limit, so Git LFS is not
needed for this handoff.  The site hopping convention is `i=6*x+y`, with
nearest-neighbor hopping `-1`; the input check verifies `C† K_site C = h1`.
No 6×6 FCI reference vector or energy is claimed here.

After cloning, run:

```bash
python3 scripts/verify_inputs.py
python3 -m unittest discover -s tests -v
```

The verified SHA256 values are in `inputs/MANIFEST.json`.  Never regenerate
or replace these support lists while running this comparison.  Keep new
results and checkpoints in `work/` or another separate output directory.

## Mathematical target

Let `P=P0+P1` project to site configurations with zero or one doublon, and
`P2` project to two doublons.  With `H=K+U Dhat`, the canonical Hermitian
second-order effective operator used here is

`H_eff^(01) = P H P - P1 K P2 K P1 / U`.

The convention is `exp(S) H exp(-S)` with `[S,H0]=-V_PQ` and `H0=U Dhat`.
`PHP` already includes the D=1 diagonal energy and direct D=0↔D=1 hopping.
Do not add the old D0 virtual D=1 correction.  `scripts/d01_operator.py`
enumerates all signed local paths without constructing the full 6×6 P basis.
`tests/test_d01_operator.py` compares it with an independently assembled
small exact matrix, including a case with no Q sector.

For fixed standing-wave generators `|D_I>`, solve the lowest generalized
Ritz problem for

`S_IJ=<D_I|P|D_J>`, `A_IJ=<D_I|P H_eff^(01) P|D_J>`, `A c = E S c`.

Sampling, as in the archived 6×6 P0 precedent, is over **site
configurations** for matrix evaluation.  It must not change the standing-wave
support or select new generators.  For a site configuration `x`, let
`phi_I(x)=<x|D_I>`, `b_I(x)=<x|H_eff^(01)|D_I>`, and
`w(x)=sum_I |phi_I(x)|²`.  With `q(x)=w(x)/Z`, independent samples estimate
`S/Z = E_q[phi† phi / w]` and `A/Z = E_q[phi† b / w]`; the common `Z` cancels
from the Ritz equation.  Sampling must move between D=0 and D=1.

At 100k, one dense 100,000×100,000 float64 matrix occupies about 80 GB;
forming/storing dense `S` and `A` is not the intended algorithm.  Use
matrix-free actions or a demonstrably equivalent compressed solver and
independent holdout samples.  Even matrix-free site-amplitude evaluation may
be expensive at 100k: benchmark it, record resource use, and preserve an
unresolved status if convergence gates fail.

## Included precedents and limits

- `reference/p0_sampler/` is a copy of the archived 6×6 **D=0-only** sampled
  matrix experiment.  It is a method precedent, not runnable D≤1 production
  code: it contains old absolute paths, D0-only proposals and D0 SW2 paths,
  and its `owners` array uses `int16`, which cannot index 100k generators.
  `RESULTS_INITIAL.json` reports an unresolved stochastic Ritz estimate.
- `reference/d01_4x4/verify_sw2.py` is the archived small-system SW2
  derivation check.  Its output path is archival and cluster-specific.
- `reference/6x6_SW_CONTINUATION_PROTOCOL.md` records the provenance of the
  standing-wave TrimCI support ladder.  Its AFQMC settings do not define this
  D≤1 Ritz task.

The original Ceph outputs and currently running 4×4 jobs are not touched by
this handoff.  This repository is for the 6×6 Ritz calculation only; AFQMC
trials and back-propagated observables require separate specifications.
