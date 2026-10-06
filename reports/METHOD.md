# D01 fixed-support sampled Ritz implementation

Input commit: `e961531e234df80b4495dc54035cf70dfe2416ba`.
The support and C files are unchanged. No singlet projection is applied.

Each chain samples q(x)=sum_I phi_I(x)^2/Z on D<=1. Proposals choose a
uniform ordered pair of distinct sites, then (with equal probability) a
uniform spin's particle-hole swap or a two-spin exchange. Invalid moves
are self transitions. Each proposal and its reverse have the same
probability; the Metropolis ratio is w(new)/w(old). Particle counts stay
fixed and particle-hole proposals allow D0<->D1. All accepted states rebuild
inverse contexts; near-singular spin matrices and near-zero update ratios
use direct determinants. Spin determinant sharing is only a cache for the
original generator list. Owner indices are int64.

The supplied signed D01 operator enumerates PHP-P1KP2KP1/U. Independent
72-spin-orbital creation/annihilation tests compare every local matrix
element for three D0 and three D1 configurations on 6x6. Separate tests
compare all stencil amplitudes with direct determinants. No D0 SW2 term is
added.

Each retained sample stores F=phi/sqrt(w), G=Hphi/sqrt(w), and the site bits.
A sample-bank action is S v=F^T(Fv)/N and
A v=[F^T(Gv)+G^T(Fv)]/(2N). This finite-bank Hermitian estimate differs from
the raw unsymmetrized estimator by sampling noise; raw Hermiticity error is
reported. Independent holdout actions use the unbiased unsymmetrized form.

For N<M, diagonalize the N x N Gram matrix FF^T/N=U lambda U^T. With
X=F^T U/(sqrt(N) lambda), X^T S X=I. Diagonalizing X^T A X is exactly the
canonical generalized Ritz solution on the retained empirical S range.
There are no M x M arrays or outer products at large M. The implementation
has O(NM) storage and O(N^2 M) compression cost; it is not a solution to the
statistical rank bottleneck when N is much smaller than M. Nullity and full
coefficient-space residual are reported explicitly. A dense-oracle unit
test checks this dual algorithm.

Training and holdout seeds are disjoint. Error bars use block jackknife
ratios and block bootstrap of the independent holdout generalized residual,
conditional on the fitted coefficients. These are finite-chain estimates,
not guarantees against missed modes. Block sizes 16/32/64/128, seed splits,
training sample halves, and overlap cutoffs 1e-8/1e-6/1e-4 are reported.
The training fixed-vector error bar is optimistic after optimization and is
not a substitute for holdout error. Initial precision gates are in
PRECISION_GATES.json. Insufficient sector crossings or effective sample
counts fail the gates. Stability of a low training energy alone never passes.

Coefficients are stored normalized to the empirical training S/Z, and also
with unit Euclidean norm. Because guide Z is not known, neither is labeled
as exact physical normalization. The optional independent uniform P0/P1
estimator computes absolute c^TSc and a separately normalized coefficient
file with its normalization uncertainty. Sector counts are C(36,16)C(20,16)
and C(36,16)*16*C(20,15). It does not use unstable harmonic-mean normalization.

A chain checkpoint records the complete RNG and site state, committed sample
count, options and input SHA256. Every 16 samples its bank files are flushed
before the checkpoint is atomically replaced. Re-running the identical
command resumes at that committed count, overwriting any uncommitted tail.
Completed manifests hash the banks. Interrupted bank tails are not samples.
The Ceph `work/` tree is excluded from Git. No AFQMC is launched.
