# Corrected Fermi-origin D01 sampled Ritz method

Input commit: 41bd5b5. The entire old matched-random experiment is invalid and
isolated in a separate checkout; no state, matrix, coefficient or checkpoint
from it is an input here. Source-only numerical routines were revalidated.

The full-H heat-bath TrimCI ladder starts from one supplied bare-Fermi
state. M100 grows separately from M1; M10 is its top-ten magnitude subset.
Continuation uses M100 -> M1000 -> M10000 -> M100000. Orbitals, support files
and list checksums are fixed before each D01 calculation. SW400 is only a
method check; native build/stopping differences are in SW400_VALIDATION_POLICY.md.
No new symmetry projection, orbital optimization or random initial bank is used.

Each site chain samples q=sum_I phi_I^2/Z on D<=1. A symmetric proposal
mixture selects a uniform ordered site pair and either a uniform spin's
particle-hole swap or a two-spin exchange. Invalid proposals are self
transitions. Metropolis acceptance is min(1,w_new/w_old); D0<->D1 proposals
are included. Unique spin determinants are shared only as an amplitude cache,
with int64 owner indices. Accepted states rebuild numerical contexts;
ill-conditioned and nodal updates use direct determinants.

The supplied signed local operator is PHP-P1KP2KP1/U. Independent 72-mode
creation/annihilation tests check all local paths on representative 6x6 D0
and D1 states; amplitude updates are checked against direct determinants.
No old D0 SW2 correction is added.

Banks store F=phi/sqrt(w), G=Hphi/sqrt(w). Matrix actions are
S v=F^T(Fv)/N and A v=[F^T(Gv)+G^T(Fv)]/(2N). Raw Hermiticity error is
reported before symmetrization. Independent holdout uses the unbiased raw
action F^T(Gv)/N. Coefficients have empirical training c^T(S/Z)c=1;
Euclidean-normalized coefficients are also saved. An independent uniform
P0/P1 estimator supplies an approximate absolute c^TSc, a physical-norm
coefficient file and explicit normalization error. Unknown Z is not hidden.

For N<M, FF^T/N=U lambda U^T yields X=F^T U/(sqrt(N) lambda), so X^T S X=I.
Diagonalizing X^T A X is exactly canonical Ritz on the empirical overlap
range. It never constructs M x M arrays. Complexity is O(NM) storage and
O(N^2 M) compression: statistical rank deficiency at small N remains a
failure, not a solved 100k problem. A dense-oracle test verifies equivalence.

Training and holdout seeds differ. Block jackknife estimates energy ratio
errors; block bootstrap estimates generalized-residual uncertainty,
conditional on fitted coefficients. Residuals use the holdout Rayleigh
energy; final analyses also report residuals at the fixed training energy.
Block sizes 16/32/64/128, independent seed groups, sample halves and overlap
cutoffs 1e-8/1e-6/1e-4 assess stability. Conservative block-size errors enter
the precision gates. These finite-chain error estimates cannot rule out
missed modes. Low training energy or tiny fitted residual alone never passes.

Every 16 samples, flushed banks precede an atomic checkpoint containing RNG,
state, counts, options and support/seed checksums. Identical commands resume
at the committed count; uncommitted tails are overwritten. Tests interrupt
and restart actual chains and compare their banks with uninterrupted runs.
Large files remain on Ceph under work/, excluded from Git. No AFQMC is run.

The implementation explicitly refuses a large-M bank for which dual Gram
compression would exceed 4096 rows or reach N>=M. It does not fall back to
a dense M x M matrix at 100k. Such banks require a further iterative solver;
this is a documented scalability limit, and the diagnostic low-rank result
must not be called a statistically converged 100k solution.

Analysis verifies every bank file SHA256, the corrected seed SHA256, the
frozen support SHA256, the train/holdout role and uniqueness of all chain
seeds before fitting. An old matched-random bank cannot pass these checks.
Final analyses also use a leave-one-chain-out energy jackknife and report
selected-coefficient autocorrelation by holdout chain. The reported
conservative energy error is the maximum of block and chain-cluster errors.

For the enlarged 10k banks, an additional matrix-free block LOBPCG solver
uses only F/G actions and diagonal preconditioning. It studies explicit
ridge strengths 1e-8/1e-6/1e-4 times a power estimate of lambda_max(S);
this is labeled ridge sensitivity, NOT hard overlap truncation. Energies,
norms and residuals are evaluated in the original unregularized A,S pencil.
Iteration warnings are retained; a converged regularized solve alone cannot
pass the original-pencil residual gate. A dense generalized-eigenproblem
oracle independently verifies the iterative algorithm at zero ridge.
This supersedes the earlier refusal-only guard for large sample banks;
there is still no dense M x M allocation for large M.
Within-chain length stability compares the first and second halves of every
holdout chain at the fitted coefficients, with a leave-one-chain-out error
for their energy difference. A drift larger than max(0.2,3 SE) fails. Only
when this and every predeclared gate pass is a final result labeled
converged_at_declared_sampling_precision; this is not an exact-matrix claim.

Residual norm uncertainty is additionally audited by bootstrapping the
centered residual VECTOR. A 95% norm ball for its variation gives a
triangle-inequality interval [max(0,r-radius),r+radius]. This avoids interpreting
upward-biased raw norm-bootstrap percentiles as an unbiased confidence
interval near zero. The more conservative upper bound enters the gate.
Audits do not refit or replace any coefficient file.
