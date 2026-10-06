# Task for a second cluster: 6×6 D01RITZ coefficient optimization

1. Clone this repository, run `scripts/verify_inputs.py` and the unit tests,
   and record the Git commit and input checksums in every result.  Use a new
   output directory.  Do not modify the frozen input states.
2. Use the exact bundled standing-wave support for each M=10,100,1000,
   10000,100000.  M means generator count **before** site projection.  M10
   is the explicitly labeled top-ten subset of M100.  Keep fixed Nup=Ndown=16
   and do not add a singlet or other symmetry projection.
3. Implement the D≤1 site sampler using `q(x)∝sum_I |phi_I(x)|²`, with both
   within-sector and D0↔D1 moves.  Use `scripts/d01_operator.py` for the
   Hermitian SW2 local action `PHP-P1 K P2 K P1/U`, including all signed
   two-hop paths.  Do not silently substitute PHP-only or old D0 SW2.
4. Validate the site amplitude updates and all local SW2 paths against
   independent direct evaluations on small lattices and representative
   6×6 configurations, including D0↔D1 and D1→D2→D1 paths.  Compare sampled
   matrix estimates at M10/100 with independently generated estimates;
   record Hermiticity error and autocorrelation.  Keep independent training
   and holdout chains with fixed, recorded seeds.
5. Benchmark M10/100 before scaling.  For large M, avoid dense M×M matrices
   and M×M outer products.  A matrix-free action must define its stochastic
   sample bank and normalization consistently.  Measure peak memory,
   CPU-hours and wall time.  Do not describe 100k as feasible until measured.
6. For each M, report the lowest Ritz energy, `c` and its SHA256, `c†Sc`,
   relative generalized residual, sample error and independent holdout
   objective energy/error.  Study stability versus sample count, chain
   length, overlap-spectrum cutoff and random seed.  A low training energy
   alone is not a pass.  Flag unresolved/failed cases without replacing the
   fixed support or relaxing a gate silently.
7. Keep trial coefficients and diagnostic files in the result directory,
   then report the exact command, software versions, scheduler resources,
   input hashes and failure modes back to the main project.  Do not run AFQMC
   as part of this task unless separately requested.

The 4×4 precedent used relative generalized residual ≤1e-7 for its exact
Ritz solve.  For a sampled 6×6 solve, report residual uncertainty on an
independent holdout set and establish a precision gate *before* declaring
the result converged.  The archived 6×6 P0 sampler was unresolved at 100
generators, so more sampling or an improved numerical method may be needed.
