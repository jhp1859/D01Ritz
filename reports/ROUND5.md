# M10000 additional precision

The earlier M10000 fit had only 16,384 training and 8,192 holdout rows,
giving training energy -43.892 and independent holdout -20.685. Its
matrix-free LOBPCG also stopped at 80 iterations with original-pencil
residual 0.00227. That is a failed fit, not a converged Ritz value.

Round5 keeps the corrected fixed M10000 support. The old corrected
16 training chains x1024 are retained as symlinks. New independent chains
add 112 training x1024 and 64 holdout x1024, for 131,072 training and
65,536 holdout rows. Each new row contains two 10,000-entry float64
vectors; approximately 29 GiB of new F/G banks is expected. The prior
CPU measurements of 0.15-0.23 s/sample predict about 7.5-11.5 CPU-hours
for sampling. All new banks and scheduler output are written directly to
`/nfs_scratch/park687/D01Ritz_fermi_41bd5b5/round5/` to avoid the
100 GB Ceph home quota. No determinant selection or support change occurs.
The existing precision gates remain unchanged. Iterative fit convergence
must be checked separately after the sample banks are complete.
