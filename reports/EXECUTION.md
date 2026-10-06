# Corrected execution and cancellation record

The emergency correction removed 30 jobs: 939905.0-3, 939906.0-23,
939907.0-1 (17 running, 13 idle). Completed old clusters 939902-939904
were preserved. All old data remain in the invalid D01Ritz checkout, with
262 files inventoried and hashed. No unrelated job was changed.

Corrected clone: D01Ritz_fermi_41bd5b5, branch compute/fermi-d01-ritz-20261006.
Input verification and original unit tests passed at commit 41bd5b5.
The target supports were absent and were genuinely generated from M1.
M100/M10 were ready before 14:59 CDT, earlier than the preliminary
15:05-15:15 estimate. The full 1k/10k/100k ladder was then generated and
verified, including nested determinant sets and unchanged orbitals.

- 939908: initial growth settings audit failed on wrapper mapping of
  pt2_correction; no target support was generated.
- 939909: SW400 independent residual check exposed the archived-style
  energy-tolerance early exit. Outputs preserved as sw400_attempt_939909.
- 939910: independent SW400 method check plus separate M1->M100 and M10.
- 939911: fresh M10/100 short training/holdout chains.
- 939912: M100->1k->10k->100k full-H growth, strict residual stopping.
- 939913: initial corrected M10/100 Ritz analysis.
- 939914: initial 1k/10k banks; 939915: 100k diagnostic banks.
- 939916: long M10/100 study (8x4096 training, 16x8192 holdout per M).
- 939917: initial 1k/10k analysis; 939919: 100k analysis.
- 939918: qis3/qis4 worker probes. qis3 passed; qis4 had START=false,
  so only its probe was removed. No machine policy was changed.
- 939920: enlarged 1k (8x2048 train and holdout) and 10k
  (16x1024 train, 8x1024 holdout) banks.
- 939921: enlarged 10k matrix-free LOBPCG analysis, 4 CPUs/16 GiB.
- 939922/939923: final M10/1k analyses.
- 939924: final M100 analysis; 939925: residual-vector uncertainty audits.

Scheduler disk requests were reduced after measured scratch usage proved
small: banks/checkpoints are written directly to Ceph, not execute scratch.
qis1 and validated qis3 were used. M10/100 short chains took approximately
60-83 s; initial 1k/10k chains took at most 141/96 s. The 100k diagnostic
chains took 453-531 s and approximately 345 MiB peak RSS each. Queue delays,
filesystem flushes and the extra precision rounds explain the later study
completion relative to a single-chain pilot estimate.

No AFQMC was run. Original main and frozen input files were not changed.
Submitted arguments, logs, banks and RNG checkpoints are preserved in the
cluster artifact manifest. The records distinguish full-H support growth
from D01 Ritz fitting, and distinguish fit completion from convergence.

M100 round2 passed every gate except the strict primary-block train/holdout
agreement: difference 0.2085904071 versus threshold 0.2066890858. The
threshold was NOT relaxed to the larger conservative-error threshold.
A final predetermined confirmation keeps the exact round2 training bank
and therefore the fitted coefficients fixed, and draws 32 entirely fresh
holdout chains x4096 (seeds 530201-530232). Round3 training directories are
read-only-use symlinks to the corrected round2 training banks, never to the
invalid old checkout. The prior failed result remains intact. Only the
fresh holdout is used for this confirmation, with the same gates.
