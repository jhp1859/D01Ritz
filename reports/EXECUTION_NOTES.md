# Execution record

2026-10-06, America/Chicago. Initial input verification and original two
tests passed at e961531. Development is on compute/d01-ritz-20261006.

Initial Condor clusters 939900/939901 matched no machines because
should_transfer_files=NO added FileSystemDomain equality. Editing only
ShouldTransferFiles and Requirements left incomplete transfer attributes
and produced starter initialization/reconnect failures before any user CPU.
Those jobs were held and removed, with logs preserved in work/scheduler.
Complete YES/ON_EXIT submissions with the existing Otten project attributes
worked. One qis2 startup-stalled chain was moved to validated qis1.

Completed short login-node pilots: M10 4.39 s/32.99 MiB and M100
5.86 s/36.62 MiB, each 200 burn-in proposals plus 16 retained samples,
stride 20. Compute-node pilots: M1000 8.30 s/39.55 MiB, M10000
9.11 s/45.01 MiB, M100000 95.40 s/233.72 MiB. These short pilots
are performance measurements, not equilibration or convergence evidence.

939902: M10/100 round1, two training and two holdout chains per M,
512 retained samples/chain, burn 4000, stride 20. M10 compute-node
chains approximately 69 s; M100 approximately 102 s.
939903: the three large-support performance pilots.
939904: M1000 (512 samples/chain) and M10000 (256 samples/chain),
two training and two holdout chains per M, burn 4000, stride 20.
939905: M100000 diagnostic bank, two training and two holdout chains,
64 samples/chain, burn 1000, stride 20. This cannot establish a
100k-dimensional metric; the empirical rank limitation is explicit.
939906: round2 M10/100 independent longer chains. M10: four train
chains x4096 and four holdout x6144. M100: eight train x4096 and
eight holdout x8192. All burn 4000, stride 20. Seeds and exact
commands are in checked-in submission files.

Only jobs submitted for this task were changed. Original inputs, main,
and unrelated jobs were preserved. No AFQMC was run.
