# Additional precision after the first corrected run

The M100 round3 holdout of the unchanged round2 fit missed the previously
declared training/holdout agreement gate. The round4 experiment increases
training as well as drawing entirely fresh holdout chains. No previous failed
result is overwritten. `scripts/prepare_round4.py` generated the symlinks,
seeds and `round4_sampling.sub` before submission. Cluster 939928 contains 152
new chains; old symlinked training banks come only from corrected round2.

| M | Corrected old train | New train | New holdout | New samples per chain |
|---:|---:|---:|---:|---:|
| 100 | 8 x 4096 | 24 x 4096 | 32 x 4096 | 4096 |
| 1000 | 8 x 2048 | 48 x 2048 | 48 x 2048 | 2048 |

Measured CPU per sample from preceding runs: M100 median 0.0793 s,
M1000 median 0.1730 s. The new chains therefore cost approximately
5.0 and 9.5 CPU-hours, respectively. Single M100 chains previously took
up to 739 s, and M1000 chains up to 622 s elapsed. Final completion depends
on the scheduler and Ceph load. The existing cutoffs and all declared gates
remain unchanged. Analysis starts only after every bank for a given M is
finished and hashed. The held-out sample is never reused for fitting.

The Ceph home quota is 100,000,000,000 bytes, with zero file-count quota.
At 100,158,238,989 bytes, the remaining 139 jobs went on hold while
transferring scheduler output. No unrelated job was changed. All round4
banks/checkpoints were copied to
`/nfs_scratch/park687/D01Ritz_fermi_41bd5b5/round4/`; a complete
`rsync -ani --checksum` comparison of old and new copies produced no
differences. Only then was the redundant Ceph round4 copy removed and
`work/round4` replaced with a symlink to that scratch directory. Ceph home
usage fell to 98,599,307,918 bytes. Job 939928.1 resumed from 4080/4096
samples to completion with exit code 0; the remaining 939928 jobs were
released afterward. The source code and old corrected analysis outputs
remain in the Git checkout and previous result directories.
