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
