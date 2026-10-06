# Corrected bare-Fermi-origin D01 Ritz results

Input commit `41bd5b5`; single M1 -> full-H M100 -> 1k -> 10k -> 100k. M10 is the top-ten subset of M100. The e961531 matched-random experiment is invalid and isolated. No old matrices, coefficients or checkpoints are reused.

All support files are frozen in `supports/`; checksums and parentage are in its MANIFEST.json. SW400 is only a method check; the different native build and stricter production stopping criterion are documented in SW400_VALIDATION_POLICY.md.

M100 round3 is a fresh independent holdout confirmation of the byte-identical round2 coefficient file. Round2 narrowly failed train/holdout agreement; that failure is preserved and its threshold was not relaxed.

## Ritz estimates

Unresolved rows are diagnostic coefficient fits, **not converged lowest Ritz energies**. Errors are conditional finite-chain estimates; see full block, seed, cutoff/ridge and sample-count diagnostics.

| M | Training E | Independent holdout E ± SE | Holdout relative residual ± bootstrap SD | Status |
|---:|---:|---:|---:|---|
| 10 | -27.778774 | -27.828070 ± 0.059371 | 0.00259 ± 0.00042 | converged_at_declared_sampling_precision |
| 100 | -28.121583 | -27.812554 ± 0.078149 | 0.00629 ± 0.00038 | unresolved |
| 1,000 | -29.803305 | -28.393259 ± 0.223278 | 0.02487 ± 0.00114 | unresolved |
| 10,000 | -43.892033 | -20.685284 ± 0.171310 | 0.12733 ± 0.00453 | unresolved |
| 100,000 | -46.434431 | -20.889190 ± 1.513199 | 0.13521 ± 0.02460 | unresolved |

M1000 remains limited by energy precision and sample/seed stability despite full empirical overlap rank. M10000 also fails the unregularized training residual and holdout residual gates; its ridge LOBPCG output is a failed diagnostic fit. M100000 has only 128 training rows and retained rank 120, so its diagnostic sample bank cannot identify a 100000-dimensional Ritz solution. More sampling and solver work are required; these levels are not declared solved.

## Resources and diagnostics

CPU hours below are sampling process CPU; full analysis and normalization CPU, active wall times, per-chain seeds, all support/coefficient hashes and failures are in SUMMARY.csv and RESULT_MANIFEST.json. Peak RSS is per process, not the sum of concurrent jobs. Queue waiting time is excluded.

| M | Train / holdout samples | Sampling CPU h | Peak RSS MiB | Overlap rank | Failure gates |
|---:|---:|---:|---:|---:|---|
| 10 | 32,768 / 131,072 | 3.166 | 90.6 | 10 |  |
| 100 | 32,768 / 131,072 | 3.280 | 407.0 | 100 | train_holdout_disagreement |
| 1,000 | 16,384 / 16,384 | 1.572 | 901.0 | 1000 | holdout_energy_precision; sample_count_stability; seed_stability; train_holdout_disagreement |
| 10,000 | 16,384 / 8,192 | 1.192 | 4429.4 | None | holdout_energy_precision; holdout_residual; residual_vector_uncertainty_gate; sample_count_stability; seed_stability; train_holdout_disagreement; training_projected_residual |
| 100,000 | 128 / 128 | 0.466 | 637.4 | 120 | holdout_energy_precision; holdout_residual; insufficient_effective_samples; insufficient_sector_crossings; rank_deficient_empirical_metric; residual_vector_uncertainty_gate; seed_stability; train_holdout_disagreement |

Coefficients are normalized to empirical training S/Z, with a separate physical-norm estimate and its uncertainty. Unknown guide Z is not treated as one. Large sample banks and exact RNG checkpoints remain on Ceph at the manifest paths; re-running a listed submission command resumes committed samples. No AFQMC was run.
