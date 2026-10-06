# SW400 validation and production stopping criterion

The supplied SW400 binary SHA256 is 5c2e54e52ba3a6f216a68f11374e2405a6b453902a2e72d4f416a541d7960c60;
the locally available binary is 061be8bc40455a0e46c01f00b347b588950834ecea89b4f4d9d8236de16fe545.
The supplied runtime path does not exist on this cluster. A frozen copy of
the local runtime is used and hashed. Exact cross-build support reproduction
is not assumed.

An independent contact-Hubbard Slater-Condon matrix verifies the archived
SW400 energy (0.9481987916643421) and its lowest eigenvalue. The archived
coefficient residual is 1.28132264e-6, despite a nominal residual_tol=1e-9,
because native Davidson also stops on energy_tol=1e-11. The initial strict
residual test failure is preserved in sw400_initial_diagnostic.txt.

The local SW400 growth begins from exactly the supplied M1. It agrees with
the archived energy history through M64, then follows a different support
and ends at E=0.9547868396. This is NOT an exact archive reproduction. Its
full-H energy, lowest eigenvalue, coefficient normalization, Fermi inclusion,
particle counts, unchanged C and growth history are checked independently.
The first native outputs are retained in work/sw400_attempt_939909.

SW400 is a method-validation run, never a target support or parent of M100.
M100 starts again from M1. Production phases retain the archived settings
except energy_tol=-1, which disables the energy-change-only exit; residual_tol
remains 1e-9. This explicit stricter termination can change selections among
nearly tied candidates. The resulting actual supports, not an assumption
of archive equivalence, are frozen and checksummed.
