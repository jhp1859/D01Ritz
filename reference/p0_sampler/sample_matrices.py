#!/usr/bin/env python3
"""Metropolis evaluation of all S/Z and A/Z entries on P0.

One process owns one independent chain.  Samples retained by this program are
used only for either training or holdout, selected on the command line.
"""
from __future__ import annotations

import argparse
import functools
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from mc_common import *


def autocorr_time(x):
    x = np.asarray(x, float)
    if len(x) < 16 or np.var(x) == 0:
        return 0.5
    y = x - np.mean(x)
    n = 1 << (2 * len(y) - 1).bit_length()
    f = np.fft.rfft(y, n=n)
    ac = np.fft.irfft(f * f.conj(), n=n)[:len(y)]
    ac /= ac[0]
    tau = 0.5
    for lag in range(1, min(len(y) // 2, 10000)):
        if ac[lag] <= 0:
            break
        tau += float(ac[lag])
        if lag > 6 * tau:
            break
    return max(tau, 0.5)


def load_problem(spin):
    z = np.load(COMMON / "generator" / "state.npz", allow_pickle=False)
    alpha = np.asarray(z["alpha_bits"], np.uint64)
    beta = np.asarray(z["beta_bits"], np.uint64)
    c0 = np.asarray(z["coefficients"], float)
    orbitals = np.asarray(z["C"], complex)
    if np.max(abs(orbitals.imag)) > 2e-13:
        raise RuntimeError("standing waves unexpectedly complex")
    orbitals = orbitals.real
    diagnostics = []
    for a, b in zip(alpha, beta):
        ns = bin(int(a)^int(b)).count("1")
        diagnostics.append({"open_shells": ns,
                            "projected_norm": (1.0 if spin == "free" else 1.0/(ns//2+1)),
                            "representation": ("identity" if spin == "free" else "exact_9_point_S0")})
    evaluator = CompactSpinGeneratorAmplitudes(orbitals, alpha, beta, spin)
    return evaluator, c0, diagnostics


def chain_worker(args):
    spin, stage, chain, seed, retained, burn, stride, block_size = args
    evaluator, c0, diagnostics = load_problem(spin)
    h = lattice_k(6)

    rng = np.random.default_rng(seed)
    for _ in range(100000):
        state = random_d0(rng)
        phi, context = evaluator.prepare(state)
        weight = float(np.vdot(phi, phi).real)
        if weight > 1e-280:
            break
    else:
        raise RuntimeError("could not initialize a non-node P0 configuration")

    # Per-chain numerical gate for determinant-update evaluation.  This is
    # independent of the small-lattice oracle and exercises the actual 6x6
    # matrices, including near nodes.
    update_error = 0.0
    for neighbor in list(heff_connections(state, h))[:12]:
        updated = evaluator.from_context(neighbor, context)
        direct = evaluator(neighbor)
        update_error = max(update_error, float(np.max(abs(updated-direct))))
    if update_error > 2e-8:
        raise RuntimeError(("6x6 determinant-update gate failed", update_error))

    accepted = {"particle_hole": 0, "spin_exchange": 0}
    proposed = {"particle_hole": 0, "spin_exchange": 0}
    sample_s, sample_a, obs_num, obs_den = [], [], [], []
    total_steps = burn + retained * stride
    tick = time.monotonic()
    for step in range(total_steps):
        candidate, kind = propose_d0(state, rng)
        proposed[kind] += 1
        next_phi = evaluator.from_context(candidate, context)
        next_weight = float(np.vdot(next_phi, next_phi).real)
        if next_weight > 0 and rng.random() < min(1.0, next_weight / weight):
            state, phi, weight = candidate, next_phi, next_weight
            # Build one stable inverse context only after acceptance.
            phi, context = evaluator.prepare(state)
            weight = float(np.vdot(phi, phi).real)
            accepted[kind] += 1
        if step < burn or (step - burn) % stride:
            continue
        b = np.zeros_like(phi)
        for neighbor, hij in heff_connections(state, h).items():
            b += hij * evaluator.from_context(neighbor, context)
        srow = np.outer(phi.conj(), phi) / weight
        arow = np.outer(phi.conj(), b) / weight
        sample_s.append(srow)
        sample_a.append(arow)
        t = np.vdot(c0, phi)
        u = np.vdot(c0, b)
        obs_num.append(float(np.real(t.conjugate() * u) / weight))
        obs_den.append(float(np.real(t.conjugate() * t) / weight))

    sample_s = np.asarray(sample_s)
    sample_a = np.asarray(sample_a)
    nblocks = retained // block_size
    used = nblocks * block_size
    if nblocks < 4:
        raise RuntimeError("at least four blocks required")
    ngen = sample_s.shape[-1]
    block_s = sample_s[:used].reshape(nblocks, block_size, ngen, ngen).mean(axis=1)
    block_a = sample_a[:used].reshape(nblocks, block_size, ngen, ngen).mean(axis=1)
    obs_num = np.asarray(obs_num[:used])
    obs_den = np.asarray(obs_den[:used])
    return {
        "chain": chain, "seed": seed, "stage": stage,
        "block_s": block_s, "block_a": block_a,
        "retained": retained, "used": used, "burn": burn, "stride": stride,
        "acceptance": {k: accepted[k] / max(proposed[k], 1) for k in proposed},
        "proposed": proposed,
        "tau_num": autocorr_time(obs_num), "tau_den": autocorr_time(obs_den),
        "seconds": time.monotonic() - tick,
        "compact_quadrature_determinants": evaluator.nq,
        "determinant_update_max_abs_error": update_error,
        "diagnostics": diagnostics,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("spin", choices=("free", "singlet"))
    p.add_argument("stage", choices=("train", "holdout"))
    p.add_argument("--chains", type=int, default=16)
    p.add_argument("--samples-per-chain", type=int, default=2000)
    p.add_argument("--burn", type=int, default=5000)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--block-size", type=int, default=100)
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    p.add_argument("--replica", type=int, default=0)
    args = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("6x6 sampling must run under Slurm")
    seed_base = {"train": 61030000, "holdout": 62030000}[args.stage]
    tasks = [(args.spin, args.stage, chain, seed_base + 1000 * args.replica + chain,
              args.samples_per_chain, args.burn, args.stride, args.block_size)
             for chain in range(args.chains)]
    started = time.monotonic()
    ctx = mp.get_context("fork")
    with ctx.Pool(min(args.workers, args.chains)) as pool:
        rows = pool.map(chain_worker, tasks)
    block_s = np.concatenate([r.pop("block_s") for r in rows])
    block_a = np.concatenate([r.pop("block_a") for r in rows])
    out = MC / "matrices" / args.spin
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.stage}_rep{args.replica:03d}"
    np.savez_compressed(out / f"{stem}.npz", S_blocks=block_s, A_blocks=block_a)
    record = {
        "status": "PASS", "spin": args.spin, "stage": args.stage,
        "replica": args.replica, "chains": args.chains,
        "samples_per_chain": args.samples_per_chain,
        "total_samples": args.chains * args.samples_per_chain,
        "blocks": len(block_s), "block_size": args.block_size,
        "matrix_file": str(out / f"{stem}.npz"),
        "matrix_sha256": sha(out / f"{stem}.npz"),
        "definition": "S/Z=E_q[phi*phi/w], A/Z=E_q[phi*b/w], q=sum_I|phi_I|^2/Z",
        "normalization_warning": "These matrices are divided by unknown guide Z and are not absolute sector norms.",
        "full_SW2": "P0KP0-P0KP1KP0/U; all signed virtual D1 paths and diagonal returns",
        "rows": rows, "wall_seconds": time.monotonic() - started,
        "peak_RSS_GiB": rss_gib(), "job": os.environ["SLURM_JOB_ID"],
    }
    dump(out / f"{stem}.json", record)
    print(json.dumps({k: v for k, v in record.items() if k != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
