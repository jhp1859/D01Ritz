#!/usr/bin/env python3
"""2x2 exact oracle for signs, SW2, spin projection, and stochastic H/S."""
from __future__ import annotations

import itertools
import json
import os
import time

import numpy as np

from mc_common import (canonical_lowest, doublons, heff_connections, kinetic_hops,
                       lattice_k, singlet_projected_column, sigma_walk, dump,
                       build_generator_expansion, GeneratorAmplitudes,
                       CompactSpinGeneratorAmplitudes,
                       MC, rss_gib)


def bits(n, k):
    return [sum(1 << i for i in q) for q in itertools.combinations(range(n), k)]


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("run validation on a compute node")
    t0 = time.monotonic()
    n, side, u = 4, 2, 8.0
    h = lattice_k(side)
    full = [(a, b) for a in bits(n, 2) for b in bits(n, 2)]
    p0 = [q for q in full if doublons(q) == 0]
    p1 = [q for q in full if doublons(q) == 1]
    index = {q: i for i, q in enumerate(full)}
    K = np.zeros((len(full), len(full)))
    for j, q in enumerate(full):
        for r, v in kinetic_hops(q, h):
            K[index[r], j] += v
    i0 = [index[q] for q in p0]
    i1 = [index[q] for q in p1]
    direct = K[np.ix_(i0, i0)] - K[np.ix_(i0, i1)] @ K[np.ix_(i1, i0)] / u
    local = np.zeros_like(direct)
    for j, q in enumerate(p0):
        for r, v in heff_connections(q, h, u).items():
            local[p0.index(r), j] += v
    heff_error = float(np.max(abs(local-direct)))

    # Four simple orbital determinants in the 2x2 standing-wave basis.
    _, C = np.linalg.eigh(h)
    generators = [(3, 3), (3, 5), (5, 3), (5, 10)]

    def site_amp(q, orbital_state):
        a, b = q; oa, ob = orbital_state
        su = [i for i in range(n) if (a >> i) & 1]
        sd = [i for i in range(n) if (b >> i) & 1]
        au = [i for i in range(n) if (oa >> i) & 1]
        ad = [i for i in range(n) if (ob >> i) & 1]
        return np.linalg.det(C[np.ix_(su, au)]) * np.linalg.det(C[np.ix_(sd, ad)])

    results = {}
    for spin in ("free", "singlet"):
        cols = []
        spin_res = 0.0
        for a, b in generators:
            if spin == "free":
                col = {(a, b): 1.0}
            else:
                col, diag = singlet_projected_column(a, b, n)
                spin_res = max(spin_res, diag["S2_residual"])
            cols.append(col)
        V = np.asarray([[sum(v*site_amp(x, q) for q, v in col.items())
                         for col in cols] for x in p0])
        states, transform, _ = build_generator_expansion(
            np.asarray([q[0] for q in generators], np.uint64),
            np.asarray([q[1] for q in generators], np.uint64), spin, n)
        explicit = GeneratorAmplitudes(C, states, transform)
        compact = CompactSpinGeneratorAmplitudes(
            C, np.asarray([q[0] for q in generators], np.uint64),
            np.asarray([q[1] for q in generators], np.uint64), spin)
        compact_error = max(float(np.max(abs(explicit(x)-compact(x)))) for x in full)
        S = V.T @ V
        A = V.T @ direct @ V
        # Independently form b_J(x) through local connections.
        B = np.zeros_like(V)
        lookup = {x: i for i, x in enumerate(p0)}
        for ix, x in enumerate(p0):
            for y, hij in heff_connections(x, h, u).items():
                B[ix] += hij * V[lookup[y]]
        action_error = float(np.max(abs(A - V.T @ B)))

        w = np.sum(V*V, axis=1)
        Z = float(w.sum())
        valid = w > 1e-30
        prob = np.where(valid, w/Z, 0.0)
        rng = np.random.default_rng(9917 if spin == "free" else 9918)
        nsamp = 400000
        draw = rng.choice(len(p0), nsamp, p=prob)
        Shat = np.mean(np.asarray([np.outer(V[i], V[i])/w[i] for i in draw]), axis=0)
        Ahat = np.mean(np.asarray([np.outer(V[i], B[i])/w[i] for i in draw]), axis=0)
        # Unknown Z cancels; compare trace-normalized matrices and Ritz roots.
        Sn, An = S/np.trace(S), A/np.trace(S)
        Snh, Anh = Shat/np.trace(Shat), Ahat/np.trace(Shat)
        _, exact_solve = canonical_lowest(An, Sn, 1e-10)
        _, mc_solve = canonical_lowest(Anh, Snh, 1e-10)
        results[spin] = {
            "rank": int(np.linalg.matrix_rank(S, tol=1e-11)),
            "S_trace_normalized_max_error": float(np.max(abs(Snh-Sn))),
            "A_trace_normalized_max_error": float(np.max(abs(Anh-An))),
            "Ritz_exact": exact_solve["energy"], "Ritz_MC": mc_solve["energy"],
            "Ritz_abs_error": abs(exact_solve["energy"]-mc_solve["energy"]),
            "local_action_max_error": action_error,
            "compact_amplitude_max_error": compact_error,
            "spin_projection_residual": spin_res, "samples": nsamp,
        }

    # Directly compare analytic sigma/I projector with the older nullspace
    # oracle for the first nontrivial 4-open-shell column.
    a, b = 3, 12
    col, d = singlet_projected_column(a, b, n)
    # S+ annihilation is an exact singlet check in M=0.
    raised = {}
    for (aa, bb), value in col.items():
        for p in range(n):
            if (bb >> p) & 1 and not ((aa >> p) & 1):
                # Apply c^+_{a,p} c_{b,p} in all-alpha/all-beta convention.
                fullmask = aa | (bb << n)
                def act(mask, orb, create):
                    if bool(mask & (1 << orb)) == create: return None
                    s = -1.0 if bin(mask & ((1 << orb)-1)).count("1") & 1 else 1.0
                    return mask ^ (1 << orb), s
                q1 = act(fullmask, n+p, False); q2 = act(q1[0], p, True)
                raised[q2[0]] = raised.get(q2[0], 0.0) + value*q1[1]*q2[1]
    direct_spin_res = float(np.sqrt(sum(v*v for v in raised.values())))

    fail = (heff_error > 1e-13 or direct_spin_res > 1e-12 or
            max(r["local_action_max_error"] for r in results.values()) > 1e-12 or
            max(r["compact_amplitude_max_error"] for r in results.values()) > 1e-12 or
            max(r["Ritz_abs_error"] for r in results.values()) > 1.5e-2)
    record = {"status": "FAIL" if fail else "PASS", "system": "2x2 OBC U8 Nup=Ndn=2",
              "heff_direct_max_error": heff_error,
              "projector_Splus_norm": direct_spin_res,
              "projector_diagonal_weight": d["projected_norm"],
              "branches": results, "wall_seconds": time.monotonic()-t0,
              "peak_RSS_GiB": rss_gib(), "job": os.environ["SLURM_JOB_ID"]}
    dump(MC/"validation"/"small_exact.json", record)
    print(json.dumps(record, indent=2), flush=True)
    if fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
