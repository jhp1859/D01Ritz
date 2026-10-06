#!/usr/bin/env python3
"""Verify the frozen 6x6 standing-wave supports and bundled inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "inputs"
LEVELS = (10, 100, 1000, 10000, 100000)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_state(path: Path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def main() -> None:
    manifest = json.loads((INPUTS / "MANIFEST.json").read_text())
    for name, expected in manifest["files"].items():
        actual = sha256(INPUTS / name)
        if actual != expected:
            raise AssertionError((name, actual, expected))

    states = {m: load_state(INPUTS / f"state_M{m}.npz") for m in LEVELS}
    orbital = np.asarray(states[100]["C"])
    if orbital.shape != (36, 36) or np.max(abs(orbital.imag)) > 2e-13:
        raise AssertionError("unexpected standing-wave orbital matrix")
    if np.max(abs(orbital.conj().T @ orbital - np.eye(36))) > 2e-12:
        raise AssertionError("standing-wave orbitals are not orthonormal")

    previous = None
    for m in LEVELS:
        state = states[m]
        alpha = np.asarray(state["alpha_bits"], np.uint64)
        beta = np.asarray(state["beta_bits"], np.uint64)
        coeff = np.asarray(state["coefficients"])
        if len(alpha) != m or len(beta) != m or len(coeff) != m:
            raise AssertionError((m, "incorrect support size"))
        if not np.array_equal(state["C"], orbital):
            raise AssertionError((m, "orbital matrix changed"))
        if any(bin(int(v)).count("1") != 16 for v in alpha) or any(bin(int(v)).count("1") != 16 for v in beta):
            raise AssertionError((m, "incorrect particle count"))
        support = set(zip(map(int, alpha), map(int, beta)))
        if len(support) != m:
            raise AssertionError((m, "duplicate standing-wave determinant"))
        if previous is not None and not previous.issubset(support):
            raise AssertionError((m, "ladder supports are not nested"))
        previous = support
        norm = float(np.vdot(coeff, coeff).real)
        if abs(norm - 1.0) > 1e-7:
            raise AssertionError((m, "coefficient norm", norm))

    source = states[100]
    indices = np.asarray(states[10]["source_indices"], dtype=int)
    expected_indices = np.sort(np.argsort(-abs(source["coefficients"]), kind="stable")[:10])
    if not np.array_equal(indices, expected_indices):
        raise AssertionError("M=10 is not the declared top-ten subset of M=100")
    for key in ("alpha_bits", "beta_bits"):
        if not np.array_equal(states[10][key], source[key][indices]):
            raise AssertionError((key, "M=10 support mismatch"))

    with np.load(INPUTS / "integrals_sw.npz", allow_pickle=False) as integrals:
        if integrals["h1"].shape != (36, 36) or integrals["eri"].shape != (36,)*4:
            raise AssertionError("transformed integrals have unexpected dimensions")
        if not np.array_equal(integrals["C"], orbital):
            raise AssertionError("integral and support orbital matrices differ")
        kinetic = np.zeros((36, 36))
        for x in range(6):
            for y in range(6):
                i = 6*x + y
                for xx, yy in ((x+1, y), (x, y+1)):
                    if xx < 6 and yy < 6:
                        j = 6*xx + yy
                        kinetic[i, j] = kinetic[j, i] = -1.0
        if np.max(abs(orbital.conj().T @ kinetic @ orbital - integrals["h1"])) > 2e-12:
            raise AssertionError("site ordering or standing-wave kinetic transform changed")
    print("PASS: checksums, particle counts, unique nested supports, fixed orbitals, M10 provenance")


if __name__ == "__main__":
    main()
