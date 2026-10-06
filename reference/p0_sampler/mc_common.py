"""Shared algebra for the 6x6 stochastic projected-subspace experiment.

The determinant convention is all alpha spin orbitals followed by all beta
spin orbitals.  Site bits and standing-wave orbital bits use x-major order.
No full P0 or P1 configuration list is ever constructed for the 6x6 run.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import resource
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

RUN_KEY = "h6x6_obc_u8_n16n16_sw2_k100_eval_v1_exactflow"
ROOT = Path("/mnt/ceph/users/jpark1/runs") / RUN_KEY
COMMON = ROOT / "common"
MC = ROOT / "mc"
N = 36
NUP = NDN = 16
U = 8.0
AFQMC_SEEDS = (6801608, 6801609, 6801610)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False, default=str) + "\n")
    tmp.replace(path)


def table(path: Path, rows) -> None:
    rows = list(rows)
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def occs(bits: int, n: int = N) -> np.ndarray:
    return np.fromiter((i for i in range(n) if (bits >> i) & 1), dtype=np.int16)


def lattice_k(side: int) -> np.ndarray:
    h = np.zeros((side * side, side * side))
    for x in range(side):
        for y in range(side):
            i = side * x + y
            for xx, yy in ((x + 1, y), (x, y + 1)):
                if xx < side and yy < side:
                    j = side * xx + yy
                    h[i, j] = h[j, i] = -1.0
    return h


def hop_one(bits: int, src: int, dst: int):
    if not ((bits >> src) & 1) or ((bits >> dst) & 1):
        return None
    lo, hi = sorted((src, dst))
    between = ((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1)
    sign = -1.0 if bin(bits & between).count("1") & 1 else 1.0
    return bits ^ (1 << src) ^ (1 << dst), sign


def kinetic_hops(state, h):
    up, dn = map(int, state)
    for spin, bits in enumerate((up, dn)):
        for dst, src in zip(*np.nonzero(h)):
            moved = hop_one(bits, int(src), int(dst))
            if moved is None:
                continue
            nb, sign = moved
            yield ((nb, dn) if spin == 0 else (up, nb)), float(h[dst, src]) * sign


def doublons(state) -> int:
    return bin(int(state[0]) & int(state[1])).count("1")


def heff_connections(state, h, u: float = U):
    """Column H_eff|state>, including all signed virtual D=1 paths."""
    if doublons(state) != 0:
        raise ValueError("H_eff external state must be D=0")
    out = defaultdict(float)
    virtual = []
    for mid, amp in kinetic_hops(state, h):
        d = doublons(mid)
        if d == 0:
            out[mid] += amp
        elif d == 1:
            virtual.append((mid, amp))
    for mid, amp in virtual:
        for final, second in kinetic_hops(mid, h):
            if doublons(final) == 0:
                out[final] -= amp * second / u
    return {q: v for q, v in out.items() if abs(v) > 1e-14}


def _act(mask: int, orbital: int, create: bool):
    occupied = bool(mask & (1 << orbital))
    if occupied == create:
        return None
    sign = -1.0 if bin(mask & ((1 << orbital) - 1)).count("1") & 1 else 1.0
    return mask ^ (1 << orbital), sign


def sigma_walk(alpha: int, beta: int, flips: int) -> float:
    """Fermion sign of the certified ascending-orbital spin-flip walk."""
    alpha, beta, flips = int(alpha), int(beta), int(flips)
    parity = 0
    while flips:
        bit = flips & -flips
        p = bit.bit_length() - 1
        flips ^= bit
        below = bit - 1
        if alpha & bit:                 # alpha -> beta
            na = bin(alpha).count("1")
            parity += (bin(alpha & below).count("1") + na - 1 +
                       bin(beta & below).count("1"))
            alpha ^= bit
            beta |= bit
        else:                           # beta -> alpha
            alpha2 = alpha | bit
            na2 = bin(alpha2).count("1")
            parity += (bin(alpha2 & below).count("1") + na2 - 1 +
                       bin(beta & below).count("1") + 1)
            alpha = alpha2
            beta ^= bit
    return -1.0 if parity & 1 else 1.0


def singlet_projected_column(alpha: int, beta: int, norb: int):
    """Exact P_{S=0}|alpha,beta> as an orbital-determinant dictionary."""
    alpha, beta = int(alpha), int(beta)
    if bin(alpha).count("1") != bin(beta).count("1"):
        raise ValueError("singlet projection requires M_S=0")
    doub = alpha & beta
    open_mask = alpha ^ beta
    open_orbs = [p for p in range(norb) if (open_mask >> p) & 1]
    ns = len(open_orbs)
    if ns % 2:
        raise ValueError("odd open-shell count")
    k = ns // 2
    m0 = []
    for chosen in itertools.combinations(open_orbs, k):
        a = doub | sum(1 << p for p in chosen)
        b = doub | (open_mask ^ sum(1 << p for p in chosen))
        m0.append((a, b))
    # Closed S=0 projector kernel.  If 2r open orbitals are flipped,
    # I(2k,2r)=r!(k-r)!/(k+1)!=1/((k+1) C(k,r)); sigma_walk supplies the
    # determinant-ordering sign.  This is algebraically the same kernel as
    # the validated native implementation, without a C(2k,k)^3 eigensolve.
    column = []
    for a, b in m0:
        flips = (alpha ^ a) & open_mask
        f = bin(flips).count("1")
        if f & 1:
            raise RuntimeError("M_S=0 spin orbit produced an odd flip count")
        r = f // 2
        value = sigma_walk(alpha, beta, flips) / ((k + 1) * math.comb(k, r))
        column.append(value)
    column = np.asarray(column)
    expected_weight = 1.0 / (k + 1)
    if abs(float(column @ column) - expected_weight) > 2e-11:
        raise RuntimeError(("singlet projector diagonal check", ns, column @ column, expected_weight))
    return {q: float(x) for q, x in zip(m0, column) if abs(x) > 2e-14}, {
        "open_shells": ns,
        "M0_dimension": len(m0),
        "singlet_rank": int(math.comb(2*k, k) // (k + 1)),
        "projected_norm": float(column @ column),
        "S2_residual": 0.0,
    }


def build_generator_expansion(alpha, beta, spin: str, norb: int = N):
    alpha = np.asarray(alpha, np.uint64)
    beta = np.asarray(beta, np.uint64)
    columns = []
    diagnostics = []
    for a, b in zip(alpha, beta):
        if spin == "free":
            columns.append({(int(a), int(b)): 1.0})
            diagnostics.append({"open_shells": bin(int(a) ^ int(b)).count("1"),
                                "M0_dimension": 1, "singlet_rank": None,
                                "projected_norm": 1.0, "S2_residual": None})
        elif spin == "singlet":
            col, rec = singlet_projected_column(int(a), int(b), norb)
            columns.append(col)
            diagnostics.append(rec)
        else:
            raise ValueError(spin)
    states = sorted(set().union(*(set(c) for c in columns)))
    lookup = {q: i for i, q in enumerate(states)}
    transform = np.zeros((len(states), len(columns)))
    for j, col in enumerate(columns):
        for q, value in col.items():
            transform[lookup[q], j] = value
    return states, transform, diagnostics


class GeneratorAmplitudes:
    """Site amplitudes of all projected/unprojected generators."""
    def __init__(self, orbitals, states, transform):
        self.c = np.asarray(orbitals, float)
        self.states = list(states)
        self.transform = np.asarray(transform, float)
        self.oa = [occs(q[0], len(self.c)) for q in self.states]
        self.ob = [occs(q[1], len(self.c)) for q in self.states]

    def determinant_amplitudes(self, state):
        up, dn = map(int, state)
        su, sd = occs(up, len(self.c)), occs(dn, len(self.c))
        return np.asarray([np.linalg.det(self.c[np.ix_(su, oa)]) *
                           np.linalg.det(self.c[np.ix_(sd, ob)])
                           for oa, ob in zip(self.oa, self.ob)])

    def __call__(self, state):
        return self.determinant_amplitudes(state) @ self.transform


class CompactSpinGeneratorAmplitudes:
    """All generator amplitudes using compact exact spin-rotation quadrature.

    A prepared configuration carries inverse overlap matrices, so every
    rank-one/rank-two local SW2 neighbor is evaluated by determinant updates
    rather than by repeating 32x32 determinants.
    """
    def __init__(self, orbitals, alpha, beta, spin: str):
        c = np.asarray(orbitals, float)
        self.n = len(c)
        self.kgen = len(alpha)
        if spin == "free":
            nodes = np.asarray([1.0])
            weights = np.asarray([1.0])
        elif spin == "singlet":
            # In x=cos(beta), every surviving M_S=0 term is a polynomial of
            # degree <= Ne/2=16.  Nine-point Gauss-Legendre is therefore
            # exact through degree 17.
            nodes, glw = np.polynomial.legendre.leggauss(9)
            weights = 0.5 * glw
        else:
            raise ValueError(spin)
        rows, qweights, owners = [], [], []
        for ig, (abits, bbits) in enumerate(zip(alpha, beta)):
            oa, ob = occs(int(abits), self.n), occs(int(bbits), self.n)
            A, B = c[:, oa], c[:, ob]
            for inode, x in enumerate(nodes):
                if spin == "free":
                    ca, sa = 1.0, 0.0
                else:
                    ca = math.sqrt(max(0.0, 0.5*(1.0+float(x))))
                    sa = math.sqrt(max(0.0, 0.5*(1.0-float(x))))
                # Rows are site spin orbitals; columns are occupied ket
                # spinors, ordered original alpha then original beta.
                ket = np.block([[ca*A, -sa*B], [sa*A, ca*B]])
                rows.append(ket)
                qweights.append(float(weights[inode]))
                owners.append(ig)
        self.rows = np.asarray(rows)
        self.weights = np.asarray(qweights)
        self.owners = np.asarray(owners, dtype=np.int16)
        self.nq = len(rows)
        self.spin = spin

    def labels(self, state):
        a, b = map(int, state)
        return np.concatenate([occs(a, self.n), self.n + occs(b, self.n)]).astype(np.int16)

    def _collect(self, dets):
        value = np.zeros(self.kgen)
        np.add.at(value, self.owners, self.weights * dets)
        return value

    def prepare(self, state):
        labels = self.labels(state)
        mats = self.rows[:, labels, :]
        dets = np.linalg.det(mats)
        inv = np.empty_like(mats)
        good = np.ones(self.nq, dtype=bool)
        try:
            inv[:] = np.linalg.inv(mats)
            good &= np.all(np.isfinite(inv), axis=(1, 2))
            good &= np.max(abs(inv), axis=(1, 2)) < 1e12
        except np.linalg.LinAlgError:
            # Batched inv aborts at the first singular member; recover each.
            for q, mat in enumerate(mats):
                try:
                    inv[q] = np.linalg.inv(mat)
                    good[q] = (np.all(np.isfinite(inv[q])) and
                               np.max(abs(inv[q])) < 1e12)
                except np.linalg.LinAlgError:
                    inv[q].fill(np.nan); good[q] = False
        return self._collect(dets), {"labels": labels, "dets": dets,
                                    "inverse": inv, "good": good}

    def from_context(self, state, context):
        old = context["labels"].tolist()
        new = self.labels(state).tolist()
        old_set, new_set = set(old), set(new)
        removed_pos = [i for i, label in enumerate(old) if label not in new_set]
        added = sorted(label for label in new if label not in old_set)
        if len(removed_pos) != len(added) or len(added) > 2:
            # This should never occur for the two-hop SW2 stencil.
            return self.prepare(state)[0]
        if not added:
            return self._collect(context["dets"])
        temp = list(old)
        for pos, label in zip(removed_pos, added):
            temp[pos] = label
        inversions = sum(temp[i] > temp[j] for i in range(len(temp))
                         for j in range(i+1, len(temp)))
        psign = -1.0 if inversions & 1 else 1.0
        new_rows = self.rows[:, added, :]
        cols = context["inverse"][:, :, removed_pos]
        small = np.matmul(new_rows, cols)
        if len(added) == 1:
            ratio = small[:, 0, 0]
        else:
            ratio = small[:, 0, 0]*small[:, 1, 1] - small[:, 0, 1]*small[:, 1, 0]
        ndet = context["dets"] * ratio * psign
        # A low-rank determinant update loses relative accuracy when the new
        # determinant is itself a node (cancellation in the small determinant),
        # even if the old matrix is well conditioned.  Recompute those rare
        # members directly; this is also the explicit zero-amplitude path.
        bad = (~context["good"] | ~np.isfinite(ratio) | (abs(ratio) < 1e-10))
        if np.any(bad):
            labels = np.asarray(new, dtype=np.int16)
            ndet[bad] = np.linalg.det(self.rows[bad][:, labels, :])
        return self._collect(ndet)

    def __call__(self, state):
        labels = self.labels(state)
        return self._collect(np.linalg.det(self.rows[:, labels, :]))


def random_d0(rng, n=N, nu=NUP, nd=NDN):
    sites = np.arange(n)
    up = rng.choice(sites, nu, replace=False)
    remaining = np.setdiff1d(sites, up, assume_unique=False)
    dn = rng.choice(remaining, nd, replace=False)
    return sum(1 << int(i) for i in up), sum(1 << int(i) for i in dn)


def propose_d0(state, rng, n=N):
    up, dn = map(int, state)
    if rng.random() < 0.5:
        spin = int(rng.integers(2))
        bits = (up, dn)[spin]
        occupied = [i for i in range(n) if (bits >> i) & 1]
        holes = [i for i in range(n) if not ((up | dn) >> i) & 1]
        src = occupied[int(rng.integers(len(occupied)))]
        dst = holes[int(rng.integers(len(holes)))]
        moved = bits ^ (1 << src) ^ (1 << dst)
        return (moved, dn) if spin == 0 else (up, moved), "particle_hole"
    only_up = [i for i in range(n) if ((up >> i) & 1) and not ((dn >> i) & 1)]
    only_dn = [i for i in range(n) if ((dn >> i) & 1) and not ((up >> i) & 1)]
    i = only_up[int(rng.integers(len(only_up)))]
    j = only_dn[int(rng.integers(len(only_dn)))]
    return (up ^ (1 << i) ^ (1 << j), dn ^ (1 << i) ^ (1 << j)), "spin_exchange"


def canonical_lowest(a, s, cutoff=1e-6):
    from scipy.linalg import eigh
    a = (np.asarray(a) + np.asarray(a).T.conj()) * 0.5
    s = (np.asarray(s) + np.asarray(s).T.conj()) * 0.5
    values, vectors = eigh(s, check_finite=False)
    threshold = max(np.finfo(float).eps * len(values) * values[-1], cutoff * values[-1])
    keep = values > threshold
    if not np.any(keep):
        raise RuntimeError("metric has zero retained rank")
    x = vectors[:, keep] / np.sqrt(values[keep])[None, :]
    h = x.T.conj() @ a @ x
    energy, y = eigh((h + h.T.conj()) * 0.5, subset_by_index=[0, 0], check_finite=False)
    c = x @ y[:, 0]
    c /= np.sqrt(np.real(c.conj() @ s @ c))
    residual = a @ c - energy[0] * (s @ c)
    return np.real_if_close(c), {
        "energy": float(np.real(energy[0])), "rank": int(np.count_nonzero(keep)),
        "nullity": int(len(values) - np.count_nonzero(keep)),
        "cutoff": cutoff, "absolute_cutoff": float(threshold),
        "metric_eigenvalues": values.tolist(),
        "generalized_residual": float(np.linalg.norm(residual)),
    }
