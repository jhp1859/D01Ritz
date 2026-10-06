#!/usr/bin/env python3
"""Independent small-lattice matrix check of the canonical D<=1 SW2 term."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

OUT = Path('/mnt/ceph/users/jpark1/runs/swfixed_d01_controls_20261006/tests/SW2_SMALL_SYSTEM.json')


def one_spin_hops(nsites, nelec):
    bits = [sum(1 << i for i in occ) for occ in itertools.combinations(range(nsites), nelec)]
    index = {mask: i for i, mask in enumerate(bits)}
    k = np.zeros((len(bits), len(bits)))
    for i, mask in enumerate(bits):
        for site in range(nsites - 1):
            jsite = site + 1
            if ((mask >> site) & 1) == ((mask >> jsite) & 1):
                continue
            moved = mask ^ (1 << site) ^ (1 << jsite)
            # Neighbor hop: no occupied orbitals strictly between the sites.
            k[index[moved], i] = -1.0
    return bits, k


def system(nsites, nup, ndn, u):
    bu, ku = one_spin_hops(nsites, nup)
    bd, kd = one_spin_hops(nsites, ndn)
    k = np.kron(ku, np.eye(len(bd))) + np.kron(np.eye(len(bu)), kd)
    d = np.array([(up & dn).bit_count() for up in bu for dn in bd], dtype=int)
    h0 = np.diag(u * d)
    return d, h0, k, h0 + k


def check(nsites, nup, ndn, u):
    d, h0, k, h = system(nsites, nup, ndn, u)
    p = np.flatnonzero(d <= 1)
    p1 = np.flatnonzero(d == 1)
    q2 = np.flatnonzero(d == 2)
    q = np.flatnonzero(d >= 2)
    php = h[np.ix_(p, p)]
    sw2 = php.copy()
    if len(q2):
        corr = k[np.ix_(p1, q2)] @ k[np.ix_(q2, p1)] / u
        local = np.searchsorted(p, p1)
        sw2[np.ix_(local, local)] -= corr
    s = np.zeros_like(h)
    for i in p:
        for j in q:
            if k[i, j] != 0:
                s[i, j] = -k[i, j] / (u * (d[j] - d[i]))
                s[j, i] = -s[i, j]
    off = h[np.ix_(p, q)]
    voff = np.zeros_like(h)
    voff[np.ix_(p, q)] = off
    voff[np.ix_(q, p)] = off.T
    independent = php + .5 * (s @ voff - voff @ s)[np.ix_(p, p)]
    matrix_error = float(np.max(np.abs(sw2 - independent)))
    hermiticity_error = float(np.max(np.abs(sw2 - sw2.T)))
    exact_spectrum = np.linalg.eigvalsh(h)
    sw2_spectrum = np.linalg.eigvalsh(sw2)
    low_count = min(6, len(p))
    return dict(nsites=nsites, nup=nup, ndn=ndn, U=u,
                full_dimension=len(d), P_dimension=len(p), Q_dimension=len(q),
                Q2_dimension=len(q2), sw2_matrix_vs_independent_error=matrix_error,
                hermiticity_error=hermiticity_error,
                no_Q_full_H_error=float(np.max(np.abs(sw2-h))) if not len(q) else None,
                exact_low_spectrum=exact_spectrum[:low_count].tolist(),
                sw2_low_spectrum=sw2_spectrum[:low_count].tolist())


def main():
    rows = [check(2, 1, 1, 8.0)]
    rows += [check(4, 2, 2, u) for u in (8.0, 16.0, 32.0, 64.0)]
    assert all(row['sw2_matrix_vs_independent_error'] < 1e-12 for row in rows)
    assert all(row['hermiticity_error'] < 1e-12 for row in rows)
    assert rows[0]['Q_dimension'] == 0 and rows[0]['no_Q_full_H_error'] == 0
    result = dict(status='PASS', convention='e^S H e^-S, [S,H0]=-V_PQ',
                  operator='PHP-Pi1 K Pi2 K Pi1/U',
                  checks=rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
