"""Site-basis action of the canonical Hermitian D<=1 Hubbard SW2 operator.

The ordered spin orbitals are all up sites followed by all down sites.  Within
each spin sector, site bits follow the archived x-major standing-wave order.
This module supplies local signed paths; it does not enumerate the 6x6 P space.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def doublons(state) -> int:
    return bin(int(state[0]) & int(state[1])).count("1")


def hop_one(bits: int, src: int, dst: int):
    if not (bits >> src) & 1 or (bits >> dst) & 1:
        return None
    lo, hi = sorted((src, dst))
    between = ((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1)
    sign = -1 if bin(bits & between).count("1") % 2 else 1
    return bits ^ (1 << src) ^ (1 << dst), sign


def kinetic_hops(state, h):
    up, down = map(int, state)
    destinations, sources = np.nonzero(h)
    for spin, bits in enumerate((up, down)):
        for dst, src in zip(destinations, sources):
            moved = hop_one(bits, int(src), int(dst))
            if moved is None:
                continue
            next_bits, sign = moved
            next_state = (next_bits, down) if spin == 0 else (up, next_bits)
            yield next_state, h[dst, src] * sign


def heff01_connections(state, h, u: float = 8.0):
    """Return row amplitude for each site ket reached by H_eff^(01)|state>.

    H_eff^(01) = P H P - P1 K P2 K P1 / U, P=P0+P1.  H0=U*Dhat,
    convention exp(S) H exp(-S) with [S,H0]=-V_PQ.  This is the O(K^2/U)
    correction from Q=D>=2; PHP is kept unexpanded.  D0<->D1 hopping is
    present directly in PHP and must not be added again as a virtual path.
    """
    state = tuple(map(int, state))
    d = doublons(state)
    if d > 1:
        raise ValueError("input must be in D<=1")
    if u <= 0:
        raise ValueError("U must be positive")
    result = defaultdict(complex)
    result[state] += u * d
    virtual = []
    for mid, amp in kinetic_hops(state, h):
        d_mid = doublons(mid)
        if d_mid <= 1:
            result[mid] += amp
        elif d == 1 and d_mid == 2:
            virtual.append((mid, amp))
    for mid, amp in virtual:
        for final, second in kinetic_hops(mid, h):
            if doublons(final) == 1:
                result[final] -= second * amp / u
    return {key: value for key, value in result.items() if abs(value) > 1e-14}
