import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from d01_operator import doublons, heff01_connections, kinetic_hops


def check(nsite, nup, ndown):
    bits_up = [sum(1 << i for i in occupied) for occupied in itertools.combinations(range(nsite), nup)]
    bits_down = [sum(1 << i for i in occupied) for occupied in itertools.combinations(range(nsite), ndown)]
    states = [(a, b) for a in bits_up for b in bits_down]
    lookup = {state: i for i, state in enumerate(states)}
    h = np.zeros((nsite, nsite))
    for i in range(nsite - 1):
        h[i, i + 1] = h[i + 1, i] = -1.0
    k = np.zeros((len(states), len(states)))
    for j, state in enumerate(states):
        for next_state, amp in kinetic_hops(state, h):
            k[lookup[next_state], j] += amp
    d = np.array([doublons(state) for state in states])
    p = np.flatnonzero(d <= 1)
    p1 = np.flatnonzero(d == 1)
    p2 = np.flatnonzero(d == 2)
    full = k + np.diag(8.0 * d)
    expected = full[np.ix_(p, p)].copy()
    local = np.searchsorted(p, p1)
    expected[np.ix_(local, local)] -= k[np.ix_(p1, p2)] @ k[np.ix_(p2, p1)] / 8.0
    actual = np.zeros_like(expected, dtype=complex)
    lookup_p = {states[i]: j for j, i in enumerate(p)}
    for j, i in enumerate(p):
        for target, amp in heff01_connections(states[i], h, 8.0).items():
            actual[lookup_p[target], j] += amp
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(actual, actual.conj().T, atol=1e-12, rtol=0)
    if not len(p2):
        np.testing.assert_allclose(actual, full, atol=1e-12, rtol=0)


class TestD01Operator(unittest.TestCase):
    def test_no_q(self):
        check(2, 1, 1)

    def test_with_d2(self):
        check(4, 2, 2)


if __name__ == "__main__":
    unittest.main()
