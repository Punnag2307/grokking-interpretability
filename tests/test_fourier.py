"""Correctness tests for the Fourier tooling the interpretability rests on."""
from __future__ import annotations

import torch

from grok.fourier import (
    filter_frequencies,
    gini,
    key_frequencies,
    make_fourier_basis,
    power_by_frequency,
)


def test_basis_is_orthonormal():
    for p in (7, 13, 113):
        F, labels = make_fourier_basis(p)
        assert F.shape == (p, p)
        assert len(labels) == p
        eye = F @ F.T
        assert torch.allclose(eye, torch.eye(p, dtype=eye.dtype), atol=1e-10)


def test_parseval_power_conserved():
    # An orthonormal transform preserves total power (Parseval): the per-frequency
    # power must sum to the Frobenius norm^2 of the original matrix.
    p, d = 13, 8
    torch.manual_seed(0)
    W = torch.randn(p, d, dtype=torch.float64)
    F, _ = make_fourier_basis(p)
    power = power_by_frequency(W, F)
    assert torch.isclose(power.sum(), (W ** 2).sum(), atol=1e-9)


def test_gini_bounds():
    n = 100
    assert abs(gini(torch.ones(n))) < 1e-9                 # uniform -> 0
    onehot = torch.zeros(n); onehot[0] = 1.0
    assert gini(onehot) > 0.98                             # concentrated -> ~1


def test_filter_frequencies_keep_and_remove():
    p, d = 23, 4
    F, _ = make_fourier_basis(p)
    torch.manual_seed(0)
    coeffs = torch.zeros(p, p, dtype=torch.float64)[:, :d]
    for k in (3, 7):                                    # build a matrix living only at freqs {3,7}
        coeffs[2 * k - 1] = torch.randn(d, dtype=torch.float64)
        coeffs[2 * k] = torch.randn(d, dtype=torch.float64)
    W = F.T @ coeffs
    # keeping its own frequencies is the identity
    assert torch.allclose(filter_frequencies(W, F, keep=[3, 7], keep_const=False), W, atol=1e-9)
    # removing its only frequencies zeroes it
    assert filter_frequencies(W, F, remove=[3, 7]).abs().max() < 1e-9
    # removing an unrelated frequency leaves it unchanged
    assert torch.allclose(filter_frequencies(W, F, remove=[5]), W, atol=1e-9)


def test_key_frequencies_picks_the_spike():
    p = 113
    power = torch.full(((p + 1) // 2,), 1e-6, dtype=torch.float64)
    power[0] = 5.0            # constant, ignored
    for k in (7, 41, 55):     # three planted key frequencies (all <= (p-1)/2 = 56)
        power[k] = 1.0
    keys, share = key_frequencies(power, cover=0.9)
    assert set(keys) == {7, 41, 55}
    assert share > 0.0
