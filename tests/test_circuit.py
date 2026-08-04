"""Tests for the clock-circuit analysis, on a *synthetic* clock with known answer.

If we hand the analysis a logit tensor that is exactly sum_k cos(w_k(a+b-c)), it
must recover: f is that sum (R^2 = 1), unit amplitudes, a peak at d = 0, 100%
accuracy, and frequency power on exactly the planted frequencies. This pins down
that Phase 3's *method* is correct, independent of any trained model.
"""
from __future__ import annotations

import math

import torch

from grok.circuit import accuracy_from_f, decision_function, fit_cosines, frequency_power
from grok.fourier import key_frequencies, make_fourier_basis


def _synthetic_clock(p: int, keys: list[int]) -> torch.Tensor:
    d = torch.arange(p, dtype=torch.float64)
    fvec = sum(torch.cos(2 * math.pi * k * d / p) for k in keys)   # f_true(d)
    a = torch.arange(p).view(p, 1, 1)
    b = torch.arange(p).view(1, p, 1)
    c = torch.arange(p).view(1, 1, p)
    return fvec[(a + b - c) % p]                                    # L[a,b,c] = f_true(a+b-c)


def test_recovers_synthetic_clock():
    p, keys = 23, [1, 3, 7]
    L = _synthetic_clock(p, keys)

    f, share = decision_function(L)
    assert share > 0.999                       # L is exactly a function of (a+b-c)

    r2, amps, _ = fit_cosines(f, keys)
    assert r2 > 0.9999                          # and that function is exactly the cosines
    for k in keys:
        assert abs(amps[k] - 1.0) < 1e-6        # planted unit amplitudes recovered

    acc, argmax_d = accuracy_from_f(f)
    assert acc == 1.0 and argmax_d == 0         # peaks at d=0 -> c=a+b, classifies perfectly


def test_frequency_power_finds_planted_frequencies():
    p, keys = 23, [1, 3, 7]
    L = _synthetic_clock(p, keys)
    F, _ = make_fourier_basis(p)
    _, freqpow = frequency_power(L, F)
    found, _ = key_frequencies(freqpow, cover=0.9)
    assert set(found) <= set(keys) and len(found) >= 1
    # essentially all diagonal power sits on the planted frequencies
    assert float(freqpow[keys].sum() / freqpow[1:].sum()) > 0.999


def test_non_clock_is_not_explained():
    # A random logit tensor is not a function of (a+b-c): low share, chance accuracy.
    p = 23
    torch.manual_seed(0)
    L = torch.randn(p, p, p, dtype=torch.float64)
    L = L - L.mean(dim=-1, keepdim=True)
    f, share = decision_function(L)
    assert share < 0.2
    acc, _ = accuracy_from_f(f)
    assert acc < 0.2
