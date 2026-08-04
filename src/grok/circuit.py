"""Reverse-engineering the grokked circuit: the "clock".

The claim (Nanda et al. 2023) is that the network's answer-determining signal is a
function of ``d = (a + b - c) mod p`` alone, and that this function is a sum of
cosines at the key frequencies:

    logit(a, b, c) ~ f(a + b - c),   f(d) = sum_k A_k cos(w_k d + phi_k)

which is maximised at ``d = 0`` — i.e. at ``c = a + b (mod p)`` — because every
cosine constructively interferes there. This module extracts ``f`` from a logit
tensor, fits it to the key-frequency cosines, and measures how far the pure
formula alone gets the answer right. All functions are model-agnostic (they take a
logit tensor), so they can be unit-tested on a synthetic clock.
"""
from __future__ import annotations

import math

import torch


def _d_index(p: int) -> torch.Tensor:
    """The tensor ``d[a, b, c] = (a + b - c) mod p``, shape (p, p, p)."""
    a = torch.arange(p).view(p, 1, 1)
    b = torch.arange(p).view(1, p, 1)
    c = torch.arange(p).view(1, 1, p)
    return (a + b - c) % p


def decision_function(L: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Extract ``f(d) = mean of the logit over all (a,b,c) with a+b-c = d``.

    Returns ``f`` (length p) and the share of the logit variance that this pure
    function of ``(a+b-c)`` explains (R^2 of ``L`` against ``f(a+b-c)``).
    """
    p = L.shape[0]
    d = _d_index(p).reshape(-1)
    y = L.reshape(-1).to(torch.float64)
    f = torch.zeros(p, dtype=torch.float64).index_add_(0, d, y)
    f = f / torch.bincount(d, minlength=p).to(torch.float64)
    yhat = f[d]
    r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return f, float(r2)


def fit_cosines(f: torch.Tensor, keys: list[int]) -> tuple[float, dict[int, float], torch.Tensor]:
    """Fit ``f(d) ~ sum_k [A_k cos + B_k sin](w_k d)`` by least squares.

    Returns R^2, the per-frequency amplitude ``sqrt(A^2+B^2)``, and the fitted
    curve (length p) for plotting.
    """
    p = f.numel()
    d = torch.arange(p, dtype=torch.float64)
    cols = [torch.ones(p, dtype=torch.float64)]
    for k in keys:
        w = 2 * math.pi * k / p
        cols += [torch.cos(w * d), torch.sin(w * d)]
    X = torch.stack(cols, dim=1)
    beta = torch.linalg.solve(X.T @ X, X.T @ f.to(torch.float64))
    fh = X @ beta
    r2 = float(1 - ((f - fh) ** 2).sum() / ((f - f.mean()) ** 2).sum())
    amps = {k: float(torch.hypot(beta[1 + 2 * i], beta[2 + 2 * i])) for i, k in enumerate(keys)}
    return r2, amps, fh


def accuracy_from_f(f: torch.Tensor) -> tuple[float, int]:
    """Accuracy of the classifier ``argmax_c f((a+b-c) mod p)`` against the true
    answer ``(a+b) mod p``. Also returns ``argmax_d f`` (the clock predicts 0)."""
    p = f.numel()
    rec = f[_d_index(p)]                                   # (p,p,p) = f(a+b-c)
    ans = (torch.arange(p)[:, None] + torch.arange(p)[None, :]) % p
    acc = float((rec.argmax(dim=-1) == ans).float().mean())
    return acc, int(f.argmax())


def frequency_power(L: torch.Tensor, F: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """2D Fourier power of the logits over the (a,b) grid.

    Returns the full ``(p, p)`` power map and the per-frequency power on the
    same-frequency diagonal blocks (length ``(p+1)//2``; index 0 is the constant).
    """
    p = L.shape[0]
    La = torch.einsum("ia,abc->ibc", F, L)
    Lab = torch.einsum("jb,ibc->ijc", F, La)
    power2d = (Lab ** 2).sum(dim=-1)
    n_freq = (p + 1) // 2
    freqpow = torch.zeros(n_freq, dtype=power2d.dtype)
    freqpow[0] = power2d[0, 0]
    for k in range(1, n_freq):
        r = [2 * k - 1, 2 * k]
        freqpow[k] = power2d[r][:, r].sum()
    return power2d, freqpow
