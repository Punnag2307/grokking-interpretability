"""Fourier analysis over Z_p — the lens that makes the grokking circuit legible.

The task lives on the cyclic group Z_p, so the natural basis for anything indexed
by a number token 0..p-1 is the discrete Fourier basis: a constant, plus
``cos(2*pi*k*n/p)`` and ``sin(2*pi*k*n/p)`` for each frequency ``k = 1..(p-1)/2``.
This basis is orthonormal (``F @ F.T = I``), so projecting a weight matrix onto it
redistributes — but never inflates — its total power. The grokking claim is that
after generalisation the token embedding is *sparse* in this basis: nearly all its
power sits on a handful of "key frequencies".
"""
from __future__ import annotations

import math

import torch


def make_fourier_basis(p: int) -> tuple[torch.Tensor, list[str]]:
    """Return the orthonormal Fourier basis ``F`` (p x p) and row labels.

    Row 0 is the constant; rows ``2k-1, 2k`` are ``cos_k, sin_k``.
    """
    n = torch.arange(p, dtype=torch.float64)
    F = torch.zeros(p, p, dtype=torch.float64)
    F[0] = 1.0 / math.sqrt(p)
    labels = ["const"]
    for k in range(1, p // 2 + 1):
        theta = 2 * math.pi * k * n / p
        F[2 * k - 1] = math.sqrt(2.0 / p) * torch.cos(theta)
        F[2 * k] = math.sqrt(2.0 / p) * torch.sin(theta)
        labels += [f"cos{k}", f"sin{k}"]
    return F, labels


def power_by_frequency(W: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """Power of matrix ``W`` (p x d) per frequency, summing cos/sin of each k.

    Returns a length-``(p+1)//2`` vector: index 0 is the constant, index ``k`` is
    the total power at frequency ``k``.
    """
    coeffs = F @ W.to(F.dtype)                 # (p, d)
    row_power = (coeffs ** 2).sum(dim=1)       # (p,)
    n_freq = (F.shape[0] - 1) // 2 + 1
    power = torch.empty(n_freq, dtype=F.dtype)
    power[0] = row_power[0]
    for k in range(1, n_freq):
        power[k] = row_power[2 * k - 1] + row_power[2 * k]
    return power


def gini(x: torch.Tensor) -> float:
    """Gini coefficient of a non-negative vector: 0 = uniform, →1 = concentrated.

    A scale-free sparsity measure; used as the Phase 2 headline number and,
    across the training trajectory, as a Phase 5 progress measure.
    """
    x = torch.sort(x.flatten())[0].to(torch.float64)
    n = x.numel()
    if x.sum() <= 0:
        return 0.0
    idx = torch.arange(1, n + 1, dtype=torch.float64)
    return float(((2 * idx - n - 1) * x).sum() / (n * x.sum()))


def filter_frequencies(
    W: torch.Tensor,
    F: torch.Tensor,
    keep: list[int] | None = None,
    remove: list[int] | None = None,
    keep_const: bool = True,
) -> torch.Tensor:
    """Project a ``(p, d)`` matrix onto / off a set of Fourier frequencies.

    With ``keep`` we zero every frequency *except* those listed (and the constant,
    if ``keep_const``); with ``remove`` we zero *only* the listed frequencies.
    Used for the Phase 4 causal ablations: keep-only-key-frequencies (sufficiency)
    and remove-key-frequencies (necessity). Returns a float64 tensor.
    """
    p = F.shape[0]
    coeffs = F @ W.to(F.dtype)                     # (p, d)
    if keep is not None:
        mask = torch.zeros(p, dtype=torch.bool)
        if keep_const:
            mask[0] = True
        for k in keep:
            mask[2 * k - 1] = True
            mask[2 * k] = True
        coeffs = coeffs * mask[:, None]
    if remove is not None:
        for k in remove:
            coeffs[2 * k - 1] = 0
            coeffs[2 * k] = 0
    return F.T @ coeffs                             # (p, d)


def key_frequencies(power: torch.Tensor, cover: float = 0.9) -> tuple[list[int], float]:
    """Frequencies (excluding the constant) that carry the embedding, sorted by
    power. Returns the minimal set whose cumulative share of *non-constant* power
    reaches ``cover``, plus the share those frequencies carry of the *total*.
    """
    freqs = power[1:]                          # drop constant
    order = torch.argsort(freqs, descending=True)
    total_nonconst = float(freqs.sum())
    if total_nonconst <= 0:
        return [], 0.0
    cum = 0.0
    keys: list[int] = []
    for i in order.tolist():
        keys.append(i + 1)                     # +1: frequencies are 1-indexed
        cum += float(freqs[i])
        if cum / total_nonconst >= cover:
            break
    share_of_total = float(power[keys].sum() / power.sum())
    return sorted(keys), share_of_total
