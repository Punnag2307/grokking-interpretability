"""The modular-arithmetic task.

Every ordered pair ``(a, b)`` with ``0 <= a, b < p`` is one example. The input
sequence is the three tokens ``[a, b, '=']`` (the '=' token has id ``p``), and
the label is ``(a op b) mod p``. The full dataset is the p^2 pairs; a fixed
random split (seeded independently of the model) puts ``train_frac`` of them in
train and the rest in test. Generalisation therefore means inferring the group
operation on pairs never seen during training.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import Config


@dataclass
class Dataset:
    train_x: torch.Tensor   # (n_train, 3) long
    train_y: torch.Tensor   # (n_train,)   long
    test_x: torch.Tensor    # (n_test, 3)  long
    test_y: torch.Tensor    # (n_test,)    long
    all_x: torch.Tensor     # (p*p, 3)     long  — canonical order, for analysis
    all_y: torch.Tensor     # (p*p,)       long


def _apply_op(a: torch.Tensor, b: torch.Tensor, op: str, p: int) -> torch.Tensor:
    if op == "add":
        return (a + b) % p
    if op == "sub":
        return (a - b) % p
    if op == "mul":
        return (a * b) % p
    raise ValueError(f"unknown op {op!r}")


def make_dataset(cfg: Config, device: str | torch.device = "cpu") -> Dataset:
    """Build the full task and a fixed, seeded train/test split.

    The split uses its own generator (``cfg.data_seed``) so that changing the
    model seed re-initialises the network but keeps the *same* train/test
    partition — essential for comparing runs.
    """
    p = cfg.p
    a = torch.arange(p).repeat_interleave(p)          # (p*p,)
    b = torch.arange(p).repeat(p)                     # (p*p,)
    eq = torch.full((p * p,), p, dtype=torch.long)    # the '=' token
    all_x = torch.stack([a, b, eq], dim=1).long()     # (p*p, 3)
    all_y = _apply_op(a, b, cfg.op, p).long()         # (p*p,)

    g = torch.Generator().manual_seed(cfg.data_seed)
    perm = torch.randperm(p * p, generator=g)
    n_train = int(round(cfg.train_frac * p * p))
    tr, te = perm[:n_train], perm[n_train:]

    dev = torch.device(device)
    return Dataset(
        train_x=all_x[tr].to(dev), train_y=all_y[tr].to(dev),
        test_x=all_x[te].to(dev), test_y=all_y[te].to(dev),
        all_x=all_x.to(dev), all_y=all_y.to(dev),
    )
