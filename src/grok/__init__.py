"""grok — mechanistic interpretability of grokking on modular arithmetic.

A from-scratch, fully hookable one-layer transformer (`model.Transformer`), the
modular-arithmetic task (`data`), and a full-batch training loop (`train`) that
reproduces the delayed-generalisation ("grokking") phenomenon and checkpoints its
whole trajectory for later reverse-engineering.
"""
from __future__ import annotations

from .config import Config
from .data import make_dataset
from .model import Transformer
from .seed import set_seed

__all__ = ["Config", "Transformer", "make_dataset", "set_seed"]
