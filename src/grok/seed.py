"""Global seeding and determinism.

Grokking is a claim about a *trajectory* (memorise, then generalise tens of
thousands of steps later), so a run must be bit-reproducible from its seed for
the analysis phases to line up with the training curve.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and Torch (CPU + CUDA) from a single integer.

    With ``deterministic=True`` we also ask cuBLAS and Torch for deterministic
    kernels. This is best-effort: a few ops have no deterministic CUDA kernel,
    so we fall back to ``warn_only`` rather than hard-failing a run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Required for deterministic cuBLAS matmuls on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
