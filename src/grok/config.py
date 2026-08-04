"""Experiment configuration.

Defaults reproduce the canonical grokking setup (Power et al. 2022; Nanda et al.
2023): a one-layer transformer on ``(a + b) mod p`` for prime ``p = 113``,
trained *full-batch* with AdamW and strong weight decay on a fixed 30% of the
p^2 pairs. Weight decay is not optional dressing — it is the force that drives
the late "cleanup" phase, so it is a first-class config field, swept in Phase 6.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # --- task -------------------------------------------------------------- #
    p: int = 113                      # prime modulus; vocabulary of numbers is 0..p-1
    op: str = "add"                   # one of {"add", "sub", "mul"} (Phase 6 sweeps this)
    train_frac: float = 0.3           # fraction of the p^2 pairs used for training
    data_seed: int = 0                # seed for the train/test split (fixed across model seeds)

    # --- model ------------------------------------------------------------- #
    d_model: int = 128
    n_heads: int = 4
    d_head: int = 32
    d_mlp: int = 512
    act: str = "relu"                 # MLP nonlinearity: {"relu", "gelu"}
    use_ln: bool = False              # LayerNorm off by default — a cleaner circuit to read (ADR-0003)

    # --- optimisation ------------------------------------------------------ #
    lr: float = 1e-3
    weight_decay: float = 1.0
    betas: tuple[float, float] = (0.9, 0.98)
    epochs: int = 40_000              # full-batch steps
    seed: int = 0                     # model-init + training seed

    # --- logging / checkpointing ------------------------------------------ #
    eval_every: int = 100             # record train/test metrics every N steps
    n_checkpoints: int = 200          # log-spaced weight snapshots kept for trajectory analysis
    device: str = "cuda"              # falls back to cpu in train() if cuda is unavailable

    # derived, not set by the user
    _ops: tuple[str, ...] = field(default=("add", "sub", "mul"), repr=False)

    @property
    def d_vocab_in(self) -> int:
        """Numbers 0..p-1 plus a single '=' token at index p."""
        return self.p + 1

    @property
    def d_vocab_out(self) -> int:
        """The answer is always a number in 0..p-1."""
        return self.p

    @property
    def n_ctx(self) -> int:
        """Sequence is [a, b, '='] — we read the prediction off the '=' position."""
        return 3

    def __post_init__(self) -> None:
        if self.op not in self._ops:
            raise ValueError(f"op must be one of {self._ops}, got {self.op!r}")
        if not (0.0 < self.train_frac < 1.0):
            raise ValueError(f"train_frac must be in (0, 1), got {self.train_frac}")
        if self.d_model != self.n_heads * self.d_head:
            raise ValueError(
                f"d_model ({self.d_model}) must equal n_heads*d_head "
                f"({self.n_heads}*{self.d_head}={self.n_heads * self.d_head})"
            )
