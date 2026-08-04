"""Full-batch training loop that reproduces grokking and records its trajectory.

The dataset is tiny (<= p^2 examples), so we train *full-batch*: every step is a
gradient step on the entire training set. Grokking is a story about a long
trajectory, so alongside the usual train/test metrics we keep a set of
log-spaced weight snapshots — the raw material the interpretability phases
(Fourier lens, circuit, progress measures) replay to see *when* the circuit
forms relative to when the test loss finally drops.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .data import Dataset, make_dataset
from .model import Transformer
from .seed import set_seed


@dataclass
class History:
    steps: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    test_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    test_acc: list[float] = field(default_factory=list)

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {k: np.asarray(v) for k, v in self.__dict__.items()}


@dataclass
class Checkpoint:
    step: int
    state_dict: dict[str, torch.Tensor]   # kept on CPU to spare VRAM


@dataclass
class Result:
    model: Transformer
    history: History
    checkpoints: list[Checkpoint]
    data: Dataset
    cfg: Config


def _metrics(model: Transformer, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    with torch.no_grad():
        logits = model(x)
        loss = F.cross_entropy(logits, y).item()
        acc = (logits.argmax(-1) == y).float().mean().item()
    return loss, acc


def _checkpoint_steps(epochs: int, n: int) -> set[int]:
    """Log-spaced step indices (plus the first and last) to snapshot weights."""
    if n >= epochs:
        return set(range(epochs))
    pts = np.unique(np.geomspace(1, epochs - 1, num=n).round().astype(int))
    return set(int(s) for s in pts) | {0, epochs - 1}


def train(cfg: Config, verbose: bool = True) -> Result:
    device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
    set_seed(cfg.seed)

    data = make_dataset(cfg, device=device)
    model = Transformer(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas
    )

    hist = History()
    ckpts: list[Checkpoint] = []
    ckpt_at = _checkpoint_steps(cfg.epochs, cfg.n_checkpoints)

    for step in range(cfg.epochs):
        model.train()
        logits = model(data.train_x)
        loss = F.cross_entropy(logits, data.train_y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step in ckpt_at:
            ckpts.append(Checkpoint(step, {k: v.detach().cpu().clone()
                                           for k, v in model.state_dict().items()}))
        if step % cfg.eval_every == 0 or step == cfg.epochs - 1:
            model.eval()
            tr_l, tr_a = _metrics(model, data.train_x, data.train_y)
            te_l, te_a = _metrics(model, data.test_x, data.test_y)
            hist.steps.append(step)
            hist.train_loss.append(tr_l); hist.test_loss.append(te_l)
            hist.train_acc.append(tr_a); hist.test_acc.append(te_a)
            if verbose and (step % (cfg.eval_every * 20) == 0 or step == cfg.epochs - 1):
                print(f"step {step:6d} | train acc {tr_a:.3f} loss {tr_l:.4f} "
                      f"| test acc {te_a:.3f} loss {te_l:.4f}")

    return Result(model=model, history=hist, checkpoints=ckpts, data=data, cfg=cfg)
