"""Load a saved training run and rebuild the model at any checkpointed step.

Phases 2-5 all replay the trajectory that Phase 1 saved to ``runs/``: the config
plus a list of log-spaced weight snapshots. This module reconstructs the
``Config`` and materialises a ``Transformer`` from any snapshot's state dict.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .config import Config
from .model import Transformer

REPO = Path(__file__).resolve().parents[2]


@dataclass
class Snapshot:
    step: int
    state_dict: dict[str, torch.Tensor]


@dataclass
class Run:
    cfg: Config
    snapshots: list[Snapshot]

    @property
    def steps(self) -> list[int]:
        return [s.step for s in self.snapshots]

    def nearest(self, step: int) -> Snapshot:
        """The snapshot whose step is closest to ``step``."""
        return min(self.snapshots, key=lambda s: abs(s.step - step))

    def final(self) -> Snapshot:
        return self.snapshots[-1]


def load_run(seed: int, repo: Path = REPO) -> Run:
    blob = torch.load(repo / "runs" / f"phase1_seed{seed}.pt", map_location="cpu",
                      weights_only=False)
    fields = Config.__dataclass_fields__
    cfg = Config(**{k: v for k, v in blob["cfg"].items() if k in fields})
    snaps = [Snapshot(c["step"], c["state_dict"]) for c in blob["checkpoints"]]
    snaps.sort(key=lambda s: s.step)
    return Run(cfg=cfg, snapshots=snaps)


def model_at(cfg: Config, state_dict: dict[str, torch.Tensor],
             device: str | torch.device = "cpu") -> Transformer:
    model = Transformer(cfg).to(device)
    model.load_state_dict({k: v.to(device) for k, v in state_dict.items()})
    model.eval()
    return model


def all_logits(model: Transformer, all_x: torch.Tensor, center: bool = True) -> torch.Tensor:
    """Logits on every (a, b) input, shaped ``(p, p, p)`` as ``L[a, b, c]``.

    ``all_x`` is the canonical-order input grid (``Dataset.all_x``). When
    ``center`` is set we subtract the per-input mean over the answer axis, since
    the softmax — and therefore the prediction — is invariant to a constant added
    to a logit vector; the centred logits are the part the circuit must explain.
    """
    p = model.cfg.p
    with torch.no_grad():
        logits = model(all_x)                     # (p*p, p)
    L = logits.reshape(p, p, p).to(torch.float64)
    if center:
        L = L - L.mean(dim=-1, keepdim=True)
    return L
