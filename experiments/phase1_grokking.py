"""Phase 1 — reproduce grokking.

Trains the canonical config to convergence for one or more seeds, saving:
  * results/phase1_history_seed{N}.npz  — the train/test loss & accuracy curves
  * runs/phase1_seed{N}.pt               — log-spaced weight snapshots (for Phases 2-5)
and, from whatever seeds are present, regenerates:
  * results/phase1_grokking.png          — the memorise-then-generalise curve
  * results/phase1_results.md            — memorisation step, grok step, and the delay

Usage:
    python experiments/phase1_grokking.py 0 1 2      # seeds to run (default: 0)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grok import Config           # noqa: E402
from grok.train import train      # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
RUNS = REPO / "runs"
GROK_ACC = 0.95   # test accuracy that counts as "generalised"
MEM_ACC = 0.99    # train accuracy that counts as "memorised"


def run_seed(seed: int) -> None:
    cfg = Config(seed=seed, device="cuda")
    print(f"\n=== Phase 1: training seed {seed} ({cfg.epochs} steps) ===")
    res = train(cfg, verbose=True)
    h = res.history.as_arrays()
    RESULTS.mkdir(exist_ok=True)
    RUNS.mkdir(exist_ok=True)
    np.savez(RESULTS / f"phase1_history_seed{seed}.npz", **h)
    torch.save(
        {"cfg": cfg.__dict__,
         "checkpoints": [{"step": c.step, "state_dict": c.state_dict} for c in res.checkpoints]},
        RUNS / f"phase1_seed{seed}.pt",
    )
    print(f"saved history + {len(res.checkpoints)} checkpoints for seed {seed}")


def _first_cross(steps: np.ndarray, values: np.ndarray, thresh: float) -> int | None:
    hit = np.where(values >= thresh)[0]
    return int(steps[hit[0]]) if len(hit) else None


def summarise_and_plot() -> None:
    files = sorted(RESULTS.glob("phase1_history_seed*.npz"))
    if not files:
        return
    hs = {int(f.stem.split("seed")[1]): dict(np.load(f)) for f in files}
    seeds = sorted(hs)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for s in seeds:
        h = hs[s]
        ax[0].plot(h["steps"], h["train_acc"], color="tab:blue", alpha=0.6)
        ax[0].plot(h["steps"], h["test_acc"], color="tab:orange", alpha=0.6)
        ax[1].plot(h["steps"], h["train_loss"], color="tab:blue", alpha=0.6)
        ax[1].plot(h["steps"], h["test_loss"], color="tab:orange", alpha=0.6)
    ax[0].plot([], [], color="tab:blue", label="train"); ax[0].plot([], [], color="tab:orange", label="test")
    ax[0].set(xscale="log", xlabel="step", ylabel="accuracy", title="Grokking — accuracy")
    ax[0].legend()
    ax[1].plot([], [], color="tab:blue", label="train"); ax[1].plot([], [], color="tab:orange", label="test")
    ax[1].set(xscale="log", yscale="log", xlabel="step", ylabel="loss", title="Grokking — loss")
    ax[1].legend()
    fig.suptitle(f"Phase 1 — delayed generalisation on (a+b) mod 113  ({len(seeds)} seed(s))")
    fig.tight_layout(); fig.savefig(RESULTS / "phase1_grokking.png", dpi=130)

    lines = ["# Phase 1 — grokking reproduced", "",
             "| seed | memorised (train>0.99) | grokked (test>0.95) | delay (steps) | final test acc |",
             "|---|---|---|---|---|"]
    delays = []
    for s in seeds:
        h = hs[s]
        mem = _first_cross(h["steps"], h["train_acc"], MEM_ACC)
        grok = _first_cross(h["steps"], h["test_acc"], GROK_ACC)
        delay = (grok - mem) if (mem is not None and grok is not None) else None
        if delay is not None:
            delays.append(delay)
        lines.append(f"| {s} | {mem} | {grok} | {delay} | {h['test_acc'][-1]:.4f} |")
    if delays:
        lines += ["", f"**Mean grokking delay: {int(np.mean(delays))} steps** "
                      f"(memorisation to generalisation) across {len(delays)} seed(s)."]
    (RESULTS / "phase1_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:]] or [0]
    for s in seeds:
        run_seed(s)
    summarise_and_plot()
