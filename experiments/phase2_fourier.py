"""Phase 2 — the Fourier lens on the token embedding.

Claim: the embedding becomes *sparse in the Fourier basis over Z_p only after the
model groks*. We take three snapshots of the seed-0 run — initialisation, a
memorised-but-not-grokked step, and the final grokked model — and for each, plot
the fraction of embedding power at each frequency and measure the sparsity (Gini).

Outputs:
  results/phase2_fourier.png    — power spectra at the three snapshots
  results/phase2_results.md     — Gini and the identified key frequencies
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grok.analysis import load_run                                   # noqa: E402
from grok.fourier import gini, key_frequencies, make_fourier_basis, power_by_frequency  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"


def main(seed: int = 0) -> None:
    run = load_run(seed)
    p = run.cfg.p
    F, _ = make_fourier_basis(p)

    picks = [("init", run.nearest(0)),
             ("memorised (pre-grok)", run.nearest(2000)),
             ("grokked (final)", run.final())]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    rows = ["# Phase 2 — the embedding is sparse in the Fourier basis (only after grokking)",
            "",
            "| snapshot | step | Gini(power) | #key freqs (90%) | key frequencies | share of total power |",
            "|---|---|---|---|---|---|"]
    freqs_axis = np.arange(1, (p + 1) // 2)   # 1..56

    for label, snap in picks:
        W_E = snap.state_dict["W_E"][:p]                    # number tokens only
        power = power_by_frequency(W_E, F)                  # length 57 (0=const)
        frac = (power / power.sum()).numpy()
        g = gini(power[1:])                                 # sparsity over real frequencies
        keys, share = key_frequencies(power, cover=0.9)
        ax.plot(freqs_axis, frac[1:], marker="o", ms=3, lw=1, label=f"{label} (step {snap.step})")
        rows.append(f"| {label} | {snap.step} | {g:.3f} | {len(keys)} | "
                    f"{keys} | {share:.3f} |")

    ax.set(xlabel="Fourier frequency k", ylabel="fraction of embedding power",
           title="Phase 2 — embedding power spectrum over Z_113")
    ax.legend()
    fig.tight_layout(); fig.savefig(RESULTS / "phase2_fourier.png", dpi=130)

    # headline: key frequencies of the grokked model
    W_E = run.final().state_dict["W_E"][:p]
    keys, share = key_frequencies(power_by_frequency(W_E, F), cover=0.9)
    rows += ["",
             f"**The grokked embedding concentrates {share:.1%} of its total power on "
             f"{len(keys)} key frequencies: {keys}.** At initialisation the same power is "
             f"spread diffusely across all {(p - 1) // 2} frequencies (low Gini)."]
    (RESULTS / "phase2_results.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
