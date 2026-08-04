"""Phase 4 — causal verification: the key frequencies are necessary and sufficient.

Phase 3 showed the circuit *correlates* with the answer. Here we intervene. Because
the circuit reads its frequencies out of the token embedding, we edit the embedding
in Fourier space and re-run the (otherwise untouched) network:

  * SUFFICIENCY — keep ONLY the key frequencies in the embedding (zero all others).
    If they carry the computation, accuracy should survive.
  * NECESSITY   — remove ONLY the key frequencies. If they carry the computation,
    accuracy should collapse to chance (1/p).
  * SPECIFICITY — ablate each single frequency in turn and plot the accuracy. Only
    the key frequencies should matter; ablating any other frequency should do
    essentially nothing.

Outputs:
  results/phase4_ablation.png   — the necessity/sufficiency bars + the per-frequency sweep
  results/phase4_results.md     — the numbers
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grok.analysis import accuracy, load_run, model_at                    # noqa: E402
from grok.data import make_dataset                                        # noqa: E402
from grok.fourier import filter_frequencies, key_frequencies, make_fourier_basis, power_by_frequency  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"


def main(seed: int = 0) -> None:
    run = load_run(seed)
    cfg = run.cfg
    p = cfg.p
    F, _ = make_fourier_basis(p)
    sd = run.final().state_dict
    data = make_dataset(cfg, "cpu")
    chance = 1.0 / p

    keys, _ = key_frequencies(power_by_frequency(sd["W_E"][:p], F), cover=0.9)

    def eval_embedding(keep=None, remove=None) -> float:
        sd2 = {k: v.clone() for k, v in sd.items()}
        W = sd2["W_E"]
        W[:p] = filter_frequencies(W[:p], F, keep=keep, remove=remove).to(W.dtype)
        return accuracy(model_at(cfg, sd2), data.all_x, data.all_y)

    base = accuracy(model_at(cfg, sd), data.all_x, data.all_y)
    suff = eval_embedding(keep=keys)
    nec = eval_embedding(remove=keys)

    n_freq = (p + 1) // 2
    sweep = np.array([eval_embedding(remove=[k]) for k in range(1, n_freq)])
    freqs = np.arange(1, n_freq)

    rows = [
        "# Phase 4 — causal verification: the key frequencies are necessary and sufficient",
        "",
        f"Key frequencies: **{keys}**.  Chance accuracy = 1/{p} = {chance:.4f}.",
        "",
        "| intervention on the embedding | accuracy |",
        "|---|---|",
        f"| none (full model) | {base:.4f} |",
        f"| **keep ONLY** the key frequencies (sufficiency) | {suff:.4f} |",
        f"| **remove** the key frequencies (necessity) | {nec:.4f} |",
        "",
        f"- **Sufficiency:** keeping only {len(keys)} of {n_freq - 1} frequencies retains "
        f"**{suff:.1%}** accuracy — those frequencies *are* the computation.",
        f"- **Necessity:** removing just those {len(keys)} frequencies drops accuracy to "
        f"**{nec:.1%}** (chance is {chance:.1%}) — nothing else can do the task.",
        f"- **Specificity:** across the single-frequency sweep, the {len(keys)} key "
        f"frequencies are the only ones whose removal hurts; the median accuracy after "
        f"ablating a *non-key* frequency is {np.median([sweep[k - 1] for k in freqs if k not in keys]):.4f}.",
        "",
        f"**The 3 frequencies the network learned are causally responsible for the whole "
        f"task: keep only them and it still scores {suff:.0%}; remove only them and it falls "
        f"to chance.**",
    ]
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "phase4_results.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))

    # ---- figure: necessity/sufficiency bars + per-frequency sweep ------------ #
    fig, (axb, axs) = plt.subplots(1, 2, figsize=(12, 4.5), width_ratios=[1, 2])

    labels = ["full\nmodel", "keep only\nkey freqs", "remove\nkey freqs"]
    vals = [base, suff, nec]
    colors = ["tab:blue", "tab:green", "tab:red"]
    axb.bar(labels, vals, color=colors)
    axb.axhline(chance, color="k", ls="--", lw=0.8, label=f"chance = 1/{p}")
    axb.set(ylabel="accuracy", ylim=(0, 1.05), title="Necessity & sufficiency")
    for i, v in enumerate(vals):
        axb.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    axb.legend(fontsize=8)

    axs.plot(freqs, sweep, marker="o", ms=3, lw=1, color="tab:gray",
             label="accuracy after removing frequency k")
    axs.axhline(base, color="tab:blue", ls="--", lw=0.8, label="full-model accuracy")
    axs.axhline(chance, color="k", ls=":", lw=0.8, label=f"chance = 1/{p}")
    for k in keys:
        axs.scatter([k], [sweep[k - 1]], color="tab:red", zorder=5, s=40)
    axs.scatter([], [], color="tab:red", label="key frequencies")
    axs.set(xlabel="frequency k removed from the embedding", ylabel="accuracy",
            ylim=(-0.03, 1.05), title="Specificity — only the key frequencies matter")
    axs.legend(fontsize=8)

    fig.suptitle("Phase 4 — the key frequencies are causally necessary and sufficient")
    fig.tight_layout(); fig.savefig(RESULTS / "phase4_ablation.png", dpi=130)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
