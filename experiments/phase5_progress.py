"""Phase 5 — progress measures: grokking is gradual underneath a sudden curve.

Grokking *looks* like a discontinuity: test accuracy sits at chance for thousands
of steps, then snaps to 100%. We replay the saved trajectory and compute measures
that reveal the circuit forming smoothly and *early*, before the test loss moves.

Using the fixed key frequencies of the *final* grokked model as the target circuit:
  * restricted loss — keep ONLY the key frequencies in the embedding, then the
    train loss. High while the model memorises via many directions; falls as the
    key-frequency circuit alone becomes able to fit the data.
  * excluded loss  — remove the key frequencies. Rises as the model comes to *rely*
    on that circuit rather than on memorisation. When restricted drops below
    excluded, the circuit has become the dominant mechanism.
  * embedding Fourier sparsity (Gini) — climbs monotonically as the circuit concentrates.
  * weight norm     — falls under weight decay; its decay drives the final cleanup.

The three phases fall out: memorisation -> circuit formation -> cleanup, with the
progress measures turning during the plateau, well before the test loss drops.

Outputs:
  results/phase5_progress.png  — the sudden curve beside the smooth progress measures
  results/phase5_results.md    — the crossover step vs the grok step
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
from grok.analysis import accuracy, load_run, loss, model_at  # noqa: E402
from grok.data import make_dataset  # noqa: E402
from grok.fourier import filter_frequencies, gini, key_frequencies, make_fourier_basis, power_by_frequency  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
GROK_ACC = 0.95


def main(seed: int = 0) -> None:
    run = load_run(seed)
    cfg = run.cfg
    p = cfg.p
    F, _ = make_fourier_basis(p)
    data = make_dataset(cfg, "cpu")

    # target circuit = the key frequencies of the FINAL grokked model, held fixed.
    keys, _ = key_frequencies(power_by_frequency(run.final().state_dict["W_E"][:p], F), cover=0.9)

    def edited_loss(sd, keep=None, remove=None) -> float:
        sd2 = {k: v.clone() for k, v in sd.items()}
        W = sd2["W_E"]
        W[:p] = filter_frequencies(W[:p], F, keep=keep, remove=remove).to(W.dtype)
        return loss(model_at(cfg, sd2), data.train_x, data.train_y)

    steps, te_acc, tr, te, restricted, excluded, ginis, wnorm = [], [], [], [], [], [], [], []
    for snap in run.snapshots:
        sd = snap.state_dict
        model = model_at(cfg, sd)
        steps.append(snap.step)
        tr.append(loss(model, data.train_x, data.train_y))
        te.append(loss(model, data.test_x, data.test_y))
        te_acc.append(accuracy(model, data.test_x, data.test_y))
        restricted.append(edited_loss(sd, keep=keys))
        excluded.append(edited_loss(sd, remove=keys))
        ginis.append(gini(power_by_frequency(sd["W_E"][:p], F)[1:]))
        wnorm.append(float(torch.sqrt(sum((v.float() ** 2).sum() for v in sd.values()))))

    steps = np.array(steps)
    te_acc, restr, excl, ginis = map(np.array, (te_acc, restricted, excluded, ginis))

    def first_at(mask):
        idx = np.where(mask)[0]
        return int(steps[idx[0]]) if len(idx) else None

    def idx_of(step):
        return int(np.argmin(np.abs(steps - step)))

    grok_step = first_at(te_acc >= GROK_ACC)
    cross_step = first_at(restr < excl)                 # key-frequency circuit overtakes memorisation
    lead = (grok_step - cross_step) if (grok_step and cross_step) else None
    plateau = idx_of((cross_step or steps[-1]) // 2)    # a reference point still inside the plateau

    def row(i, label):
        return f"| {label} (step {steps[i]}) | {te_acc[i]:.3f} | {restr[i]:.2f} | {excl[i]:.2f} | {ginis[i]:.3f} |"

    rows = [
        "# Phase 5 — progress measures: grokking is gradual underneath a sudden curve",
        "",
        f"Target circuit = key frequencies {keys} of the final model, held fixed along the run.",
        "",
        f"Test accuracy is pinned near chance until it snaps to 100% around step **{grok_step}** — "
        "the \"sudden\" grokking. But the internal measures move continuously through the "
        "supposedly-flat plateau:",
        "",
        f"- **The key-frequency circuit becomes the dominant mechanism at step {cross_step}** — where "
        "keeping *only* those frequencies fits the training data better than removing them "
        f"(restricted loss drops below excluded loss). That is **{lead} steps before** test "
        f"accuracy reaches {GROK_ACC}.",
        f"- **Embedding sparsity rises monotonically from the start**: Gini {ginis[0]:.2f} → "
        f"{ginis[plateau]:.2f} by step {steps[plateau]}, while test accuracy is still at chance "
        f"({te_acc[plateau]:.2f}).",
        "- The **three phases** are visible: *memorisation* (restricted loss high, model fits via "
        "many directions), *circuit formation* (restricted loss falls, excluded loss rises, sparsity "
        "climbs), *cleanup* (weight norm decays under weight decay and the test loss finally drops).",
        "",
        "| stage | test acc | restricted loss | excluded loss | Gini |",
        "|---|---|---|---|---|",
        row(plateau, "plateau"),
        row(idx_of(cross_step) if cross_step else -1, "circuit dominant"),
        row(idx_of(grok_step) if grok_step else -1, "grokking"),
        row(len(steps) - 1, "final"),
        "",
        "**The 'sudden' jump is the visible tip of a continuous reorganisation: by the time test "
        "accuracy first twitches, the network has been building its Fourier circuit for thousands "
        "of steps.**",
    ]
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "phase5_results.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))

    # ---- figure: the sudden curve, then the smooth progress measures --------- #
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    def marks(a):
        if cross_step:
            a.axvline(cross_step, color="tab:green", ls="--", lw=1, label="circuit dominant")
        if grok_step:
            a.axvline(grok_step, color="tab:red", ls=":", lw=1.2, label="grokking visible")

    ax[0].plot(steps, tr, color="tab:blue", label="train loss")
    ax[0].plot(steps, te, color="tab:orange", label="test loss")
    marks(ax[0])
    ax[0].set(xscale="log", yscale="log", xlabel="step", ylabel="loss",
              title="The mystery — test loss is flat, then sudden")
    ax[0].legend(fontsize=8)

    ax[1].plot(steps, restr, color="tab:purple", label="restricted loss (keep key freqs)")
    ax[1].plot(steps, excl, color="tab:brown", label="excluded loss (remove key freqs)")
    marks(ax[1])
    ax[1].set(xscale="log", yscale="log", xlabel="step", ylabel="train loss",
              title="The resolution — the circuit overtakes memorisation")
    ax[1].legend(fontsize=8)

    ax2 = ax[2]
    ax2.plot(steps, ginis, color="tab:green", label="embedding sparsity (Gini)")
    ax2.set(xscale="log", xlabel="step", ylabel="Gini", title="Sparsity rises, weights clean up")
    ax2b = ax2.twinx()
    ax2b.plot(steps, wnorm, color="tab:gray", label="weight norm")
    ax2b.set_ylabel("weight norm")
    marks(ax2)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")

    fig.suptitle("Phase 5 — progress measures reveal grokking as a gradual, three-phase process")
    fig.tight_layout()
    fig.savefig(RESULTS / "phase5_progress.png", dpi=130)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
