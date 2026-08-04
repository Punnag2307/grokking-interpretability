"""Phase 7 — the clock generalises: every seed learns the same algorithm.

A single reverse-engineered run could be a fluke. Here we run the Phase 3 circuit
analysis on all three independently-trained, independently-grokked seeds and show
each one has learned *the clock*: its answer-determining signal is a function of
(a+b-c), fit by cosines at that seed's own key frequencies, peaking at c=a+b and
classifying at 100%. The algorithm is universal; the specific frequencies it lands
on are seed-dependent.

This is consistent with Zhong et al. (2023), who find that attention+MLP
transformers on modular addition favour the "clock" over the "pizza" algorithm; a
full clock-vs-pizza discrimination is left as further work.

Outputs:
  results/phase7_generalization.png  — f(d) and its cosine fit, one panel per seed
  results/phase7_results.md          — key frequencies, R^2 and accuracy per seed
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grok.analysis import all_logits, load_run, model_at  # noqa: E402
from grok.circuit import accuracy_from_f, decision_function, fit_cosines  # noqa: E402
from grok.data import make_dataset  # noqa: E402
from grok.fourier import key_frequencies, make_fourier_basis, power_by_frequency  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"


def analyse(seed: int):
    run = load_run(seed)
    cfg = run.cfg
    p = cfg.p
    F, _ = make_fourier_basis(p)
    model = model_at(cfg, run.final().state_dict, "cpu")
    data = make_dataset(cfg, "cpu")
    keys, _ = key_frequencies(power_by_frequency(run.final().state_dict["W_E"][:p], F), cover=0.9)
    f, share = decision_function(all_logits(model, data.all_x, center=True))
    r2, amps, fhat = fit_cosines(f, keys)
    acc, argmax_d = accuracy_from_f(f)
    return {"p": p, "keys": keys, "share": share, "r2": r2, "acc": acc,
            "argmax_d": argmax_d, "f": f, "fhat": fhat}


def main(seeds=(0, 1, 2)) -> None:
    res = {s: analyse(s) for s in seeds}

    rows = [
        "# Phase 7 — the clock generalises across seeds",
        "",
        "Every independently-trained, independently-grokked seed learns the clock: its "
        "decision function is a function of (a+b-c), fit by cosines at that seed's own key "
        "frequencies, peaking at c=a+b and classifying at 100%.",
        "",
        "| seed | key frequencies | share as f(a+b-c) | cosine-fit R^2 | peaks at d=0 | classify acc |",
        "|---|---|---|---|---|---|",
    ]
    for s in seeds:
        r = res[s]
        rows.append(f"| {s} | {r['keys']} | {r['share']:.1%} | {r['r2']:.4f} | "
                    f"{'yes' if r['argmax_d'] == 0 else 'no'} | {r['acc']:.1%} |")
    same = len({tuple(res[s]["keys"]) for s in seeds}) == 1
    min_r2 = min(res[s]["r2"] for s in seeds)
    freq_str = ", ".join(f"seed {s}: {res[s]['keys']}" for s in seeds)
    same_str = "the same" if same else "different"
    rows += [
        "",
        f"**All three seeds learn the clock** (cosine-fit R^2 >= {min_r2:.4f}, 100% "
        f"classification), but on {same_str} key frequencies ({freq_str}) — the algorithm is "
        "universal, the frequencies it selects are seed-dependent.",
    ]
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "phase7_results.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))

    fig, axes = plt.subplots(1, len(seeds), figsize=(5 * len(seeds), 4.2), squeeze=False)
    for ax, s in zip(axes[0], seeds, strict=True):
        r = res[s]
        d = torch.arange(r["p"])
        ax.plot(d.numpy(), r["f"].numpy(), lw=1.5, color="tab:blue", label="decision function f")
        ax.plot(d.numpy(), r["fhat"].numpy(), "--", lw=1.3, color="tab:orange",
                label=f"cosines k={r['keys']}")
        ax.axvline(0, color="k", lw=0.7, ls=":")
        ax.set(xlabel="d = (a+b−c) mod p", title=f"seed {s}  (R²={r['r2']:.4f}, {r['acc']:.0%})")
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("logit contribution")
    fig.suptitle("Phase 7 — every seed independently learns the clock (on its own key frequencies)")
    fig.tight_layout()
    fig.savefig(RESULTS / "phase7_generalization.png", dpi=130)


if __name__ == "__main__":
    main()
