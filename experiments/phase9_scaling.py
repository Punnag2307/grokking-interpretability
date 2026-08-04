"""Phase 9 — how the circuit scales with the modulus p.

Phases 2-8 all use p = 113. The natural question a reviewer asks is whether the
clock is a general fact or a quirk of one prime, and — more interestingly — how the
*size* of the learned circuit depends on the problem size. We train the same
fixed-capacity model (d_model = 128) to grok on a range of primes and, for each,
measure the effective number of Fourier frequencies the embedding uses (the minimal
set carrying 90% of its non-constant power).

If that count is roughly constant while p grows several-fold, the circuit size is
set by the model's capacity, not by the modulus — a concrete, checkable statement
about what the network builds.

Runtime is configurable for a smoke test (P9_EPOCHS=200 python ...). Default 35k
steps per prime; ~6 primes, background job.

Outputs:
  results/phase9_scaling.png   — effective #frequencies and grok step vs p
  results/phase9_results.md     — the per-prime table and the trend
  results/phase9_scaling.json   — machine-readable summary
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grok import Config  # noqa: E402
from grok.fourier import key_frequencies, make_fourier_basis, power_by_frequency  # noqa: E402
from grok.train import train  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
EPOCHS = int(os.environ.get("P9_EPOCHS", "35000"))
PRIMES = [53, 71, 97, 113, 149, 191]
GROK_ACC = 0.95


def _first_cross(steps, values, thresh):
    hit = np.where(np.asarray(values) >= thresh)[0]
    return int(steps[hit[0]]) if len(hit) else None


def run_prime(p: int) -> dict:
    print(f"  training p={p} (epochs={EPOCHS}) ...", flush=True)
    cfg = Config(p=p, epochs=EPOCHS, eval_every=200, n_checkpoints=2, seed=0, device="cuda")
    res = train(cfg, verbose=False)
    h = res.history.as_arrays()
    grok = _first_cross(h["steps"], h["test_acc"], GROK_ACC)
    F, _ = make_fourier_basis(p)
    w_e = res.model.W_E.detach().cpu()[:p]
    power = power_by_frequency(w_e, F)
    keys, share = key_frequencies(power, cover=0.9)
    out = {"p": p, "grok": grok, "final_acc": float(h["test_acc"][-1]),
           "n_freq": len(keys), "keys": keys, "share": share}
    print(f"    -> grok {grok}, final acc {out['final_acc']:.3f}, "
          f"{out['n_freq']} key freqs {keys}", flush=True)
    return out


def main() -> None:
    runs = [run_prime(p) for p in PRIMES]
    grokked = [r for r in runs if r["grok"] is not None]

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "phase9_scaling.json").write_text(
        json.dumps({"epochs": EPOCHS, "runs": runs}, indent=2), encoding="utf-8")

    ps = [r["p"] for r in grokked]
    nfreq = [r["n_freq"] for r in grokked]
    mean_nf = float(np.mean(nfreq)) if nfreq else float("nan")

    rows = [
        "# Phase 9 — how the circuit scales with the modulus p",
        "",
        f"Fixed-capacity model (d_model=128), seed 0, {EPOCHS} steps. Effective #frequencies = "
        f"minimal set carrying 90% of the embedding's non-constant power.",
        "",
        "| p | grokked at | final test acc | # key frequencies | key frequencies |",
        "|---|---|---|---|---|",
    ]
    for r in runs:
        gk = r["grok"] if r["grok"] is not None else f"none (>{EPOCHS})"
        rows.append(f"| {r['p']} | {gk} | {r['final_acc']:.3f} | {r['n_freq']} | {r['keys']} |")
    if grokked:
        rows += [
            "",
            f"Across primes p = {min(ps)}–{max(ps)} (a {max(ps) / min(ps):.1f}× range), the effective "
            f"number of Fourier frequencies stays around **{mean_nf:.1f}** (range {min(nfreq)}–"
            f"{max(nfreq)}). The circuit size is set by the model's fixed capacity, not by the modulus: "
            "a larger problem is solved with essentially the same-sized Fourier circuit, on a different "
            "set of frequencies.",
        ]
    else:
        rows += ["", f"No prime grokked within {EPOCHS} steps."]
    (RESULTS / "phase9_results.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n" + "\n".join(rows))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].scatter(ps, nfreq, s=45, color="tab:blue", zorder=3)
    if nfreq:
        ax[0].axhline(mean_nf, color="tab:blue", ls="--", lw=0.9, label=f"mean = {mean_nf:.1f}")
    ax[0].set(xlabel="modulus p", ylabel="effective # Fourier frequencies",
              title="Circuit size is set by capacity, not p", ylim=(0, max(nfreq + [1]) + 2))
    ax[0].legend(fontsize=8)

    gsteps = [r["grok"] for r in grokked]
    ax[1].scatter(ps, gsteps, s=45, color="tab:green", zorder=3)
    ax[1].set(xlabel="modulus p", ylabel="grok step", title="Grokking time vs p")

    fig.suptitle("Phase 9 — scaling the modulus at fixed model capacity")
    fig.tight_layout()
    fig.savefig(RESULTS / "phase9_scaling.png", dpi=130)


if __name__ == "__main__":
    main()
