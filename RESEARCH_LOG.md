# Research log

A dated, first-person account of how the project actually went — including the
dead ends and the corrections, which are the parts worth remembering. Engineering
choices are in [DECISIONS.md](DECISIONS.md); this is the narrative.

## 2026-08-04 — Phase 0: scaffolding

Set up the repo the way its sibling project (DeepBSDE) taught: pinned deps, a
seeded config, a test suite and CI from the first commit. The one non-obvious call
was to write the transformer **from scratch** with named weight matrices and an
activation `cache`, rather than reach for a library. The interpretability phases
were going to need to read individual weights and activations, and a hand-written
model makes each of those a named tensor. LayerNorm off by default, float32.

Smoke test after the loop was wired: at step ~500 the model already sat at 100%
train accuracy with test accuracy at chance and test *loss* exploding to ~20 — the
memorisation plateau, exactly the precondition for grokking. Good sign.

## 2026-08-04 — Phase 1: grokking reproduced, first try

The canonical config (p=113, full-batch AdamW, weight decay 1.0, 30% train) grokked
on the first run: memorised by step 200, generalised at step ~4,800 — a 4,600-step
delay, with test loss spiking to 24 before collapsing to zero. Ran three seeds; all
grok, delays 4,600 / 5,800 / 8,200. The *phenomenon* is robust; the *timing* is
seed-dependent, which turned out to matter later (Phase 7). Kept 164 log-spaced
weight snapshots per seed — the raw material for everything downstream.

## 2026-08-04 — Phase 2: the Fourier lens

The task lives on Z_p, so the natural basis is the discrete Fourier transform.
Projecting the grokked embedding onto it: **96% of its power on 3 frequencies**
(k = 1, 8, 21), against a diffuse spectrum at initialisation. The Gini sparsity
rose 0.05 → 0.25 → 0.93 across memorisation → transition → grokked. First real
evidence the network had learned *structure*, not just fit.

## 2026-08-04 — Phase 3: the circuit, and an honest reframing

This was the phase that needed the most care. My first cut tried to reconstruct the
**raw logits** from the key-frequency 2D-Fourier blocks and measure accuracy — and
got a confusing 46% reconstruction accuracy, even though the same 3 frequencies gave
a 100% classifier under a least-squares cosine fit. The discrepancy was the tell:
the raw logits carry structure that the clean clock formula doesn't, but that
structure doesn't change the *prediction*.

The right object is the **decision function** `f(d)` — the logit averaged over all
(a,b,c) with `a+b-c = d`. That function is a sum of cosines at the key frequencies
to **R² = 0.9999**, peaks exactly at `d = 0` (i.e. `c = a+b`), and reduced to
nothing but that formula still classifies at 100%. The honest caveat, stated not
hidden: `f(d)` accounts for ~66% of the raw centred-logit variance; the residual is
c-structure that doesn't affect the argmax. Reframing around `f(d)` — and reporting
the 66% — is what makes this a true result instead of a cherry-picked R².

To make sure the R²=0.9999 wasn't an artefact of the analysis code, I added a test
that builds a *synthetic* clock (`Σ cos(w(a+b-c))`) with known answer and checks the
analysis recovers it to R²=1, 100% accuracy, and the planted frequencies. The method
is verified independently of any trained model.

## 2026-08-05 — Phase 4: causal verification

Editing the embedding in Fourier space and re-running the untouched network: keep
only the 3 key frequencies → 100% (sufficient); remove only them → chance
(necessary); remove any other single frequency → no effect (specific). The cleanest
phase — necessity, sufficiency and specificity all fell straight out. A nice honest
detail in the sweep: removing *one* key frequency only drops accuracy to ~25% (the
other two still carry signal); it takes removing all three to reach chance.

## 2026-08-05 — Phase 5: progress measures, and a metric that was measuring the wrong thing

The crown result, and it bit me first. I defined "circuit formed" as the step where
the restricted loss falls within 1% of its floor — and got a **−564-step lead**,
i.e. the circuit "formed" *after* grokking. That marker was measuring circuit
*completion*, which is concurrent with generalisation, not onset.

Looking at the actual trajectory fixed it. The embedding sparsity (Gini) rises
monotonically from step ~90, climbing all through the flat plateau. The principled
onset marker is the **restricted/excluded crossover**: the step where keeping only
the key frequencies fits better than removing them — the moment the circuit becomes
the dominant mechanism. That lands at step 2,791, **2,222 steps before** test
accuracy reaches 95%. The three phases — memorisation, circuit formation, cleanup —
are visible, and the "sudden" jump is revealed as the tip of a gradual process.

## 2026-08-05 — CI lint failure

CI went red: ruff flagged 23 style issues (mostly `E702` semicolons in the plotting
code, plus a few long lines). Fixed properly rather than by relaxing the linter —
split every semicolon statement, sorted imports, and set line-length to 120 (a
better fit for the long results-string builders than the initial 100). Pinned the
lesson: run `ruff check` locally against the CI config before every push.

## 2026-08-05 — Phase 6: ablations, and a claim the data didn't support

Swept the recipe. Weight decay is unambiguously the essential knob: **wd = 0 never
groks** (memorises forever), and stronger decay groks sooner (wd 3/1/0.3 → 1,200 /
4,800 / 18,800 steps). A train-fraction threshold sits between 0.2 (never groks) and
0.3. My draft write-up asserted that subtraction "groks like addition" because it is
group-isomorphic — but the run said otherwise: **sub groks ~5× slower** (23,200 vs
4,800). Corrected the claim to match the measurement; isomorphic tasks need not share
training dynamics. (Cut the 1-vs-2-layer depth sweep here for time — a noted gap.)

## 2026-08-05 — Phase 7: the clock generalises

Ran the Phase 3 analysis on all three seeds. Each independently learns the clock
(cosine-fit R² ≥ 0.9998, 100% classification, peak at d=0) — but on **different key
frequencies** (seed 0: [1,8,21]; seed 1: [1,18,45,56]; seed 2: [13,25,44,53]). The
algorithm is universal; the specific frequencies are seed-dependent. Consistent with
Zhong et al. (2023) that attention+MLP transformers favour the "clock"; a full
clock-vs-pizza discrimination is left as further work.

## Open threads / further work

- Sweep the modulus `p` (everything here is p=113).
- The 1-vs-2-layer depth ablation cut from Phase 6.
- A direct clock-vs-pizza discriminator, and the circuit for multiplication (which
  lives on the multiplicative group and should use a different, log-indexed basis).
