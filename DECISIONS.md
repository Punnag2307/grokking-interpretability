# Decisions

Short ADR-style records of the major engineering choices, each with a brief
rationale. Later entries can supersede earlier ones. (A running log; extended as
the project grows.)

## ADR-0001 — Framework: PyTorch

Use PyTorch (its sibling project DeepBSDE used JAX). Rationale: mechanistic
interpretability lives in the PyTorch ecosystem (TransformerLens, the grokking
literature's released code), so results are easy to cross-check; and eager
execution with explicit `einsum`s keeps the from-scratch model's forward pass
transparent, which is the whole point when every activation must be hookable.

## ADR-0002 — Dependency pinning: exact versions from a verified install

Pin `torch==2.11.0`, `numpy==1.26.4`, `matplotlib==3.8.2`, `pytest==7.4.4` to the
versions verified on the dev machine. Grokking is a claim about a training
*trajectory*, and RNG/optimiser behaviour can shift across library versions;
exact pins keep a clean checkout reproducible.

## ADR-0003 — From-scratch, hookable transformer with no LayerNorm

Write the one-layer transformer by hand with named weight matrices (`W_E`, `W_Q`,
…) and explicit einsums, returning an optional activation `cache`, rather than
using `nn.Transformer` or a library. Rationale: the interpretability phases must
read individual weights, attention patterns and neuron activations; a hand-written
model makes every one of these a named tensor. LayerNorm is off by default so the
learned circuit is a small set of linear maps with a clean closed form (it can be
toggled back on via `Config.use_ln`).

## ADR-0004 — Numerical precision: float32

Use float32 (DeepBSDE used float64). Grokking is about optimisation dynamics on a
classification task, not high-precision numerics validated against closed forms;
float32 is the field standard, matches released baselines, and is faster. The
Fourier *analysis* is done in float64 for clean orthonormality/Parseval checks.

## ADR-0005 — Full-batch training with log-spaced checkpointing

The dataset is tiny (all p² pairs), so train full-batch — the canonical grokking
setup, and it makes the trajectory low-variance. Save a set of log-spaced weight
snapshots per run: grokking spans decades of steps, and the analysis phases
(2–5) replay these snapshots to see *when* the circuit forms relative to when the
test loss drops.

## ADR-0006 — Weight decay is a first-class config field (default 1.0)

Weight decay is not incidental: it drives the late "cleanup" phase that removes
the memorising components, and without it the model memorises but does not grok.
It is therefore an explicit `Config` field with a stability-motivated default of
1.0, and it is swept directly in Phase 6.

## ADR-0007 — Analysis lens: the Fourier basis over Z_p

The task lives on the cyclic group Z_p, so anything indexed by a number token is
analysed in the discrete Fourier basis (constant + cos/sin at each frequency).
The basis is orthonormal, so it redistributes but never inflates power (Parseval),
making "fraction of power at a frequency" and the Gini sparsity measure
well-defined. Every interpretability result — sparsity, key frequencies, the
circuit, the causal ablations, the progress measures — is expressed in this basis.

## ADR-0008 — Report the decision function, and the residual, honestly (Phase 3)

The clock claim is stated about `f(a+b-c)`, the answer-determining part of the
logits, which the key-frequency cosines fit to R²=0.9999 and which classifies at
100%. That function accounts for ~66% of the *raw* centred-logit variance; the
residual is c-structure that does not change the argmax. We report both numbers
rather than quoting only the R²=0.9999 on the restricted object — the residual is
a real property, stated not hidden.

## ADR-0009 — Causal test by editing the embedding in Fourier space (Phase 4)

To test necessity/sufficiency we filter the token embedding to keep or remove
chosen frequencies and re-run the otherwise-untouched network, rather than
ablating internal activations. Rationale: the circuit reads its frequencies out of
the embedding, so this is the cleanest single intervention, and it composes
trivially with the Fourier tooling already built.

## ADR-0010 — Progress-measure onset via the restricted/excluded crossover (Phase 5)

Mark "the circuit has taken over" at the step where the restricted loss (keep only
the key frequencies) drops below the excluded loss (remove them) — a principled,
data-driven point at which the key-frequency circuit fits better than everything
else. An earlier candidate ("restricted loss within 1% of its floor") measured
*completion*, which is concurrent with grokking and gave a meaningless negative
lead; it was rejected.

## ADR-0011 — Interpretability math lives in tested library modules

The analysis primitives (`fourier.py`, `circuit.py`) live in `src/grok` and are
unit-tested on synthetic inputs: Fourier-basis orthonormality and Parseval, Gini
bounds, frequency filtering, and — critically — recovery of a *synthetic* clock
(a hand-built `sum_k cos(w_k(a+b-c))` tensor) to R²=1 with 100% accuracy. This
verifies the *method* independently of any trained model, so a headline number can
never be an artefact of the analysis code.

## ADR-0012 — Lint/format: ruff, line-length 120, no semicolons

CI runs `ruff check`. Line length is 120 (an initial 100 was too tight for the
results-string builders and produced churn) and multi-statement (semicolon) lines
are disallowed (E702). `ruff` is run locally against the same config before every
push so CI stays green.
