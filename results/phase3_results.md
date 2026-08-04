# Phase 3 — the circuit: logits = sum over key frequencies of cos(w_k (a+b-c))

Key frequencies from the **embedding**: [1, 8, 21]. Dominant frequencies of the **logits** (computed independently): [1, 8, 21]. **They are the same set** — the embedding's frequencies are exactly the ones the logits compute with.

## The decision function is the closed-form clock
- The answer-determining signal is a pure function `f(d)` of `d = (a+b-c) mod p`, accounting for **65.8%** of the centred-logit variance.
- `f(d)` fits `sum_k [A_k cos + B_k sin](w_k d)` over the 3 key frequencies to **R^2 = 0.9999**.
- `f(d)` is **maximised at d = 0** (the clock predicts 0, i.e. c = a+b). ✓
- Per-frequency amplitude: k=1: 9.63, k=8: 14.62, k=21: 12.36

## Sufficiency
- Reduced to nothing but `argmax_c f(a+b-c)`, the formula classifies the task with **100.0% accuracy**.

## Honest scope
- `f(d)` captures 66% of the centred-logit variance; the residual is c-dependent structure not of the form (a+b-c) that does **not** change the argmax (hence 100% accuracy despite the residual). It is reported, not hidden.

**A transformer trained only on examples has become the closed-form expression `sum_k cos(w_k(a+b-c))` over just 3 frequencies (R^2 = 0.9999 on its own decision function), and reduced to nothing but that formula it still solves modular addition with 100% accuracy.**
