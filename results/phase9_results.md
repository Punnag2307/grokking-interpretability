# Phase 9 — how the circuit scales with the modulus p

Fixed-capacity model (d_model=128), seed 0, 35000 steps. Effective #frequencies = minimal set carrying 90% of the embedding's non-constant power.

| p | grokked at | final test acc | # key frequencies | key frequencies |
|---|---|---|---|---|
| 53 | 11600 | 0.992 | 3 | [1, 6, 14] |
| 71 | 7800 | 0.998 | 3 | [14, 29, 32] |
| 97 | 8600 | 1.000 | 3 | [1, 10, 45] |
| 113 | 4800 | 1.000 | 3 | [1, 8, 21] |
| 149 | 11200 | 1.000 | 6 | [5, 6, 8, 23, 44, 46] |
| 191 | 3600 | 1.000 | 6 | [6, 8, 11, 21, 22, 62] |

Across primes p = 53–191 (a 3.6× range), the effective number of Fourier frequencies stays around **4.0** (range 3–6). The circuit size is set by the model's fixed capacity, not by the modulus: a larger problem is solved with essentially the same-sized Fourier circuit, on a different set of frequencies.
