# Phase 2 — the embedding is sparse in the Fourier basis (only after grokking)

| snapshot | step | Gini(power) | #key freqs (90%) | key frequencies | share of total power |
|---|---|---|---|---|---|
| init | 0 | 0.047 | 50 | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 47, 48, 50, 51, 52, 55, 56] | 0.900 |
| memorised (pre-grok) | 2028 | 0.247 | 46 | [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 28, 29, 30, 31, 33, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 53, 55, 56] | 0.900 |
| grokked (final) | 39999 | 0.928 | 3 | [1, 8, 21] | 0.960 |

**The grokked embedding concentrates 96.0% of its total power on 3 key frequencies: [1, 8, 21].** At initialisation the same power is spread diffusely across all 56 frequencies (low Gini).
