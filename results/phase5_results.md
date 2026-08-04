# Phase 5 — progress measures: grokking is gradual underneath a sudden curve

Target circuit = key frequencies [1, 8, 21] of the final model, held fixed along the run.

Test accuracy is pinned near chance until it snaps to 100% around step **5013** — the "sudden" grokking. But the internal measures move continuously through the supposedly-flat plateau:

- **The key-frequency circuit becomes the dominant mechanism at step 2791** — where keeping *only* those frequencies fits the training data better than removing them (restricted loss drops below excluded loss). That is **2222 steps before** test accuracy reaches 0.95.
- **Embedding sparsity rises monotonically from the start**: Gini 0.05 → 0.21 by step 1397, while test accuracy is still at chance (0.04).
- The **three phases** are visible: *memorisation* (restricted loss high, model fits via many directions), *circuit formation* (restricted loss falls, excluded loss rises, sparsity climbs), *cleanup* (weight norm decays under weight decay and the test loss finally drops).

| stage | test acc | restricted loss | excluded loss | Gini |
|---|---|---|---|---|
| plateau (step 1397) | 0.035 | 10.45 | 3.59 | 0.206 |
| circuit dominant (step 2791) | 0.084 | 4.86 | 4.87 | 0.306 |
| grokking (step 5013) | 0.983 | 0.17 | 19.86 | 0.644 |
| final (step 39999) | 1.000 | 0.00 | 7.94 | 0.928 |

**The 'sudden' jump is the visible tip of a continuous reorganisation: by the time test accuracy first twitches, the network has been building its Fourier circuit for thousands of steps.**
