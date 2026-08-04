# Phase 4 — causal verification: the key frequencies are necessary and sufficient

Key frequencies: **[1, 8, 21]**.  Chance accuracy = 1/113 = 0.0088.

| intervention on the embedding | accuracy |
|---|---|
| none (full model) | 1.0000 |
| **keep ONLY** the key frequencies (sufficiency) | 1.0000 |
| **remove** the key frequencies (necessity) | 0.0089 |

- **Sufficiency:** keeping only 3 of 56 frequencies retains **100.0%** accuracy — those frequencies *are* the computation.
- **Necessity:** removing just those 3 frequencies drops accuracy to **0.9%** (chance is 0.9%) — nothing else can do the task.
- **Specificity:** across the single-frequency sweep, the 3 key frequencies are the only ones whose removal hurts; the median accuracy after ablating a *non-key* frequency is 1.0000.

**The 3 frequencies the network learned are causally responsible for the whole task: keep only them and it still scores 100%; remove only them and it falls to chance.**
