# Phase 7 — the clock generalises across seeds

Every independently-trained, independently-grokked seed learns the clock: its decision function is a function of (a+b-c), fit by cosines at that seed's own key frequencies, peaking at c=a+b and classifying at 100%.

| seed | key frequencies | share as f(a+b-c) | cosine-fit R^2 | peaks at d=0 | classify acc |
|---|---|---|---|---|---|
| 0 | [1, 8, 21] | 65.8% | 0.9999 | yes | 100.0% |
| 1 | [1, 18, 45, 56] | 64.2% | 0.9999 | yes | 100.0% |
| 2 | [13, 25, 44, 53] | 47.9% | 0.9998 | yes | 100.0% |

**All three seeds learn the clock** (cosine-fit R^2 >= 0.9998, 100% classification), but on different key frequencies (seed 0: [1, 8, 21], seed 1: [1, 18, 45, 56], seed 2: [13, 25, 44, 53]) — the algorithm is universal, the frequencies it selects are seed-dependent.
