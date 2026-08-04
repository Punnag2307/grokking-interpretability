# Phase 6 — ablations: where grokking breaks

All runs: p=113, seed 0, 30000 steps. Grok step = first step with test acc >= 0.95.

## Weight decay is the essential knob (frac=0.3, add)
| weight decay | grok step |
|---|---|
| 0.0 | none (>30000) |
| 0.3 | 18800 |
| 1.0 | 4800 |
| 3.0 | 1200 |

Without weight decay the cleanup phase never fires: the model memorises and **never groks**. Weight decay is what forces the generalising circuit; more of it groks sooner.

## Train fraction — a data threshold (wd=1.0, add)
| train fraction | grok step |
|---|---|
| 0.2 | none (>30000) |
| 0.3 | 4800 |
| 0.4 | 1200 |
| 0.5 | 600 |

## Operation (wd=1.0, frac=0.3)
| operation | grok step | final test acc |
|---|---|---|
| add | 4800 | 1.000 |
| sub | 23200 | 1.000 |
| mul | 6200 | 1.000 |

All three operations grok to 100%, but their training dynamics differ: subtraction — though group-isomorphic to addition — groks substantially slower here (23,200 vs 4,800 steps), and multiplication (the multiplicative group mod p) is a genuinely different structure that groks in a comparable time to addition (6,200). Isomorphic tasks need not share training dynamics.
