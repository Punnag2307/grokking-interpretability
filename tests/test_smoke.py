"""Phase 0 correctness and reproducibility tests.

Fast (CPU, tiny p and few steps). They pin down the contracts the analysis
phases rely on: correct task labels, sane shapes, a working cache, that the
model can actually fit its training data, and — critically for grokking, which
is a claim about a trajectory — that a run is bit-reproducible from its seed.
"""
from __future__ import annotations

import torch

from grok import Config, Transformer, make_dataset, set_seed
from grok.train import train


def _tiny(**kw) -> Config:
    base = dict(p=7, d_model=32, n_heads=4, d_head=8, d_mlp=64,
                epochs=40, eval_every=10, n_checkpoints=5, device="cpu")
    base.update(kw)
    return Config(**base)


def test_task_labels_correct():
    for op, fn in [("add", lambda a, b, p: (a + b) % p),
                   ("sub", lambda a, b, p: (a - b) % p),
                   ("mul", lambda a, b, p: (a * b) % p)]:
        cfg = _tiny(op=op)
        d = make_dataset(cfg, device="cpu")
        a, b, eq = d.all_x[:, 0], d.all_x[:, 1], d.all_x[:, 2]
        assert (eq == cfg.p).all(), "third token must be the '=' id (== p)"
        assert torch.equal(d.all_y, fn(a, b, cfg.p))
        assert d.all_x.shape == (cfg.p * cfg.p, 3)
        # train/test partition is a disjoint cover of all p^2 pairs
        assert d.train_x.shape[0] + d.test_x.shape[0] == cfg.p * cfg.p


def test_forward_shapes_and_cache():
    cfg = _tiny()
    set_seed(0)
    model = Transformer(cfg)
    d = make_dataset(cfg, device="cpu")
    logits, cache = model(d.all_x, return_cache=True)
    assert logits.shape == (cfg.p * cfg.p, cfg.d_vocab_out)
    assert cache["attn_pattern"].shape == (cfg.p * cfg.p, cfg.n_heads, cfg.n_ctx, cfg.n_ctx)
    assert cache["mlp_post"].shape == (cfg.p * cfg.p, cfg.n_ctx, cfg.d_mlp)
    # attention rows are probability distributions (sum to 1 over keys)
    assert torch.allclose(cache["attn_pattern"].sum(-1),
                          torch.ones(cfg.p * cfg.p, cfg.n_heads, cfg.n_ctx), atol=1e-5)


def test_can_fit_training_data():
    # On a tiny task the model should memorise train quickly — a sanity check
    # that the optimisation loop and gradients are wired correctly.
    res = train(_tiny(epochs=400, weight_decay=0.0), verbose=False)
    assert res.history.train_acc[-1] > 0.9


def test_seed_reproducibility():
    res1 = train(_tiny(), verbose=False)
    res2 = train(_tiny(), verbose=False)
    assert res1.history.train_loss == res2.history.train_loss
    for k in res1.model.state_dict():
        assert torch.equal(res1.model.state_dict()[k], res2.model.state_dict()[k])


def test_checkpoints_span_run():
    res = train(_tiny(epochs=100, n_checkpoints=8), verbose=False)
    steps = [c.step for c in res.checkpoints]
    assert steps == sorted(steps)
    assert steps[0] == 0 and steps[-1] == 99
