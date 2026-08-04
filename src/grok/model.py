"""A from-scratch, fully hookable one-layer transformer.

Written with named weight matrices (``W_E``, ``W_pos``, ``W_Q``, ``W_K``,
``W_V``, ``W_O``, ``W_in``, ``W_out``, ``W_U``) and explicit einsums rather than
``nn.Linear`` / ``nn.MultiheadAttention``, for two reasons:

1. Interpretability. Every activation on the residual stream is returned in an
   optional ``cache`` so later phases can Fourier-analyse the embedding, read
   attention patterns, inspect MLP neuron activations, and do direct logit
   attribution — without monkey-patching or a heavyweight library.
2. Cleanliness. No biases and (by default) no LayerNorm, so the learned circuit
   is a small set of linear maps around one attention block and one MLP — the
   configuration in which the grokking circuit has a clean closed form.

Shapes use single letters: b=batch, q/k=query/key position, h=head, e=d_head,
d=d_model, m=d_mlp, v=vocab. Sequence is ``[a, b, '=']`` and the prediction is
read off the final ('=') position.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


class Transformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d, h, e, m = cfg.d_model, cfg.n_heads, cfg.d_head, cfg.d_mlp

        def param(*shape: int, fan_in: int) -> nn.Parameter:
            # Zero-mean Gaussian init scaled by 1/sqrt(fan_in); standard and
            # grokking-robust once weight decay is on.
            return nn.Parameter(torch.randn(*shape) / math.sqrt(fan_in))

        self.W_E = param(cfg.d_vocab_in, d, fan_in=d)      # token embedding
        self.W_pos = param(cfg.n_ctx, d, fan_in=d)         # learned positional embedding

        self.W_Q = param(h, d, e, fan_in=d)
        self.W_K = param(h, d, e, fan_in=d)
        self.W_V = param(h, d, e, fan_in=d)
        self.W_O = param(h, e, d, fan_in=e * h)

        self.W_in = param(d, m, fan_in=d)                  # MLP up-projection
        self.W_out = param(m, d, fan_in=m)                 # MLP down-projection

        self.W_U = param(d, cfg.d_vocab_out, fan_in=d)     # unembedding

        if cfg.use_ln:
            self.ln1 = nn.LayerNorm(d)
            self.ln2 = nn.LayerNorm(d)

        # causal mask (query cannot attend to future keys)
        mask = torch.tril(torch.ones(cfg.n_ctx, cfg.n_ctx))
        self.register_buffer("causal_mask", mask.bool(), persistent=False)

        self._act = F.relu if cfg.act == "relu" else F.gelu

    # ------------------------------------------------------------------ #
    def forward(
        self, tokens: torch.Tensor, return_cache: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """tokens: (batch, n_ctx) long. Returns logits (batch, d_vocab_out).

        If ``return_cache`` is True, also returns a dict of intermediate
        activations for interpretability.
        """
        cfg = self.cfg
        cache: dict[str, torch.Tensor] = {}

        embed = self.W_E[tokens]                              # (b, q, d)
        resid_pre = embed + self.W_pos[None]                 # broadcast pos over batch

        x = self.ln1(resid_pre) if cfg.use_ln else resid_pre
        q = torch.einsum("bqd,hde->bqhe", x, self.W_Q)
        k = torch.einsum("bkd,hde->bkhe", x, self.W_K)
        v = torch.einsum("bkd,hde->bkhe", x, self.W_V)
        scores = torch.einsum("bqhe,bkhe->bhqk", q, k) / math.sqrt(cfg.d_head)
        scores = scores.masked_fill(~self.causal_mask[None, None], float("-inf"))
        pattern = scores.softmax(dim=-1)                     # (b, h, q, k)
        z = torch.einsum("bhqk,bkhe->bqhe", pattern, v)      # weighted values
        attn_out = torch.einsum("bqhe,hed->bqd", z, self.W_O)
        resid_mid = resid_pre + attn_out

        y = self.ln2(resid_mid) if cfg.use_ln else resid_mid
        mlp_pre = torch.einsum("bqd,dm->bqm", y, self.W_in)  # neuron pre-activations
        mlp_post = self._act(mlp_pre)
        mlp_out = torch.einsum("bqm,md->bqd", mlp_post, self.W_out)
        resid_post = resid_mid + mlp_out

        logits = torch.einsum("bqd,dv->bqv", resid_post, self.W_U)[:, -1, :]

        if return_cache:
            cache.update(
                embed=embed, resid_pre=resid_pre, attn_pattern=pattern, z=z,
                attn_out=attn_out, resid_mid=resid_mid, mlp_pre=mlp_pre,
                mlp_post=mlp_post, mlp_out=mlp_out, resid_post=resid_post,
                logits_all=torch.einsum("bqd,dv->bqv", resid_post, self.W_U),
            )
            return logits, cache
        return logits

    # ------------------------------------------------------------------ #
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
