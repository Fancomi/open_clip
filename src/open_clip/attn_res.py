""" Attention Residuals (AttnRes), memory-efficient re-implementation.

Ported from Kimi-K3's ``modeling_kimi_linear.py`` (``_apply_attn_res`` /
``KimiDecoderLayer._forward_attn_residual``, HF revision 9f62e4e).

What AttnRes does
-----------------
Instead of one identity highway spanning all L layers, the depth axis is cut
into blocks of ``block_size``. At every block boundary the running residual
stream (a *block partial sum*) is frozen into an "anchor"; the stream then
restarts from zero. Each sub-layer reads from ``[anchors..., current_stream]``
via a softmax over depth, where the query is a single learned direction
(rank-1, static, shared across tokens) applied to RMS-normalised slots.

So it is not attention in the usual sense: it is a single-head, static-query
weighted read along depth. Cost is O(L * K) with K = ceil(L / block_size)
anchors, not O(L^2).

Why this file exists
--------------------
The reference implementation materialises three ``(T, S, H)`` tensors per call
site (the ``cat``, its fp32 copy, and the normalised copy), all of which stay
alive for backward. At L=93/H=7168/T=4096 that is hundreds of GB. None of it
is actually needed: the slots are already in the graph (anchors are consumed by
every later layer, the stream is the trunk), and everything else is cheap to
recompute. ``attn_res_mix`` below keeps the slots as a *list* of ``(T, H)``
tensors, never concatenates, and saves only ``probs``/``scores`` of shape
``(T, S)``. See ``analysis/attn_res_bench.py`` for measured numbers.
"""
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

__all__ = [
    'attn_res_mix',
    'attn_res_mix_naive',
    'AttnResGate',
    'attn_res_anchor_layers',
]


def attn_res_anchor_layers(layers: int, block_size: int) -> List[int]:
    """Layer indices that freeze an anchor (K3: ``layer_idx % block_size == 0``)."""
    return [i for i in range(layers) if i % block_size == 0]


def _acc_dtype(dtype: torch.dtype) -> torch.dtype:
    """Accumulation dtype: fp32 for low precision, but never *downcast* fp64.

    K3 hardcodes ``.float()``; that silently degrades fp64 gradcheck runs, so
    promote instead.
    """
    return torch.promote_types(dtype, torch.float32)


def attn_res_mix_naive(
        slots: Sequence[torch.Tensor],
        score_weight: torch.Tensor,
        eps: float = 1e-6,
        stream_logit: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Faithful transcription of K3's ``_apply_attn_res``, for correctness reference.

    Materialises ``(T, S, H)`` three times over. Only use as a baseline.

    Args:
        slots: S tensors of shape ``(T, H)``; last one is the live residual stream.
        score_weight: ``(H,)``, equal to ``norm.weight * proj.weight``.
        stream_logit: optional scalar added to the last slot's score. Not part of
            K3; see :class:`AttnResGate` for why it exists.
    """
    v = torch.stack(list(slots), dim=1)                     # (T, S, H)
    acc = _acc_dtype(v.dtype)
    v_float = v.to(acc)
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + eps)
    scores = (k * score_weight.to(acc)).sum(-1)             # (T, S)
    if stream_logit is not None:
        bias = scores.new_zeros(scores.shape[-1])
        bias[-1] = stream_logit.to(acc)
        scores = scores + bias
    probs = scores.softmax(-1).unsqueeze(1)                 # (T, 1, S)
    return torch.matmul(probs, v_float).squeeze(1).to(v.dtype)


class _AttnResMix(torch.autograd.Function):
    """Slot-wise AttnRes with a hand-written backward and no ``(T, S, H)`` buffers.

    The trick is that the RMSNorm-then-project score collapses to two scalar
    reductions per slot::

        r_s     = rsqrt(mean(v_s^2) + eps)          # (T,)
        dot_s   = v_s @ w                           # (T,)
        score_s = r_s * dot_s

    so the normalised slot ``k_s`` is never formed. Backward needs only
    ``probs``, ``r``, ``dot`` -- all ``(T, S)`` -- plus the slot tensors, which
    the surrounding graph already holds alive (anchors feed every later layer,
    the stream is the trunk).

    fp32 is used for the ``(T, S)`` statistics, the softmax, and the output
    accumulator (where S terms are summed); the ``(T, H)`` elementwise work
    stays in the input dtype so no full-size fp32 copy is ever allocated. That
    is the same split a fused kernel would use internally.
    """

    @staticmethod
    def forward(
            ctx,
            score_weight: torch.Tensor,
            stream_logit: Optional[torch.Tensor],
            eps: float,
            *slots: torch.Tensor,
    ):
        n_tok, hidden = slots[0].shape
        n_slot = len(slots)
        dtype = slots[0].dtype
        acc = _acc_dtype(dtype)
        w_native = score_weight.to(dtype)

        r = torch.empty((n_tok, n_slot), dtype=acc, device=slots[0].device)
        dot = torch.empty((n_tok, n_slot), dtype=acc, device=slots[0].device)
        for s, v in enumerate(slots):
            r[:, s] = torch.rsqrt(v.pow(2).mean(-1, dtype=acc) + eps)
            dot[:, s] = (v @ w_native).to(acc)

        scores = r * dot
        if stream_logit is not None:
            scores[:, -1] += stream_logit.to(acc)
        probs = scores.softmax(-1)                          # (T, S)
        del scores

        out = torch.zeros(n_tok, hidden, dtype=acc, device=slots[0].device)
        for s, v in enumerate(slots):
            out.addcmul_(v, probs[:, s:s + 1])              # mixed-dtype, no copy

        ctx.save_for_backward(probs, r, dot, score_weight, *slots)
        ctx.eps = eps
        ctx.hidden = hidden
        ctx.acc = acc
        ctx.has_logit = stream_logit is not None
        return out.to(dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        probs, r, dot, score_weight, *slots = ctx.saved_tensors
        hidden = ctx.hidden
        acc = ctx.acc
        n_slot = len(slots)
        dtype = slots[0].dtype
        w_native = score_weight.to(dtype)

        # d L / d probs[:, s] -- from out = sum_s probs_s * v_s
        d_probs = torch.empty_like(probs)
        for s, v in enumerate(slots):
            d_probs[:, s] = (grad_out * v).sum(-1, dtype=acc)

        # softmax Jacobian
        d_scores = probs * (d_probs - (probs * d_probs).sum(-1, keepdim=True))
        del d_probs

        need_w = ctx.needs_input_grad[0]
        d_w = torch.zeros(hidden, dtype=acc, device=grad_out.device) if need_w else None
        grads: List[Optional[torch.Tensor]] = [None] * n_slot

        for s, v in enumerate(slots):
            ds = d_scores[:, s:s + 1]                       # (T, 1)
            rs = r[:, s:s + 1]
            ds_rs = (ds * rs).to(dtype)
            if need_w:
                # d score_s / d w = r_s * v_s
                d_w += (ds_rs * v).sum(0, dtype=acc)
            if not ctx.needs_input_grad[3 + s]:
                continue
            # direct path: out = sum_s probs_s * v_s
            gv = grad_out * probs[:, s:s + 1].to(dtype)
            # score path: score_s = rsqrt(mean(v_s^2)+eps) * (v_s . w)
            #   d/dv = r_s * w  -  dot_s * r_s^3 / H * v_s
            coef = (-(ds * dot[:, s:s + 1] * rs.pow(3)) / hidden).to(dtype)
            gv += ds_rs * w_native
            gv.addcmul_(v, coef)
            grads[s] = gv

        d_w_out = d_w.to(score_weight.dtype) if need_w else None
        d_logit = None
        if ctx.has_logit and ctx.needs_input_grad[1]:
            d_logit = d_scores[:, -1].sum().reshape(())
        return (d_w_out, d_logit, None, *grads)


def attn_res_mix(
        slots: Sequence[torch.Tensor],
        score_weight: torch.Tensor,
        eps: float = 1e-6,
        stream_logit: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Memory-efficient AttnRes read. Numerically matches :func:`attn_res_mix_naive`.

    Args:
        slots: S tensors of shape ``(T, H)``; last one is the live residual stream.
        score_weight: ``(H,)``, equal to ``norm.weight * proj.weight``.
        stream_logit: optional scalar added to the last slot's score.
    """
    if len(slots) == 1:
        # softmax over a single slot is exactly 1.0 -- skip the whole thing
        return slots[0]
    return _AttnResMix.apply(score_weight, stream_logit, eps, *slots)


class AttnResGate(nn.Module):
    """One AttnRes read head: the RMSNorm gain and the rank-1 query direction.

    Mirrors K3's ``(*_res_norm, *_res_proj)`` pairs. Those only ever appear as
    the elementwise product ``norm.weight * proj.weight``, so both are kept
    (checkpoint-compatible) but folded before use.

    ``identity_init`` adds a scalar logit on the live-stream slot, initialised to
    ``+8`` so the softmax starts as ~one-hot on the stream. This is *not* in K3
    and it is **not** a no-op w.r.t. a plain pre-norm transformer -- no softmax
    gate can be, since a convex combination cannot reproduce ``anchor + stream``.
    What it buys is the right activation *scale* at init: the K3-faithful form
    (``proj_weight = 0``) emits the uniform mean over all slots, shrinking the
    signal by ``1/S`` and blending in stale anchors, which wrecks pretrained
    weights immediately. Stream-dominant init at least starts from "block-local
    transformer, anchors not yet read".
    """

    def __init__(
            self,
            width: int,
            eps: float = 1e-6,
            naive: bool = False,
            identity_init: bool = False,
            identity_logit: float = 8.0,
    ):
        super().__init__()
        self.eps = eps
        self.naive = naive
        self.norm_weight = nn.Parameter(torch.ones(width))
        self.proj_weight = nn.Parameter(torch.zeros(1, width))
        if identity_init:
            self.stream_logit = nn.Parameter(torch.tensor(identity_logit))
        else:
            self.register_parameter('stream_logit', None)

    def extra_repr(self) -> str:
        return (f'width={self.norm_weight.numel()}, eps={self.eps}, '
                f'naive={self.naive}, identity_init={self.stream_logit is not None}')

    def forward(self, slots: Sequence[torch.Tensor]) -> torch.Tensor:
        score_weight = self.norm_weight * self.proj_weight.squeeze(0)
        fn = attn_res_mix_naive if self.naive else attn_res_mix
        return fn(slots, score_weight, self.eps, self.stream_logit)
