"""Correctness of the memory-efficient AttnRes kernel vs the K3 transcription."""
import pytest
import torch

from open_clip.attn_res import attn_res_mix, attn_res_mix_naive, AttnResGate, attn_res_anchor_layers


def _slots(n_slot, n_tok, hidden, dtype, device, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    return [
        (torch.randn(n_tok, hidden, generator=g, dtype=torch.float64) * (0.5 + s))
        .to(device=device, dtype=dtype).requires_grad_(True)
        for s in range(n_slot)
    ]


@pytest.mark.parametrize('n_slot', [2, 3, 9])
def test_forward_matches_naive(n_slot):
    torch.manual_seed(0)
    w = torch.randn(64, dtype=torch.float64) * 0.1
    a = _slots(n_slot, 17, 64, torch.float64, 'cpu')
    b = [t.detach().clone().requires_grad_(True) for t in a]
    out_fast = attn_res_mix(a, w)
    out_ref = attn_res_mix_naive(b, w)
    torch.testing.assert_close(out_fast, out_ref, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('n_slot', [2, 3, 9])
def test_backward_matches_naive(n_slot):
    torch.manual_seed(0)
    w_fast = (torch.randn(64, dtype=torch.float64) * 0.1).requires_grad_(True)
    w_ref = w_fast.detach().clone().requires_grad_(True)
    a = _slots(n_slot, 17, 64, torch.float64, 'cpu')
    b = [t.detach().clone().requires_grad_(True) for t in a]
    gy = torch.randn(17, 64, dtype=torch.float64)

    attn_res_mix(a, w_fast).backward(gy)
    attn_res_mix_naive(b, w_ref).backward(gy)

    torch.testing.assert_close(w_fast.grad, w_ref.grad, rtol=1e-9, atol=1e-11)
    for i, (x, y) in enumerate(zip(a, b)):
        torch.testing.assert_close(x.grad, y.grad, rtol=1e-9, atol=1e-11)


def test_gradcheck():
    """Analytic backward against finite differences -- catches sign/coefficient slips."""
    w = (torch.randn(8, dtype=torch.float64) * 0.3).requires_grad_(True)
    slots = _slots(3, 5, 8, torch.float64, 'cpu', seed=3)
    assert torch.autograd.gradcheck(
        lambda ww, *ss: attn_res_mix(ss, ww), (w, *slots), eps=1e-7, atol=1e-8)


def test_stream_logit_gradcheck():
    w = (torch.randn(8, dtype=torch.float64) * 0.3).requires_grad_(True)
    logit = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
    slots = _slots(3, 5, 8, torch.float64, 'cpu', seed=4)
    assert torch.autograd.gradcheck(
        lambda ww, ll, *ss: attn_res_mix(ss, ww, 1e-6, ll),
        (w, logit, *slots), eps=1e-7, atol=1e-8)


def test_stream_logit_matches_naive():
    w = (torch.randn(16, dtype=torch.float64) * 0.2).requires_grad_(True)
    w2 = w.detach().clone().requires_grad_(True)
    l1 = torch.tensor(2.5, dtype=torch.float64, requires_grad=True)
    l2 = l1.detach().clone().requires_grad_(True)
    a = _slots(4, 7, 16, torch.float64, 'cpu', seed=5)
    b = [t.detach().clone().requires_grad_(True) for t in a]
    gy = torch.randn(7, 16, dtype=torch.float64)
    attn_res_mix(a, w, 1e-6, l1).backward(gy)
    attn_res_mix_naive(b, w2, 1e-6, l2).backward(gy)
    torch.testing.assert_close(l1.grad, l2.grad, rtol=1e-9, atol=1e-11)
    torch.testing.assert_close(w.grad, w2.grad, rtol=1e-9, atol=1e-11)


def test_single_slot_is_identity():
    x = torch.randn(4, 8, requires_grad=True)
    w = torch.randn(8)
    out = attn_res_mix([x], w)
    assert out is x


def test_zero_init_gate_is_uniform_mean():
    """Documents the K3-faithful failure mode: proj_weight=0 -> uniform average."""
    gate = AttnResGate(32, identity_init=False).double()
    slots = _slots(3, 6, 32, torch.float64, 'cpu', seed=7)
    out = gate(slots)
    torch.testing.assert_close(out, sum(s.detach() for s in slots) / 3,
                               rtol=1e-9, atol=1e-11)


def test_identity_init_gate_preserves_stream():
    gate = AttnResGate(32, identity_init=True, identity_logit=20.0).double()
    slots = _slots(3, 6, 32, torch.float64, 'cpu', seed=7)
    torch.testing.assert_close(gate(slots), slots[-1].detach(), rtol=1e-6, atol=1e-7)


def test_anchor_layers():
    assert attn_res_anchor_layers(93, 12) == [0, 12, 24, 36, 48, 60, 72, 84]
    assert attn_res_anchor_layers(24, 6) == [0, 6, 12, 18]


def test_transformer_shapes_and_state_dict_compat():
    """AttnResTransformer must accept a plain Transformer checkpoint (gates aside)."""
    from open_clip.transformer import Transformer, AttnResTransformer
    torch.manual_seed(0)
    plain = Transformer(width=32, layers=6, heads=4)
    ar = AttnResTransformer(width=32, layers=6, heads=4, attn_res_block_size=3)

    missing, unexpected = ar.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected, unexpected
    assert all(('sa_gates' in k or 'mlp_gates' in k or 'output_gate' in k)
               for k in missing), missing
    assert ar.anchor_layers == [0, 3]

    x = torch.randn(2, 7, 32)
    assert ar(x).shape == x.shape
    # layer 0 never reads anchors -> its sa gate is deliberately absent
    assert isinstance(ar.sa_gates[0], torch.nn.Identity)


def test_transformer_forward_intermediates():
    from open_clip.transformer import AttnResTransformer
    ar = AttnResTransformer(width=32, layers=6, heads=4, attn_res_block_size=3)
    x = torch.randn(2, 7, 32)
    out, inter = ar.forward_intermediates(x, indices=[1, 3, 5])
    assert out.shape == x.shape
    assert len(inter) == 3 and all(t.shape == x.shape for t in inter)


def test_transformer_grad_checkpointing_matches():
    from open_clip.transformer import AttnResTransformer
    torch.manual_seed(0)
    ar = AttnResTransformer(width=32, layers=6, heads=4, attn_res_block_size=3).double()
    x = torch.randn(2, 7, 32, dtype=torch.float64, requires_grad=True)

    ar.grad_checkpointing = False
    o1 = ar(x)
    g1 = torch.autograd.grad(o1.square().sum(), x, retain_graph=False)[0]
    ar.grad_checkpointing = True
    o2 = ar(x)
    g2 = torch.autograd.grad(o2.square().sum(), x)[0]

    torch.testing.assert_close(o1, o2, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(g1, g2, rtol=1e-9, atol=1e-11)


def test_transformer_fast_matches_naive_kernel():
    from open_clip.transformer import AttnResTransformer
    torch.manual_seed(0)
    fast = AttnResTransformer(width=32, layers=8, heads=4, attn_res_block_size=3).double()
    slow = AttnResTransformer(width=32, layers=8, heads=4, attn_res_block_size=3,
                              naive_attn_res=True).double()
    slow.load_state_dict(fast.state_dict())
    x = torch.randn(2, 7, 32, dtype=torch.float64)
    torch.testing.assert_close(fast(x), slow(x), rtol=1e-10, atol=1e-12)


def test_vision_transformer_wiring():
    from open_clip.transformer import VisionTransformer, AttnResTransformer
    vt = VisionTransformer(
        image_size=32, patch_size=16, width=32, layers=6, heads=4, mlp_ratio=4.0,
        output_dim=16, attn_res_block_size=3)
    assert isinstance(vt.transformer, AttnResTransformer)
    assert vt(torch.randn(2, 3, 32, 32)).shape == (2, 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs cuda')
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_cuda_no_worse_than_naive(dtype):
    """Both kernels are scored against an fp64 reference, not against each other.

    The fast kernel keeps the ``(T, H)`` elementwise work in the input dtype
    (that is the whole point -- no full-size fp32 copy), so at bf16 it differs
    from the reference kernel by more than bf16 eps. What must hold is that its
    error against ground truth is in the same league.
    """
    S, T, H = 9, 512, 256
    g = torch.Generator().manual_seed(1)
    w64 = torch.randn(H, generator=g, dtype=torch.float64) * 0.1
    s64 = [torch.randn(T, H, generator=g, dtype=torch.float64) * (0.5 + i) for i in range(S)]
    ref = attn_res_mix_naive([s.cuda() for s in s64], w64.cuda())

    a = [s.cuda().to(dtype) for s in s64]
    w = w64.cuda().to(dtype)
    err_fast = (attn_res_mix(a, w).double() - ref).pow(2).mean().sqrt()
    err_naive = (attn_res_mix_naive(a, w).double() - ref).pow(2).mean().sqrt()
    scale = ref.pow(2).mean().sqrt()

    tol = 3e-2 if dtype is torch.bfloat16 else 1e-5
    assert err_fast / scale < tol, f'{err_fast / scale}'
    assert err_fast < 2.0 * err_naive, f'fast {err_fast} vs naive {err_naive}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs cuda')
def test_cuda_fp32_backward_matches_naive():
    dtype = torch.float32
    w = (torch.randn(128, device='cuda', dtype=dtype) * 0.1).requires_grad_(True)
    w2 = w.detach().clone().requires_grad_(True)
    a = _slots(9, 256, 128, dtype, 'cuda', seed=11)
    b = [t.detach().clone().requires_grad_(True) for t in a]
    gy = torch.randn(256, 128, device='cuda', dtype=dtype)
    attn_res_mix(a, w).backward(gy)
    attn_res_mix_naive(b, w2).backward(gy)
    tol = dict(rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(w.grad, w2.grad, **tol)
    for x, y in zip(a, b):
        torch.testing.assert_close(x.grad, y.grad, **tol)
