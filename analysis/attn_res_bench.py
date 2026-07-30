"""AttnRes memory/speed benchmark: reference kernel vs slot-wise kernel.

Two parts.

1. ``gate``  -- the ``_apply_attn_res`` call site in isolation, at K3's own
   dimensions (H=7168), sweeping slot count. Isolates the kernel from the
   transformer around it.
2. ``tower`` -- a full vision tower forward+backward, sweeping
   ``attn_res_block_size``. ``block_size=1`` is the "every layer keeps an
   anchor" variant that OOMs; ``None`` is the plain baseline.

Usage::

    python analysis/attn_res_bench.py                 # both parts
    python analysis/attn_res_bench.py --part gate
    python analysis/attn_res_bench.py --layers 27 --width 1024 --tokens 1024
"""
import argparse
import gc
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from open_clip.attn_res import attn_res_mix, attn_res_mix_naive  # noqa: E402
from open_clip.transformer import Transformer, AttnResTransformer  # noqa: E402

OOM = (torch.cuda.OutOfMemoryError, RuntimeError)


def _reset(device):
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def _peak_mb(device):
    return torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == 'cuda' else float('nan')


def _timed(fn, device, iters=3):
    """Peak MB and median ms over `iters` runs, or (None, None) on OOM."""
    _reset(device)
    try:
        fn()  # warmup, also the peak-memory sample
    except OOM as e:
        if 'out of memory' not in str(e).lower():
            raise
        _reset(device)
        return None, None
    if device.type == 'cuda':
        torch.cuda.synchronize()
    peak = _peak_mb(device)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return peak, times[len(times) // 2]


def bench_gate(device, dtype, hidden, tokens, slot_counts):
    """One AttnRes call site, K3 dimensions. Slots are leaves, so this measures
    exactly the tensors the kernel itself adds on top of the graph.

    ``resident`` is what the slots + their grads + the incoming cotangent cost
    regardless of kernel, so ``peak - resident`` is the kernel's own footprint --
    the number that actually differs between the two implementations.
    """
    print(f'\n=== gate: H={hidden} T={tokens} dtype={dtype} ===')
    itemsize = torch.empty((), dtype=dtype).element_size()
    print(f'{"slots":>6} {"resid MB":>9} | {"naive MB":>9} {"ovh":>8} | '
          f'{"fast MB":>8} {"ovh":>7} | {"ovh ratio":>10} | {"naive ms":>9} {"fast ms":>8}')
    rows = []
    for n_slot in slot_counts:
        w = torch.randn(hidden, device=device, dtype=dtype, requires_grad=True)
        slots = [torch.randn(tokens, hidden, device=device, dtype=dtype, requires_grad=True)
                 for _ in range(n_slot)]
        gy = torch.randn(tokens, hidden, device=device, dtype=dtype)
        # slots + their grads + cotangent
        resident = (2 * n_slot + 1) * tokens * hidden * itemsize / 2 ** 20

        def run(fn):
            def _f():
                for s in slots:
                    s.grad = None
                w.grad = None
                fn(slots, w).backward(gy)
            return _f

        m_naive, t_naive = _timed(run(attn_res_mix_naive), device)
        m_fast, t_fast = _timed(run(attn_res_mix), device)
        del w, slots, gy
        _reset(device)

        o_naive = m_naive - resident if m_naive else None
        o_fast = m_fast - resident if m_fast else None
        ratio = o_naive / o_fast if (o_naive and o_fast) else float('nan')
        fmt = lambda v, s='{:.1f}': 'OOM' if v is None else s.format(v)
        print(f'{n_slot:6d} {resident:9.1f} | {fmt(m_naive):>9} {fmt(o_naive):>8} | '
              f'{fmt(m_fast):>8} {fmt(o_fast):>7} | {fmt(ratio, "{:.1f}x"):>10} | '
              f'{fmt(t_naive, "{:.2f}"):>9} {fmt(t_fast, "{:.2f}"):>8}')
        rows.append((n_slot, resident, m_naive, m_fast, t_naive, t_fast))
    return rows


def bench_tower(device, dtype, layers, width, heads, tokens, batch, block_sizes, ckpt=False):
    """Full tower fwd+bwd. block_size None -> plain Transformer baseline."""
    print(f'\n=== tower: L={layers} W={width} T={tokens} B={batch} dtype={dtype} '
          f'grad_ckpt={ckpt} ===')
    print(f'{"block":>7} {"anchors":>8} {"kernel":>7} {"peak MB":>9} {"ms":>8} '
          f'{"vs base":>8} {"extra par":>10}')

    base_peak = None
    for block_size in block_sizes:
        for kernel in (['naive', 'fast'] if block_size is not None else ['-']):
            torch.manual_seed(0)
            if block_size is None:
                model = Transformer(width=width, layers=layers, heads=heads)
                n_anchor = 0
            else:
                model = AttnResTransformer(
                    width=width, layers=layers, heads=heads,
                    attn_res_block_size=block_size, naive_attn_res=(kernel == 'naive'))
                n_anchor = len(model.anchor_layers)
            model = model.to(device=device, dtype=dtype)
            model.grad_checkpointing = ckpt
            n_extra = sum(p.numel() for n, p in model.named_parameters()
                          if 'gates' in n or 'output_gate' in n)

            x = torch.randn(batch, tokens, width, device=device, dtype=dtype)

            def _f():
                model.zero_grad(set_to_none=True)
                model(x).square().mean().backward()

            peak, ms = _timed(_f, device, iters=5)
            if block_size is None and peak is not None:
                base_peak = peak
            rel = peak / base_peak if (peak and base_peak) else float('nan')

            del model, x
            _reset(device)

            fmt = lambda v, s='{:.1f}': 'OOM' if v is None else s.format(v)
            print(f'{str(block_size):>7} {n_anchor:8d} {kernel:>7} {fmt(peak):>9} '
                  f'{fmt(ms, "{:.1f}"):>8} {fmt(rel, "{:.2f}x"):>8} {n_extra / 1e3:9.1f}K')


def bench_signal(device, layers, width, heads, tokens, batch, block_sizes):
    """Signal/gradient propagation at init -- why an AttnRes graft can train badly.

    Measures, in fp32 on freshly initialised towers:
      * ``out/in`` -- RMS of the output over RMS of the input. A plain pre-norm
        transformer grows this above 1 (residual accumulation). A softmax gate
        cannot: convex combinations of slots cannot exceed the largest slot, so
        the tower is norm-contracting unless the gate learns otherwise.
      * ``grad`` -- RMS gradient reaching the input for a fixed output cotangent.
        This is the quantity that collapses when block boundaries cut the
        identity highway.
    """
    print(f'\n=== signal at init: L={layers} W={width} T={tokens} B={batch} fp32 ===')
    print(f'{"block":>7} {"init":>9} {"out/in":>8} {"grad RMS":>10} {"vs base":>8}')
    base_grad = None
    for block_size in block_sizes:
        for init in ([False, True] if block_size is not None else [None]):
            torch.manual_seed(0)
            if block_size is None:
                model = Transformer(width=width, layers=layers, heads=heads)
            else:
                model = AttnResTransformer(
                    width=width, layers=layers, heads=heads,
                    attn_res_block_size=block_size, identity_init=init)
            model = model.to(device=device, dtype=torch.float32)

            torch.manual_seed(1)
            x = torch.randn(batch, tokens, width, device=device, requires_grad=True)
            out = model(x)
            gy = torch.randn_like(out)
            g = torch.autograd.grad(out, x, gy)[0]

            rms = lambda t: t.float().pow(2).mean().sqrt().item()
            ratio, grms = rms(out) / rms(x), rms(g)
            if block_size is None:
                base_grad = grms
            label = '-' if init is None else ('identity' if init else 'k3-zero')
            print(f'{str(block_size):>7} {label:>9} {ratio:8.3f} {grms:10.4f} '
                  f'{grms / base_grad:7.2f}x')
            del model, x, out, g
            _reset(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--part', choices=['gate', 'tower', 'signal', 'both', 'all'], default='all')
    p.add_argument('--layers', type=int, default=27, help='K3 vision tower depth')
    p.add_argument('--width', type=int, default=1024, help='K3 vision tower width')
    p.add_argument('--heads', type=int, default=16)
    p.add_argument('--tokens', type=int, default=1024)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--gate-hidden', type=int, default=7168, help='K3 text hidden size')
    p.add_argument('--gate-tokens', type=int, default=4096)
    p.add_argument('--dtype', default='bfloat16')
    p.add_argument('--grad-ckpt', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = getattr(torch, args.dtype)
    if device.type == 'cpu':
        print('WARNING: no CUDA, memory numbers will be nan')

    if args.part in ('gate', 'both', 'all'):
        bench_gate(device, dtype, args.gate_hidden, args.gate_tokens, [2, 3, 5, 9, 17, 47])
    if args.part in ('tower', 'both', 'all'):
        bench_tower(device, dtype, args.layers, args.width, args.heads,
                    args.tokens, args.batch, [None, 1, 2, 3, 4, 7, 14],
                    ckpt=args.grad_ckpt)
    if args.part in ('signal', 'all'):
        bench_signal(device, args.layers, args.width, args.heads,
                     min(args.tokens, 256), 2, [None, 1, 3, 7, 14])


if __name__ == '__main__':
    main()
