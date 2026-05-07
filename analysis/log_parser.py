"""Parse training logs and render result tables + plots.

Usage (from repo root):
    python -m analysis.log_parser                        # all wmc_* logs → MD + plots
    python -m analysis.log_parser --prefix wmc_aux_      # filter prefix
    python -m analysis.log_parser --no-plot               # table only
    python -m analysis.log_parser --single <logdir> --out <csv>  # per-epoch CSV
"""
import argparse, csv, re, json, logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

LOGS_DIR    = Path("logs")
MD_PATH     = Path("analysis/research/modality_gap_wm.md")
TABLE_START = "<!-- RESULTS_TABLE_START -->"
TABLE_END   = "<!-- RESULTS_TABLE_END -->"


# ── tag parsing ───────────────────────────────────────────────────────────────

# modifier → (modal_sides, lambda_scale)
_MOD = {
    'img':  ('img',  0.01),
    'txt':  ('txt',  0.01),
    'both': ('both', 0.01),
    'wm':   ('both', None),   # wm uses special lambda decoding
}

def _decode_wm_lam(s: str) -> float:
    if s.startswith("0"):   return int(s) / 100.0   # 025→0.25
    if len(s) == 1:         return float(s)           # 1→1.0
    return int(s) / 10.0                              # 15→1.5

def parse_tag(dirname: str):
    """dirname wmc_{tag}_{MMDD}_{HHMM} → (tag, sides, lambda)

    sides encodes both modality and series prefix, e.g. "aux_txt", "v2_img".
    Any prefix before the modifier is preserved automatically.
    """
    m = re.match(r"wmc_(.+?)_\d{4}_\d{4}/?$", dirname)
    if not m:
        return None, None, None
    tag = m.group(1)

    if re.match(r"^baseline\d*$", tag):
        return tag, "—", 0.0

    # Generic: optional_prefix _ modifier number
    mo = re.match(r"^(?:(.+)_)?(img|txt|both|wm)(\d+)$", tag)
    if mo:
        prefix, mod, num = mo.group(1), mo.group(2), mo.group(3)
        modal, scale = _MOD[mod]
        sides = f"{prefix}_{modal}" if prefix else modal
        lam   = _decode_wm_lam(num) if scale is None else int(num) * scale
        return tag, sides, lam

    return tag, "?", 0.0


# ── log parsing ───────────────────────────────────────────────────────────────

# Fields to exclude from per-epoch data (not useful for analysis)
_SKIP_FIELDS = {'epoch', 'Epoch', 'num_samples'}

# Canonical short names for known verbose keys
_KEY_ALIAS = {
    'image_to_text_mean_rank':   'i2t_mean_rank',
    'image_to_text_median_rank': 'i2t_median_rank',
    'image_to_text_R@1':         'i2t_r1',
    'image_to_text_R@5':         'i2t_r5',
    'image_to_text_R@10':        'i2t_r10',
    'text_to_image_mean_rank':   't2i_mean_rank',
    'text_to_image_median_rank': 't2i_median_rank',
    'text_to_image_R@1':         't2i_r1',
    'text_to_image_R@5':         't2i_r5',
    'text_to_image_R@10':        't2i_r10',
    'clip_val_loss':             'val_loss',
    'siglip_val_loss':           'val_loss',
}

def _parse_kv(line: str) -> dict:
    """Extract all `key: float` pairs from a log line."""
    return {k: float(v) for k, v in re.findall(r'([\w@]+):\s+([0-9.]+)', line)}

def parse_log(log_path: Path) -> dict | None:
    """Return final eval metrics + full per-epoch history from out.log, or None."""
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except Exception:
        return None

    epochs: dict[int, dict] = {}
    last_scale = last_bias = None

    for line in lines:
        # eval result lines (new: [i2t]/[t2i]/[etc], old: single-line)
        m = re.search(r'Eval Epoch: (\d+)\b', line)
        if m and re.search(r'(image_to_text|text_to_image|val_loss)', line):
            ep  = int(m.group(1))
            kv  = _parse_kv(line)
            row = epochs.setdefault(ep, {})
            for raw_k, v in kv.items():
                k = _KEY_ALIAS.get(raw_k, raw_k)
                if k not in _SKIP_FIELDS:
                    row[k] = v

        m = re.search(r'Logit Scale: ([0-9.]+)\s+Logit Bias: (-?[0-9.]+)', line)
        if m:
            last_scale, last_bias = float(m.group(1)), float(m.group(2))

    complete = {ep: v for ep, v in epochs.items() if 'i2t_r1' in v}
    if not complete:
        return None

    last_ep = max(complete)
    best_ep = max(complete, key=lambda ep: complete[ep].get('i2t_r1', 0))
    return {
        'epoch':       last_ep,
        'i2t_r1':      complete[last_ep].get('i2t_r1'),
        'i2t_r5':      complete[last_ep].get('i2t_r5'),
        'i2t_r10':     complete[last_ep].get('i2t_r10'),
        't2i_r1':      complete[last_ep].get('t2i_r1'),
        't2i_r5':      complete[last_ep].get('t2i_r5'),
        't2i_r10':     complete[last_ep].get('t2i_r10'),
        'val_loss':    complete[last_ep].get('val_loss'),
        'scale':       last_scale,
        'bias':        last_bias,
        # best epoch (by i2t R@1)
        'best_epoch':  best_ep,
        'best_i2t_r1': complete[best_ep].get('i2t_r1'),
        'best_t2i_r1': complete[best_ep].get('t2i_r1'),
        'history':     {ep: v for ep, v in sorted(complete.items())},
    }


# ── table ─────────────────────────────────────────────────────────────────────

def _f(v, d=4):
    return "—" if v is None else f"{v:.{d}f}"

def _lam_str(sides, lam):
    if sides == "—": return "0"
    return f"{lam:.2f}".rstrip("0").rstrip(".")

def build_table(entries: list) -> str:
    hdr = ("| 实验 | sides | λ | i2t R@1 | t2i R@1 | "
           "i2t R@5 | t2i R@5 | val_loss | Scale | Epoch |\n")
    sep = ("|------|-------|---|---------|---------|"
           "---------|---------|----------|-------|-------|\n")
    rows = []
    for tag, sides, lam, r in entries:
        note = " ★" if r["epoch"] is not None and r["epoch"] < 18 else ""
        rows.append(
            f"| {tag} | {sides} | {_lam_str(sides, lam)} "
            f"| {_f(r['i2t_r1'])} | {_f(r['t2i_r1'])} "
            f"| {_f(r['i2t_r5'])} | {_f(r['t2i_r5'])} "
            f"| {_f(r.get('val_loss'), 4)} "
            f"| {_f(r['scale'])} | {r['epoch']}{note} |\n"
        )
    return hdr + sep + "".join(rows)


# ── MD injection ──────────────────────────────────────────────────────────────

def inject_md(table: str):
    if not MD_PATH.exists():
        logging.warning(f"[log_parser] MD not found: {MD_PATH}")
        return
    md = MD_PATH.read_text()
    if TABLE_START not in md or TABLE_END not in md:
        logging.warning(f"[log_parser] table markers missing in {MD_PATH}")
        return
    new_md = re.sub(
        re.escape(TABLE_START) + r".*?" + re.escape(TABLE_END),
        TABLE_START + "\n" + table + TABLE_END,
        md, flags=re.DOTALL,
    )
    if new_md != md:
        MD_PATH.write_text(new_md)
        logging.info(f"[log_parser] MD updated: {MD_PATH}")


# ── plots ─────────────────────────────────────────────────────────────────────

_MOD_ORDER = {"—": 0, "img": 1, "txt": 2, "both": 3}

def _sides_sort_key(sides: str) -> tuple:
    """Sort by modifier family first, then series prefix alphabetically."""
    mod = sides.split('_')[-1]
    prefix = sides[: -(len(mod) + 1)] if '_' in sides else ''
    return (_MOD_ORDER.get(mod, 99), prefix)

def _plot_results(entries: list, out_dir: Path):
    """Bar chart: best i2t/t2i R@1 per experiment; line chart: best R@1 vs λ per side."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # bar: sorted by tag name
    bar_entries = sorted(entries, key=lambda e: e[0])
    tags   = [e[0] for e in bar_entries]
    i2t_r1 = [e[3].get('best_i2t_r1') or 0 for e in bar_entries]
    t2i_r1 = [e[3].get('best_t2i_r1') or 0 for e in bar_entries]

    # ── bar chart ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, len(tags) * 0.6), 4))
    x = np.arange(len(tags))
    w = 0.38
    ax.bar(x - w/2, i2t_r1, w, label='i2t R@1 (best)', color='steelblue')
    ax.bar(x + w/2, t2i_r1, w, label='t2i R@1 (best)', color='tomato')
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('R@1')
    ax.set_title('Recall@1 per experiment')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    p = out_dir / 'wmc_r1_bar.png'
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    logging.info(f'[log_parser] {p}')

    # ── line: best R@1 vs λ, grouped by sides ────────────────────────────
    groups: dict[str, list] = {}
    for tag, sides, lam, r in entries:
        if sides == "—": continue
        groups.setdefault(sides, []).append(
            (lam, r.get('best_i2t_r1') or 0, r.get('best_t2i_r1') or 0))
    if not groups:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Color by modifier family; cycle shades for multiple series in same family
    _family_base = {'img': '#1f77b4', 'txt': '#d62728', 'both': '#2ca02c'}
    _family_members: dict[str, list] = {}
    for s in sorted(groups):
        mod = s.split('_')[-1]   # last segment is always the modifier
        _family_members.setdefault(mod, []).append(s)
    # build color map: first member gets base color, rest lighten progressively
    _color: dict[str, str] = {}
    import colorsys
    for mod, members in _family_members.items():
        base_hex = _family_base.get(mod, '#888888')
        r, g, b = (int(base_hex[i:i+2], 16)/255 for i in (1, 3, 5))
        h, s_val, v = colorsys.rgb_to_hsv(r, g, b)
        for i, sid in enumerate(members):
            lightness = min(0.95, v + i * 0.22)
            r2, g2, b2 = colorsys.hsv_to_rgb(h, max(0.25, s_val - i * 0.3), lightness)
            _color[sid] = '#{:02x}{:02x}{:02x}'.format(int(r2*255), int(g2*255), int(b2*255))
    for sides, pts in groups.items():
        pts.sort()
        lams, i2t, t2i = zip(*pts)
        c = _color.get(sides, 'gray')
        axes[0].plot(lams, i2t, 'o-', label=sides, color=c)
        axes[1].plot(lams, t2i, 'o-', label=sides, color=c)
    for ax, title in zip(axes, ['i2t R@1 (best) vs λ', 't2i R@1 (best) vs λ']):
        ax.set_xlabel('λ'); ax.set_ylabel('R@1')
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    p = out_dir / 'wmc_r1_vs_lambda.png'
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    logging.info(f'[log_parser] {p}')

    # ── training curves (val_loss + i2t R@1 per epoch) ───────────────────
    has_history = [(tag, r) for tag, _, _, r in entries
                   if len(r.get("history", {})) > 1]
    if not has_history:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    cmap = plt.cm.tab20
    for idx, (tag, r) in enumerate(has_history):
        hist = r["history"]
        eps  = sorted(hist)
        i2ts = [hist[ep].get("i2t_r1", float('nan')) for ep in eps]
        vloss= [hist[ep].get("val_loss", float('nan')) for ep in eps]
        c = cmap(idx % 20)
        axes[0].plot(eps, i2ts,  marker='.', lw=1, color=c, label=tag, alpha=0.85)
        axes[1].plot(eps, vloss, marker='.', lw=1, color=c, label=tag, alpha=0.85)
    for ax, title in zip(axes, ['i2t R@1 training curve', 'val_loss training curve']):
        ax.set_xlabel('Epoch'); ax.set_ylabel(title.split()[0])
        ax.set_title(title); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6, ncol=2)
    plt.tight_layout()
    p = out_dir / 'wmc_training_curves.png'
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    logging.info(f'[log_parser] {p}')


# ── main ──────────────────────────────────────────────────────────────────────

def collect_entries(prefix: str = "wmc_") -> list:
    entries = []
    for d in sorted(LOGS_DIR.iterdir()):
        if not d.name.startswith(prefix):
            continue
        tag, sides, lam = parse_tag(d.name)
        if tag is None:
            continue
        res = parse_log(d / "out.log")
        if res is None:
            logging.info(f"[log_parser] SKIP {d.name} (no eval)")
            continue
        entries.append((tag, sides, lam, res))
    entries.sort(key=lambda x: (_sides_sort_key(x[1]), x[2]))
    return entries


# ── single-experiment export ───────────────────────────────────────────────────

# Plot layout: (title, [(col, label, color), ...])
_CURVE_PANELS = [
    ('Recall@K — Image→Text',  [('i2t_r1',  'R@1',  '#1f77b4'),
                                 ('i2t_r5',  'R@5',  '#ff7f0e'),
                                 ('i2t_r10', 'R@10', '#2ca02c')]),
    ('Recall@K — Text→Image',  [('t2i_r1',  'R@1',  '#1f77b4'),
                                 ('t2i_r5',  'R@5',  '#ff7f0e'),
                                 ('t2i_r10', 'R@10', '#2ca02c')]),
    ('Mean Rank',              [('i2t_mean_rank', 'i2t', '#1f77b4'),
                                 ('t2i_mean_rank', 't2i', '#d62728')]),
    ('Median Rank',            [('i2t_median_rank', 'i2t', '#1f77b4'),
                                 ('t2i_median_rank', 't2i', '#d62728')]),
    ('Val Loss',               [('val_loss', 'loss', '#9467bd')]),
]


def plot_training_curves(history: dict, out_path: Path, exp_name: str = ''):
    """Plot per-epoch training metrics to out_path.

    history : {epoch: {metric: value, ...}}
    """
    eps = sorted(history)
    def _series(key):
        return [history[ep].get(key) for ep in eps]

    # filter panels to those with at least one non-None series
    active = [(title, cols) for title, cols in _CURVE_PANELS
              if any(any(v is not None for v in _series(c)) for c, *_ in cols)]
    if not active:
        return

    n = len(active)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (title, cols) in zip(axes, active):
        for col, label, color in cols:
            ys = _series(col)
            if all(v is None for v in ys):
                continue
            ax.plot(eps, ys, marker='o', ms=3, lw=1.4, color=color, label=label)
        ax.set_xlabel('Epoch')
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        # rank axes: lower is better — invert so improvement goes up
        if 'Rank' in title:
            ax.invert_yaxis()

    if exp_name:
        fig.suptitle(exp_name, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f'[log_parser] plot → {out_path}')


def export_single(logdir: Path, out_path: Path) -> bool:
    """Write per-epoch metrics CSV + training curve plot for one experiment.

    Returns True on success, False if no eval data found.
    """
    res = parse_log(logdir / "out.log")
    if not res or not res.get("history"):
        return False
    hist   = res["history"]
    fields = sorted({k for ep_data in hist.values() for k in ep_data})
    with open(out_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['epoch'] + fields)
        w.writeheader()
        for ep in sorted(hist):
            w.writerow({'epoch': ep, **{f: hist[ep].get(f, '') for f in fields}})
    plot_training_curves(hist, out_path.with_name('training_curves.png'),
                         exp_name=logdir.name)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix",   default="wmc_",
                    help="log dir prefix filter (default: wmc_)")
    ap.add_argument("--no-plot",  action="store_true", help="skip plot generation")
    ap.add_argument("--plot-dir", default="analysis/research/plots",
                    help="output dir for plots (default: analysis/research/plots)")
    ap.add_argument("--no-md",    action="store_true", help="skip MD injection")
    # single-experiment export
    ap.add_argument("--single",   default=None, metavar="LOGDIR",
                    help="export per-epoch CSV for one logdir (skips global scan)")
    ap.add_argument("--out",      default=None, metavar="CSV",
                    help="output CSV path for --single (required with --single)")
    args = ap.parse_args()

    if args.single:
        logdir = Path(args.single)
        out    = Path(args.out) if args.out else logdir / "probe" / "plots" / "training_metrics.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        ok = export_single(logdir, out)
        if ok:
            logging.info(f"[log_parser] {logdir.name} → {out}")
        else:
            logging.warning(f"[log_parser] no eval data in {logdir / 'out.log'}")
        return

    entries = collect_entries(args.prefix)
    if not entries:
        logging.warning(f"[log_parser] no logs found with prefix '{args.prefix}'")
        return

    table = build_table(entries)
    print(table)

    if not args.no_md:
        inject_md(table)

    if not args.no_plot:
        _plot_results(entries, Path(args.plot_dir))

    logging.info(f"[log_parser] {len(entries)} experiments")


if __name__ == "__main__":
    main()
