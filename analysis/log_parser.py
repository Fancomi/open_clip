"""Parse training logs and render result tables + plots.

Usage (from repo root):
    python -m analysis.log_parser                             # all logs → MD + plots
    python -m analysis.log_parser --prefix ft_               # filter prefix
    python -m analysis.log_parser --logs-dir logs/book_run   # custom log root
    python -m analysis.log_parser --no-plot                  # table only
    python -m analysis.log_parser --single <logdir> --out <csv>
"""
import argparse, csv, re, json, logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

LOGS_DIR    = Path("logs")   # default; overridden by --logs-dir at runtime
MD_PATH     = Path("analysis/research/mgap_02_within_modal_repulsion.md")
TABLE_START = "<!-- RESULTS_TABLE_START -->"
TABLE_END   = "<!-- RESULTS_TABLE_END -->"

_DEFAULT_MD_MARKER = "RESULTS_TABLE"  # marker name (without <!-- ... -->)


# ── tag parsing ───────────────────────────────────────────────────────────────

# modifier → (method_label, lambda_scale)
# scale=None means use float-decode (e.g. wm's special encoding)
_MOD = {
    'img':   ('img',     0.01),
    'txt':   ('txt',     0.01),
    'both':  ('both',    0.01),
    'wm':    ('both',    None),    # wm uses special lambda decoding
    'koleo': ('koleo',   0.001),   # koleo005 → λ=0.005
    'uni':   ('uni',     0.001),   # uni05  → λ=0.05
    'gap':   ('gap',     0.001),   # gap001 → λ=0.001
    'hmix':  ('hmix',    None),    # hmix: no single λ, use tag as-is
    'mix':   ('mix',     None),    # mix_*: composite
    'ada':   ('ada',     0.001),
}

def _decode_wm_lam(s: str) -> float:
    if s.startswith("0"):   return int(s) / 100.0   # 025→0.25
    if len(s) == 1:         return float(s)           # 1→1.0
    return int(s) / 10.0                              # 15→1.5

def parse_tag(dirname: str):
    """dirname {anything}[_{MMDD}_{HHMM}] → (tag, method, lambda)

    Strips optional trailing timestamp _DDDD_DDDD and any leading experiment
    batch prefix (e.g. wmc_, ft_) to produce a clean short tag.

    Supports generic modifier naming: koleo, uni, gap, wm, img, txt, both, hmix, mix, ada.
    """
    # Strip optional trailing timestamp  _DDDD_DDDD  (e.g. _0506_2307)
    m = re.match(r"^(.+?)_(\d{4}_\d{4})/?$", dirname)
    core = m.group(1) if m else dirname.rstrip("/")

    # baseline: any name that starts with "baseline" (after stripping batch prefix)
    core_stripped = re.sub(r'^(wmc|ft|eval|ada)_', '', core)
    core_nodate   = re.sub(r'_\d{4}(_\d{4})?$', '', core_stripped)  # strip _MMDD or _MMDD_HHMM
    if re.match(r'^baseline', core_nodate):
        return core_nodate, "—", 0.0

    # Strip well-known batch prefix (wmc_, ft_, eval_, etc.) for display tag
    tag_display = re.sub(r'^(wmc|ft|eval|ada)_', '', core)

    # For regex matching, also strip round-suffix _r\d+ and date suffix _MMDD
    # e.g. koleo002_0510_r2 → koleo002, hmix_k001_u03_0510_r3 → hmix_k001_u03
    match_core = re.sub(r'_r\d+$', '', tag_display)   # strip _rN
    match_core = re.sub(r'_\d{4}$', '', match_core)    # strip trailing _MMDD

    # Generic: optional_prefix _ modifier number
    mo = re.match(r"^(?:(.+)_)?(img|txt|both|wm|koleo|uni|gap|ada)(\d+)$", match_core)
    if mo:
        prefix, mod, num = mo.group(1), mo.group(2), mo.group(3)
        method_lbl, scale = _MOD[mod]
        sides  = f"{prefix}_{method_lbl}" if prefix else method_lbl
        if scale is None:
            lam = _decode_wm_lam(num)
        else:
            lam = int(num) * scale
        return tag_display, sides, lam

    return tag_display, "?", 0.0


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
    best    = complete[best_ep]
    return {
        'epoch':      last_ep,       # last completed epoch (training completeness)
        'best_epoch': best_ep,       # epoch with best i2t R@1 (primary)
        'i2t_r1':    best.get('i2t_r1'),
        'i2t_r5':    best.get('i2t_r5'),
        'i2t_r10':   best.get('i2t_r10'),
        't2i_r1':    best.get('t2i_r1'),
        't2i_r5':    best.get('t2i_r5'),
        't2i_r10':   best.get('t2i_r10'),
        'val_loss':  best.get('val_loss'),
        'scale':     last_scale,
        'bias':      last_bias,
        'history':   {ep: v for ep, v in sorted(complete.items())},
    }


# ── table ─────────────────────────────────────────────────────────────────────

def _f(v, d=4):
    return "—" if v is None else f"{v:.{d}f}"

def build_table(entries: list) -> str:
    hdr = "| 实验 | i2t R@1 | t2i R@1 | i2t R@5 | t2i R@5 | val_loss | Scale | Epoch |\n"
    sep = "|------|---------|---------|---------|---------|----------|-------|-------|\n"
    rows = []
    for tag, _, _, r in entries:
        note = " ★" if r["epoch"] is not None and r["epoch"] < 18 else ""
        rows.append(
            f"| {tag} | {_f(r['i2t_r1'])} | {_f(r['t2i_r1'])} "
            f"| {_f(r['i2t_r5'])} | {_f(r['t2i_r5'])} "
            f"| {_f(r.get('val_loss'), 4)} "
            f"| {_f(r['scale'])} | {r['best_epoch']}{note} |\n"
        )
    return hdr + sep + "".join(rows)


# ── MD injection ──────────────────────────────────────────────────────────────

def inject_md(table: str, md_path: Path = None, marker: str = None):
    path = md_path or MD_PATH
    start_tag = f"<!-- {marker or _DEFAULT_MD_MARKER}_START -->"
    end_tag   = f"<!-- {marker or _DEFAULT_MD_MARKER}_END -->"
    if not path.exists():
        logging.warning(f"[log_parser] MD not found: {path}")
        return
    md = path.read_text()
    if start_tag not in md or end_tag not in md:
        logging.warning(f"[log_parser] table markers '{start_tag}' / '{end_tag}' missing in {path}")
        return
    new_md = re.sub(
        re.escape(start_tag) + r".*?" + re.escape(end_tag),
        start_tag + "\n" + table + end_tag,
        md, flags=re.DOTALL,
    )
    if new_md != md:
        path.write_text(new_md)
        logging.info(f"[log_parser] MD updated: {path}")


# ── plots ─────────────────────────────────────────────────────────────────────

_METHOD_ORDER = {'—': 0, 'gap': 1, 'koleo': 2, 'uni': 3, 'ada': 4}

def _plot_results(entries: list, out_dir: Path):
    """Bar chart: best i2t/t2i R@1 per experiment; training curves."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # bar: sorted by tag name
    bar_entries = sorted(entries, key=lambda e: e[0])
    tags   = [e[0] for e in bar_entries]
    i2t_r1 = [e[3].get('i2t_r1') or 0 for e in bar_entries]
    t2i_r1 = [e[3].get('t2i_r1') or 0 for e in bar_entries]

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
    p = out_dir / 'r1_bar.png'
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
    p = out_dir / 'training_curves.png'
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    logging.info(f'[log_parser] {p}')


# ── JSON + MD fragment export ─────────────────────────────────────────────────

def export_results(entries: list, out_dir: Path):
    """Write results.json and results_md.md alongside the plots.

    results.json  — structured data: list of {tag, sides, lambda, metrics}
    results_md.md — ready-to-paste MD fragment: table + brief per-group summary
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── JSON ──────────────────────────────────────────────────────────────────
    records = []
    for tag, method, lam, r in entries:
        records.append({
            "tag":        tag,
            "method":     method,
            "lambda":     lam,
            "best_epoch": r.get("best_epoch"),
            "last_epoch": r.get("epoch"),
            "i2t_r1":    r.get("i2t_r1"),
            "i2t_r5":    r.get("i2t_r5"),
            "i2t_r10":   r.get("i2t_r10"),
            "t2i_r1":    r.get("t2i_r1"),
            "t2i_r5":    r.get("t2i_r5"),
            "t2i_r10":   r.get("t2i_r10"),
            "val_loss":  r.get("val_loss"),
            "scale":     r.get("scale"),
            "bias":      r.get("bias"),
        })
    json_path = out_dir / "results.json"
    with open(json_path, "w") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)
    logging.info(f"[log_parser] JSON → {json_path}")

    # ── MD fragment (table + compact summary) ─────────────────────────────────
    table = build_table(entries)

    # Top-5 by i2t_r1 (best epoch)
    baseline = next((r for tag, _, _, r in entries if tag == 'baseline'), None)
    base_i2t = baseline["i2t_r1"] if baseline else None
    base_t2i = baseline["t2i_r1"] if baseline else None

    def _delta(v, base):
        if v is None or base is None or base == 0: return "—"
        return f"{(v - base) / base * 100:+.1f}%"

    ranked = sorted(entries, key=lambda e: e[3].get("i2t_r1") or 0, reverse=True)
    top5_lines = [
        f"| {tag} | {_f(r['i2t_r1'])} ({_delta(r['i2t_r1'], base_i2t)}) "
        f"| {_f(r['t2i_r1'])} ({_delta(r['t2i_r1'], base_t2i)}) |"
        for tag, _, _, r in ranked[:5]
    ]
    top5_block = (
        "### Top-5 by best i2t R@1\n\n"
        "| 实验 | i2t R@1 (best) | t2i R@1 (best) |\n"
        "|------|---------------|----------------|\n"
        + "\n".join(top5_lines) + "\n"
    )

    md_fragment = (
        f"<!-- auto-generated by log_parser on {__import__('datetime').date.today()} -->\n\n"
        + table + "\n"
        + top5_block
    )
    md_path = out_dir / "results_md.md"
    md_path.write_text(md_fragment)
    logging.info(f"[log_parser] MD fragment → {md_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def collect_entries(prefix: str = "", logs_dir: Path = None) -> list:
    root = logs_dir or LOGS_DIR
    entries = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if prefix and not d.name.startswith(prefix):
            continue
        tag, sides, lam = parse_tag(d.name)
        if tag is None:
            continue
        res = parse_log(d / "out.log")
        if res is None:
            logging.info(f"[log_parser] SKIP {d.name} (no eval)")
            continue
        entries.append((tag, sides, lam, res))
    entries.sort(key=lambda x: (_METHOD_ORDER.get(x[1], 99), x[2], x[0]))
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
    ap.add_argument("--prefix",    default="",
                    help="log dir prefix filter; empty = all dirs (default: all)")
    ap.add_argument("--logs-dir",  default=None, metavar="DIR",
                    help="root directory containing experiment log dirs "
                         "(default: logs/)")
    ap.add_argument("--no-plot",   action="store_true", help="skip plot generation")
    ap.add_argument("--plot-dir",  default=None,
                    help="output dir for plots; auto-derived from --logs-dir / "
                         "--prefix when omitted")
    ap.add_argument("--inject-md", action="store_true",
                    help="inject results table into a markdown file "
                         "(default: off)")
    ap.add_argument("--md-path",   default=None, metavar="MD",
                    help="path to the target .md file for --inject-md "
                         f"(default: {MD_PATH})")
    ap.add_argument("--md-marker", default=None, metavar="MARKER",
                    help="marker name without angle-brackets, e.g. RESULTS_WMC → "
                         "<!-- RESULTS_WMC_START/END -->  "
                         f"(default: {_DEFAULT_MD_MARKER})")
    ap.add_argument("--json",      action="store_true",
                    help="also write results.json + results_md.md into the plot dir")
    # single-experiment export
    ap.add_argument("--single",    default=None, metavar="LOGDIR",
                    help="export per-epoch CSV for one logdir (skips global scan)")
    ap.add_argument("--out",       default=None, metavar="CSV",
                    help="output CSV path for --single (required with --single)")
    ap.add_argument("--force",     action="store_true",
                    help="overwrite existing output files (default: skip if present)")
    args = ap.parse_args()

    if args.single:
        logdir = Path(args.single)
        out    = Path(args.out) if args.out else logdir / "probe" / "plots" / "training_metrics.csv"
        if out.exists() and not args.force:
            logging.info(f'[log_parser] SKIP {logdir.name} (sentinel exists, pass --force to rerun)')
            return
        out.parent.mkdir(parents=True, exist_ok=True)
        ok = export_single(logdir, out)
        if ok:
            logging.info(f"[log_parser] {logdir.name} → {out}")
        else:
            logging.warning(f"[log_parser] no eval data in {logdir / 'out.log'}")
        return

    logs_dir = Path(args.logs_dir) if args.logs_dir else None

    # Auto-derive plot_dir:
    #   --logs-dir provided → plots/  inside that dir
    #   --prefix provided   → analysis/research/plots/<prefix stripped _>
    #   neither             → analysis/research/plots
    if args.plot_dir:
        plot_dir = Path(args.plot_dir)
    elif logs_dir is not None:
        plot_dir = logs_dir / "plots"
    elif args.prefix:
        slug = args.prefix.strip("_")
        plot_dir = Path("analysis/research/plots") / slug
    else:
        plot_dir = Path("analysis/research/plots")

    entries = collect_entries(args.prefix, logs_dir)
    if not entries:
        logging.warning(f"[log_parser] no logs found"
                        + (f" with prefix '{args.prefix}'" if args.prefix else "")
                        + (f" in '{logs_dir}'" if logs_dir else ""))
        return

    table = build_table(entries)
    print(table)

    if not args.no_plot:
        _plot_results(entries, plot_dir)

    if args.json:
        export_results(entries, plot_dir)

    if args.inject_md:
        md_path = Path(args.md_path) if args.md_path else None
        inject_md(table, md_path=md_path, marker=args.md_marker)

    logging.info(f"[log_parser] {len(entries)} experiments  plots→{plot_dir}")


if __name__ == "__main__":
    main()
