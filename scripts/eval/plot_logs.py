#!/usr/bin/env python3
"""OpenCLIP训练日志可视化脚本"""

import re
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import colorsys
import numpy as np


def smooth_ema(values: list, alpha: float = 0.9) -> list:
    """指数移动平均平滑"""
    if not values:
        return []
    smoothed, last = [], values[0]
    for v in values:
        last = alpha * last + (1 - alpha) * v
        smoothed.append(last)
    return smoothed


def parse_log(log_path: str) -> dict:
    """解析单个日志文件。
    "Train Epoch: E [Y/Z (P%)]" 中 Y/Z 已是样本数，
    全局已见样本 = (E-1)*Z + Y，eval 对齐到 epoch*Z。
    """
    result = {
        'train_seen': [],  'train_loss': [],
        'lr_seen': [],     'lr': [],
        'eval_epochs': [],
        'val_loss': [],
        'i2t_r1': [], 'i2t_r5': [], 'i2t_r10': [],
        't2i_r1': [], 't2i_r5': [], 't2i_r10': [],
        'i2t_mean_rank': [], 'i2t_median_rank': [],
        't2i_mean_rank': [], 't2i_median_rank': [],
        'samples_per_epoch': None,
    }

    with open(log_path, 'r') as f:
        for line in f:
            # 训练行：从同行 "Train Epoch: E [Y/Z]" 解析全局已见样本
            m_hdr  = re.search(r'Train Epoch:\s*(\d+)\s*\[(\d+)/(\d+)', line)
            m_loss = re.search(
                r'(?:Siglip_loss|Contrastive_loss):\s*[\d.]+\s*\(([\d.]+)\)', line)
            if m_hdr and m_loss:
                ep, y, z = int(m_hdr.group(1)), int(m_hdr.group(2)), int(m_hdr.group(3))
                result['samples_per_epoch'] = z          # 每轮总样本数
                seen = (ep - 1) * z + y                  # 全局已见样本（与eval同单位）
                result['train_loss'].append(float(m_loss.group(1)))
                result['train_seen'].append(seen)
                m_lr = re.search(r'\bLR:\s*([\d.eE+\-]+)', line)
                if m_lr:
                    result['lr'].append(float(m_lr.group(1)))
                    result['lr_seen'].append(seen)

            # 验证指标
            m_eval = re.search(
                r'Eval Epoch:\s*(\d+).*?'
                r'image_to_text_mean_rank:\s*([\d.]+).*?'
                r'image_to_text_median_rank:\s*([\d.]+).*?'
                r'image_to_text_R@1:\s*([\d.]+).*?'
                r'image_to_text_R@5:\s*([\d.]+).*?'
                r'image_to_text_R@10:\s*([\d.]+).*?'
                r'text_to_image_mean_rank:\s*([\d.]+).*?'
                r'text_to_image_median_rank:\s*([\d.]+).*?'
                r'text_to_image_R@1:\s*([\d.]+).*?'
                r'text_to_image_R@5:\s*([\d.]+).*?'
                r'text_to_image_R@10:\s*([\d.]+).*?'
                r'(?:siglip_val_loss|clip_val_loss):\s*([\d.]+)', line)
            if m_eval:
                result['eval_epochs'].append(int(m_eval.group(1)))
                result['i2t_mean_rank'].append(float(m_eval.group(2)))
                result['i2t_median_rank'].append(float(m_eval.group(3)))
                result['i2t_r1'].append(float(m_eval.group(4)))
                result['i2t_r5'].append(float(m_eval.group(5)))
                result['i2t_r10'].append(float(m_eval.group(6)))
                result['t2i_mean_rank'].append(float(m_eval.group(7)))
                result['t2i_median_rank'].append(float(m_eval.group(8)))
                result['t2i_r1'].append(float(m_eval.group(9)))
                result['t2i_r5'].append(float(m_eval.group(10)))
                result['t2i_r10'].append(float(m_eval.group(11)))
                result['val_loss'].append(float(m_eval.group(12)))

    return result


# ── 图表布局：3行×5列 ─────────────────────────────────────────────────────────
#   行0: Train Loss | Val Loss | LR | [拐点表] | (隐藏)
#   行1: I2T R@1   | I2T R@5  | I2T R@10 | I2T Mean Rank | I2T Median Rank
#   行2: T2I R@1   | T2I R@5  | T2I R@10 | T2I Mean Rank | T2I Median Rank
#
# 每格: (data_key, xs_key, title, ylabel, direction, do_smooth)
# 特殊值: 'INFLECTION' = 拐点表，None = 隐藏
GRID = [
    [
        ('train_loss',      'train_seen',  'Train Loss',      'Loss', '↓', True),
        ('val_loss',        'eval_epochs', 'Val Loss',        'Loss', '↓', True),
        ('lr',              'lr_seen',     'LR',              'LR',   '',  False),
        'INFLECTION',
        None,
    ],
    [
        ('i2t_r1',          'eval_epochs', 'I2T R@1',         'R@1',  '↑', True),
        ('i2t_r5',          'eval_epochs', 'I2T R@5',         'R@5',  '↑', True),
        ('i2t_r10',         'eval_epochs', 'I2T R@10',        'R@10', '↑', True),
        ('i2t_mean_rank',   'eval_epochs', 'I2T Mean Rank',   'Rank', '↓', True),
        ('i2t_median_rank', 'eval_epochs', 'I2T Median Rank', 'Rank', '↓', True),
    ],
    [
        ('t2i_r1',          'eval_epochs', 'T2I R@1',         'R@1',  '↑', True),
        ('t2i_r5',          'eval_epochs', 'T2I R@5',         'R@5',  '↑', True),
        ('t2i_r10',         'eval_epochs', 'T2I R@10',        'R@10', '↑', True),
        ('t2i_mean_rank',   'eval_epochs', 'T2I Mean Rank',   'Rank', '↓', True),
        ('t2i_median_rank', 'eval_epochs', 'T2I Median Rank', 'Rank', '↓', True),
    ],
]

NROWS = len(GRID)
NCOLS = max(len(row) for row in GRID)

# ↓ 指标：首次上升即为拐点
INFLECTION_METRICS = [
    ('val_loss',      'Val Loss'),
    ('i2t_mean_rank', 'I2T Mean Rank'),
    ('t2i_mean_rank', 'T2I Mean Rank'),
]

STAT_KEYS = [
    'train_loss', 'val_loss', 'lr',
    'i2t_r1', 'i2t_r5', 'i2t_r10', 'i2t_mean_rank', 'i2t_median_rank',
    't2i_r1', 't2i_r5', 't2i_r10', 't2i_mean_rank', 't2i_median_rank',
]


def make_colors(n: int) -> list:
    nice_hues = [0.0, 0.08, 0.58, 0.78, 0.15, 0.35, 0.48, 0.68]
    return [
        colorsys.hls_to_rgb(
            nice_hues[(j // 2) % len(nice_hues)],
            0.60 if j % 2 == 0 else 0.30,
            0.80,
        )
        for j in range(n)
    ]


def to_seen(raw_xs: list, xs_key: str, data: dict) -> list:
    """统一转换为 seen samples。
    train_seen / lr_seen 已是绝对样本数；eval_epochs 需乘 samples_per_epoch。
    """
    if xs_key == 'eval_epochs':
        spe = data.get('samples_per_epoch') or 1
        return [int(e) * spe for e in raw_xs]
    return list(raw_xs)


def fmt_samples(n) -> str:
    """3145728 → '3.15M'"""
    if n is None:
        return '—'
    if abs(n) >= 1e9:
        return f'{n / 1e9:.2f}B'
    if abs(n) >= 1e6:
        return f'{n / 1e6:.2f}M'
    if abs(n) >= 1e3:
        return f'{n / 1e3:.1f}K'
    return str(int(n))


def find_inflection(eval_epochs: list, values: list, eval_seen: list,
                    after_epoch: int = 2):
    """ep{after_epoch} 之后首次 ↓ 指标上升时的 seen samples，无则返回 None。"""
    pts = [(e, v, s) for e, v, s in zip(eval_epochs, values, eval_seen)
           if e > after_epoch]
    for i in range(1, len(pts)):
        if pts[i][1] > pts[i - 1][1]:
            return pts[i][2]
    return None


def compute_inflections(all_data: list, names: list) -> dict:
    out = {}
    for data, name in zip(all_data, names):
        spe = data.get('samples_per_epoch') or 1
        eval_seen = [e * spe for e in data['eval_epochs']]
        out[name] = {
            key: find_inflection(data['eval_epochs'], data.get(key, []), eval_seen)
            for key, _ in INFLECTION_METRICS
        }
    return out


def draw_inflection_table(ax, names: list, colors: list, inflections: dict) -> None:
    """在空白格中用 Table 展示拐点（seen samples）。"""
    ax.axis('off')
    ax.set_title('Inflection after ep2  (↓ metric first rises)', fontsize=8.5, pad=6)

    col_labels = [label for _, label in INFLECTION_METRICS]
    cell_text, cell_colors = [], []
    for name in names:
        row, row_c = [], []
        for key, _ in INFLECTION_METRICS:
            row.append(fmt_samples(inflections[name].get(key)))
            row_c.append('#f8f8f8')
        cell_text.append(row)
        cell_colors.append(row_c)

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=names,
        colLabels=col_labels,
        cellColours=cell_colors,
        cellLoc='center',
        loc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.15, 1.9)

    # 用实验颜色标记行 label 背景
    for i, color in enumerate(colors[:len(names)]):
        cell = tbl[i + 1, -1]           # row i+1（跳过表头）, col -1（行标签列）
        cell.set_facecolor((*color, 0.25))


def plot_metrics(all_data, names, colors, out_path: Path,
                 smoothed: bool = False, alpha: float = 0.9) -> None:
    inflections = compute_inflections(all_data, names)

    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(NCOLS * 4, NROWS * 4),
                             constrained_layout=True)

    for ri, row_def in enumerate(GRID):
        for ci in range(NCOLS):
            ax   = axes[ri][ci]
            cell = row_def[ci] if ci < len(row_def) else None

            if cell is None:
                ax.set_visible(False)
                continue

            if cell == 'INFLECTION':
                draw_inflection_table(ax, names, colors, inflections)
                continue

            key, xs_key, title, ylabel, direction, do_smooth = cell
            has_data = False

            for idx, (data, name) in enumerate(zip(all_data, names)):
                ys     = data.get(key, [])
                raw_xs = data.get(xs_key, [])
                if not ys or not raw_xs:
                    continue
                has_data = True

                ls      = '--' if idx % 2 == 0 else '-'
                xs      = to_seen(raw_xs, xs_key, data)
                apply_s = smoothed and do_smooth
                plot_ys = smooth_ema(ys, alpha) if apply_s else ys

                if xs_key != 'eval_epochs':          # step-based (dense)
                    ax.plot(xs, plot_ys, label=name, alpha=0.85,
                            color=colors[idx], linestyle=ls)
                else:                                # epoch-based (sparse)
                    marker = None if apply_s else 'o'
                    ax.plot(xs, plot_ys, linestyle=ls, marker=marker,
                            markersize=4, label=name,
                            color=colors[idx], alpha=0.85)

            if not has_data:
                ax.set_visible(False)
                continue

            # 拐点竖线（仅 val_loss / i2t_mean_rank / t2i_mean_rank）
            if key in {k for k, _ in INFLECTION_METRICS}:
                for idx2, (data2, name2) in enumerate(zip(all_data, names)):
                    x_inf = inflections[name2].get(key)
                    if x_inf is not None:
                        ax.axvline(x=x_inf, color=colors[idx2],
                                   linestyle=':', linewidth=1.5, alpha=0.8)
                        # 在竖线底部标注样本数（x=数据坐标，y=axes坐标）
                        ax.text(x_inf, 0.02, fmt_samples(x_inf),
                                transform=ax.get_xaxis_transform(),
                                fontsize=6, color=colors[idx2],
                                rotation=90, va='bottom', ha='center')

            title_str = f'{title} ({direction})' if direction else title
            if smoothed and do_smooth:
                title_str += ' (smoothed)'
            ax.set_xlabel('Seen Samples')
            ax.set_ylabel(ylabel)
            ax.set_title(title_str)
            ax.grid(True, alpha=0.3)
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

    # 所有子图共享同一图例，放在底部，避免遮挡数据
    handles, labels = next(
        (ax.get_legend_handles_labels()
         for ax in axes.flat
         if ax.get_visible() and ax.get_legend_handles_labels()[0]),
        ([], [])
    )
    if handles:
        ncol = min(len(names), 6)
        fig.legend(handles, labels, loc='lower center', ncol=ncol,
                   fontsize=8, framealpha=0.9,
                   bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure)

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


def compute_stats(all_data, names, alpha: float = 0.9) -> dict:
    inflections = compute_inflections(all_data, names)
    stats = {}
    for data, name in zip(all_data, names):
        entry = {
            'samples_per_epoch': data.get('samples_per_epoch'),
            'inflection_points': {
                key: inflections[name].get(key)
                for key, _ in INFLECTION_METRICS
            },
        }
        for key in STAT_KEYS:
            vals = data.get(key, [])
            if not vals:
                continue
            sm = smooth_ema(vals, alpha)
            entry[key] = {
                'min':            round(float(np.min(vals)),  6),
                'max':            round(float(np.max(vals)),  6),
                'mean':           round(float(np.mean(vals)), 6),
                'final':          round(float(vals[-1]),      6),
                'smoothed_min':   round(float(np.min(sm)),    6),
                'smoothed_max':   round(float(np.max(sm)),    6),
                'smoothed_final': round(float(sm[-1]),        6),
            }
        stats[name] = entry
    return stats


def main():
    parser = argparse.ArgumentParser(description='OpenCLIP日志可视化')
    parser.add_argument('--logs',  nargs='+', required=True, help='日志文件列表')
    parser.add_argument('--names', nargs='+', help='实验名称列表')
    parser.add_argument('--output', '-o', default='logs', help='输出目录')
    parser.add_argument('--smooth_alpha', type=float, default=0.9,
                        help='EMA平滑系数 (default: 0.9)')
    args = parser.parse_args()

    if args.names and len(args.names) != len(args.logs):
        print(f'错误: 名称数量({len(args.names)})与日志数量({len(args.logs)})不匹配')
        return

    all_data = [parse_log(p) for p in args.logs]
    names    = args.names if args.names else [Path(p).parent.name for p in args.logs]
    colors   = make_colors(len(names))

    for name, data in zip(names, all_data):
        spe = data.get('samples_per_epoch')
        print(f'[{name}] samples_per_epoch={spe:,}' if spe
              else f'[{name}] 未解析到 samples_per_epoch，eval x 轴将不准确')

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_metrics(all_data, names, colors,
                 out_dir / 'all_metrics.png', smoothed=False)
    plot_metrics(all_data, names, colors,
                 out_dir / 'all_metrics_smooth.png',
                 smoothed=True, alpha=args.smooth_alpha)

    stats    = compute_stats(all_data, names, alpha=args.smooth_alpha)
    json_path = out_dir / 'stats.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f'Saved: {json_path}')
    print(f'共处理 {len(args.logs)} 个日志文件')


if __name__ == '__main__':
    main()
