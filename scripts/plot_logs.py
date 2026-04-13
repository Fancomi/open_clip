#!/usr/bin/env python3
"""OpenCLIP训练日志可视化脚本"""

import re
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import colorsys


def parse_log(log_path: str) -> dict:
    """解析单个日志文件"""
    result = {
        'train_steps': [], 'train_loss': [],
        'eval_epochs': [],
        'val_loss': [],
        'i2t_r1': [], 'i2t_r5': [], 'i2t_r10': [],
        't2i_r1': [], 't2i_r5': [], 't2i_r10': [],
        'i2t_mean_rank': [], 'i2t_median_rank': [],
        't2i_mean_rank': [], 't2i_median_rank': [],
    }

    with open(log_path, 'r') as f:
        step = 0
        for line in f:
            # 训练loss: 支持Siglip_loss和Contrastive_loss两种格式
            m = re.search(r'(?:Siglip_loss|Contrastive_loss):\s*[\d.]+\s*\(([\d.]+)\)', line)
            if m:
                result['train_loss'].append(float(m.group(1)))
                result['train_steps'].append(step)
                step += 1

            # 验证指标: 支持siglip_val_loss和clip_val_loss
            m = re.search(
                r'Eval Epoch:\s*(\d+).*?image_to_text_mean_rank:\s*([\d.]+).*?'
                r'image_to_text_median_rank:\s*([\d.]+).*?'
                r'image_to_text_R@1:\s*([\d.]+).*?image_to_text_R@5:\s*([\d.]+).*?image_to_text_R@10:\s*([\d.]+).*?'
                r'text_to_image_mean_rank:\s*([\d.]+).*?text_to_image_median_rank:\s*([\d.]+).*?'
                r'text_to_image_R@1:\s*([\d.]+).*?text_to_image_R@5:\s*([\d.]+).*?text_to_image_R@10:\s*([\d.]+).*?'
                r'(?:siglip_val_loss|clip_val_loss):\s*([\d.]+)', line)
            if m:
                result['eval_epochs'].append(int(m.group(1)))
                result['i2t_mean_rank'].append(float(m.group(2)))
                result['i2t_median_rank'].append(float(m.group(3)))
                result['i2t_r1'].append(float(m.group(4)))
                result['i2t_r5'].append(float(m.group(5)))
                result['i2t_r10'].append(float(m.group(6)))
                result['t2i_mean_rank'].append(float(m.group(7)))
                result['t2i_median_rank'].append(float(m.group(8)))
                result['t2i_r1'].append(float(m.group(9)))
                result['t2i_r5'].append(float(m.group(10)))
                result['t2i_r10'].append(float(m.group(11)))
                result['val_loss'].append(float(m.group(12)))

    return result


def main():
    parser = argparse.ArgumentParser(description='OpenCLIP日志可视化')
    parser.add_argument('--logs', nargs='+', required=True, help='日志文件列表')
    parser.add_argument('--names', nargs='+', help='实验名称列表')
    parser.add_argument('--output', '-o', default='logs', help='输出目录')
    args = parser.parse_args()

    if args.names and len(args.names) != len(args.logs):
        print(f'错误: 名称数量({len(args.names)})与日志数量({len(args.logs)})不匹配')
        return

    all_data = [parse_log(p) for p in args.logs]
    names = args.names if args.names else [Path(p).parent.name for p in args.logs]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    axes = axes.flatten()

    # 单指标子图: (key, title, xlabel, ylabel, direction, use_train_steps)
    single_metrics = [
        ('train_loss', 'Train Loss', 'Step', 'Loss', '↓', True),
        ('val_loss', 'Val Loss', 'Epoch', 'Loss', '↓', False),
        ('i2t_r1', 'I2T R@1', 'Epoch', 'R@1', '↑', False),
        ('i2t_r5', 'I2T R@5', 'Epoch', 'R@5', '↑', False),
        ('i2t_r10', 'I2T R@10', 'Epoch', 'R@10', '↑', False),
        ('t2i_r1', 'T2I R@1', 'Epoch', 'R@1', '↑', False),
        ('t2i_r5', 'T2I R@5', 'Epoch', 'R@5', '↑', False),
        ('t2i_r10', 'T2I R@10', 'Epoch', 'R@10', '↑', False),
        ('i2t_mean_rank', 'I2T Mean Rank', 'Epoch', 'Rank', '↓', False),
        ('t2i_mean_rank', 'T2I Mean Rank', 'Epoch', 'Rank', '↓', False),
        ('i2t_median_rank', 'I2T Median Rank', 'Epoch', 'Rank', '↓', False),
        ('t2i_median_rank', 'T2I Median Rank', 'Epoch', 'Rank', '↓', False),
    ]

    # 精选色相, 避开难看的黄绿区间; 每对(j//2)同色相, 偶数浅色虚线、奇数深色实线
    nice_hues = [0.0, 0.08, 0.58, 0.78, 0.15, 0.35, 0.48, 0.68]
    n = len(names)
    colors = []
    for j in range(n):
        h = nice_hues[(j // 2) % len(nice_hues)]
        l = 0.60 if j % 2 == 0 else 0.30
        colors.append(colorsys.hls_to_rgb(h, l, 0.80))

    for i, (key, title, xlabel, ylabel, direction, use_train) in enumerate(single_metrics):
        ax = axes[i]
        for idx, (data, name) in enumerate(zip(all_data, names)):
            ls = '--' if idx % 2 == 0 else '-'
            if use_train:
                ax.plot(data['train_steps'], data[key], label=name, alpha=0.85,
                        color=colors[idx], linestyle=ls)
            else:
                ax.plot(data['eval_epochs'], data[key], 'o-', label=name, markersize=5,
                        color=colors[idx], linestyle=ls)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title} ({direction})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = out_dir / 'all_metrics.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Saved: {save_path}')
    print(f'共处理 {len(args.logs)} 个日志文件')


if __name__ == '__main__':
    main()
