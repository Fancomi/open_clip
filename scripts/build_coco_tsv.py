#!/usr/bin/env python3
"""
将 COCO captions JSON 转换为训练/评估所需的 TSV 文件。

输出 TSV 格式: filepath\tcaption（tab 分隔，首行为 header）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
文件说明
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
clip_train_dedup.tsv
  训练集，来自 captions_train2014.json。
  每张图随机保留 1 条 caption（seed=42），最大化图像覆盖率。
  82,783 对。

karpathy_5cap.tsv   ← 主评估文件，与论文数值对齐
  官方 Karpathy test split，5000 张图 × 5 条 caption = 25,000 行。
  来源：Salesforce LAVIS coco_karpathy_test.json，全部来自 val2014。
  评估协议：5K Karpathy test，R@1 = 5条中至少1条命中（CLIP-style binary）。
  复现结果（本项目实测）：
    PE-Core-B-16:     T2I R@1=50.2  I2T R@1=71.1   论文报告 T2I=50.9 ✅
    ViT-B-16-SigLIP2: T2I R@1=53.2  I2T R@1=69.4   论文报告 I2T≈68.9 ✅
  eval 传参：--val-num-captions-per-image 5

karpathy_1cap.tsv
  同一批 5000 张图，每图取第 1 条 caption，用于快速 1:1 对比。
  eval 传参：--val-num-captions-per-image 1（默认）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
数据获取步骤（一次性）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 下载 Karpathy split JSON（~1.6MB）
COCO_ANN=/path/to/coco/annotations
wget -q "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json" \\
     -O ${COCO_ANN}/coco_karpathy_test.json

# 2. 生成所有 TSV（需要 COCO images/train2014 和 images/val2014）
python3 scripts/build_coco_tsv.py \\
    --coco-root /path/to/coco \\
    --karpathy-json ${COCO_ANN}/coco_karpathy_test.json

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
注意事项
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- caption 内嵌换行符会拆行导致 TSV 格式错误（COCO 中存在此类数据），
  已通过 _clean() 统一替换为空格。
- Karpathy 5cap 每图严格 5 条（原始数据恰好 5 条，截断以防万一），
  保证 total_rows = n_images × 5，避免 all_img[::5] 越界。
"""
import json
import random
import argparse
from pathlib import Path


def _clean(cap: str) -> str:
    """去除首尾空白，替换内嵌换行符，防止 TSV 行分裂。"""
    return cap.strip().replace('\n', ' ').replace('\r', ' ')


def build_train_tsv(ann_file: Path, img_dir: Path, out_file: Path, seed: int = 42):
    """训练集：每图随机保留 1 条 caption。"""
    data = json.loads(ann_file.read_text())
    id2name = {img["id"]: img["file_name"] for img in data["images"]}
    id2caps: dict[int, list[str]] = {}
    for ann in data["annotations"]:
        id2caps.setdefault(ann["image_id"], []).append(ann["caption"])

    rng = random.Random(seed)
    rows, missing = [], 0
    for img_id, caps in id2caps.items():
        img_path = img_dir / id2name[img_id]
        if not img_path.exists():
            missing += 1
            continue
        rows.append(f"{img_path}\t{_clean(rng.choice(caps))}")

    out_file.write_text("filepath\tcaption\n" + "\n".join(rows) + "\n")
    print(f"[train] {len(rows)} pairs -> {out_file}  (skipped {missing})")


def build_tsv_karpathy(karpathy_json: Path, coco_root: Path, out_1cap: Path, out_5cap: Path):
    """官方 Karpathy test split：5000 张图，每图 5 条 caption。

    JSON 格式（Salesforce LAVIS coco_karpathy_test.json）：
      [{"image": "val2014/COCO_val2014_XXXXXX.jpg", "caption": ["cap1",...]}, ...]
    所有图像均来自 val2014，无需 train2014。
    """
    data = json.loads(karpathy_json.read_text())

    rows_1cap, rows_5cap, missing = [], [], 0
    for item in data:
        img_path = coco_root / "images" / item["image"]
        if not img_path.exists():
            missing += 1
            continue
        caps = item["caption"]
        rows_1cap.append(f"{img_path}\t{_clean(caps[0])}")
        for cap in caps[:5]:   # 严格 5 条，防止极少数图有 >5 条
            rows_5cap.append(f"{img_path}\t{_clean(cap)}")

    n_img = len(rows_1cap)
    assert len(rows_5cap) == n_img * 5, \
        f"Expected {n_img*5} rows, got {len(rows_5cap)} — some images have != 5 captions"

    out_1cap.write_text("filepath\tcaption\n" + "\n".join(rows_1cap) + "\n")
    out_5cap.write_text("filepath\tcaption\n" + "\n".join(rows_5cap) + "\n")
    print(f"[karpathy 1cap] {n_img} images -> {out_1cap}  (skipped {missing})")
    print(f"[karpathy 5cap] {n_img} images × 5 = {len(rows_5cap)} pairs -> {out_5cap}")


def main():
    parser = argparse.ArgumentParser(description="Build COCO TSV for SigLIP training/eval")
    parser.add_argument("--coco-root", default="/root/paddlejob/workspace/env_run/penghaotian/datas/coco")
    parser.add_argument("--out-dir",   default=None, help="输出目录，默认 <coco-root>/annotations")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--karpathy-json",
                        default="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations/coco_karpathy_test.json",
                        help="coco_karpathy_test.json 路径（Salesforce LAVIS 格式）")
    args = parser.parse_args()

    coco = Path(args.coco_root)
    ann  = coco / "annotations"
    out  = Path(args.out_dir) if args.out_dir else ann

    # 训练集
    build_train_tsv(ann / "captions_train2014.json", coco / "images/train2014",
                    out / "clip_train_dedup.tsv", seed=args.seed)

    # Karpathy 官方 split（eval 标准）
    build_tsv_karpathy(
        Path(args.karpathy_json), coco,
        out / "karpathy_1cap.tsv",
        out / "karpathy_5cap.tsv",
    )


if __name__ == "__main__":
    main()
