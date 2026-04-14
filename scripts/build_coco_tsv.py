#!/usr/bin/env python3
"""
将 COCO captions JSON 转换为 SigLIP 训练所需的 TSV 文件。

- train: 每张图随机保留 1 条 caption（去重，最大化覆盖率）
- val:   每张图取第 1 条 caption（确定性，保证一一对应）

输出 TSV 格式: filepath\tcaption
"""
import json
import random
import argparse
from pathlib import Path


def build_tsv(ann_file: Path, img_dir: Path, out_file: Path, is_train: bool, seed: int = 42):
    data = json.loads(ann_file.read_text())

    # image_id -> file_name
    id2name = {img["id"]: img["file_name"] for img in data["images"]}

    # image_id -> [captions]
    id2caps: dict[int, list[str]] = {}
    for ann in data["annotations"]:
        id2caps.setdefault(ann["image_id"], []).append(ann["caption"])

    rng = random.Random(seed)
    rows = []
    missing = 0
    for img_id, caps in id2caps.items():
        img_path = img_dir / id2name[img_id]
        if not img_path.exists():
            missing += 1
            continue
        caption = rng.choice(caps) if is_train else caps[0]
        rows.append(f"{img_path}\t{caption}")

    out_file.write_text("filepath\tcaption\n" + "\n".join(rows) + "\n")
    tag = "train(dedup)" if is_train else "val"
    print(f"[{tag}] {len(rows)} pairs written -> {out_file}  (skipped {missing} missing images)")


def main():
    parser = argparse.ArgumentParser(description="Build COCO TSV for SigLIP training")
    parser.add_argument("--coco-root", default="/root/paddlejob/workspace/env_run/penghaotian/datas/coco")
    parser.add_argument("--out-dir",   default=None, help="default: <coco-root>/annotations")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    coco = Path(args.coco_root)
    ann  = coco / "annotations"
    out  = Path(args.out_dir) if args.out_dir else ann

    splits = [
        (ann / "captions_train2014.json", coco / "images/train2014", out / "clip_train_dedup.tsv", True),
        (ann / "captions_val2014.json",   coco / "images/val2014",   out / "clip_val.tsv",         False),
    ]
    for ann_file, img_dir, out_file, is_train in splits:
        build_tsv(ann_file, img_dir, out_file, is_train, args.seed)


if __name__ == "__main__":
    main()
