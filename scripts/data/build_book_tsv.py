#!/usr/bin/env python3
"""
将 COCO 格式 JSON（captions_train.json / captions_val.json）转为 open_clip 训练用 TSV。

输出 TSV 格式: filepath\tcaption（tab 分隔，首行 header）

用法:
    python3 scripts/data/build_book_tsv.py --data-root /path/to/book_20260507

数据目录结构（与 COCO 一致）:
    <data-root>/
        annotations/
            annotations/captions_train.json
            annotations/captions_val.json
            images/xxx.jpg
        → 输出: annotations/train.tsv, annotations/val.tsv
"""
import json
import re
import argparse
from pathlib import Path

MAX_CAPTION_CHARS = 1024  # 覆盖 SigLIP2 英文容量(~250 chars)，且远低于 pandas 4KB buffer 限制


def _clean(cap: str, max_chars: int = MAX_CAPTION_CHARS) -> str:
    """清理 caption：去控制字符/引号/tab，截断到安全长度。"""
    return re.sub(r'[\t\n\r\x0b\x0c"\\]+', ' ', cap)[:max_chars]


def build_tsv(json_path: Path, img_dir: Path, out_path: Path):
    """从 COCO captions JSON 生成 TSV。"""
    data = json.loads(json_path.read_text())
    id2file = {img['id']: img['file_name'] for img in data['images']}

    rows, missing = [], 0
    for ann in data['annotations']:
        img_path = img_dir / id2file[ann['image_id']]
        if not img_path.exists():
            missing += 1
            continue
        rows.append(f"{img_path}\t{_clean(ann['caption'])}")

    out_path.write_text("filepath\tcaption\n" + "\n".join(rows) + "\n")
    print(f"[book] {out_path.name}: {len(rows)} pairs (skipped {missing})")


def main():
    parser = argparse.ArgumentParser(description="Build book TSV for training/eval")
    parser.add_argument("--data-root",
                        default="/root/paddlejob/workspace/env_run/penghaotian/datas/book_20260507")
    args = parser.parse_args()

    root = Path(args.data_root)
    ann_dir = root / "annotations"
    img_dir = ann_dir / "images"
    json_dir = ann_dir / "annotations"

    for split in ("train", "val"):
        json_path = json_dir / f"captions_{split}.json"
        out_path = ann_dir / f"{split}.tsv"
        if not json_path.exists():
            print(f"[book] SKIP {split}: {json_path} not found")
            continue
        build_tsv(json_path, img_dir, out_path)


if __name__ == "__main__":
    main()
