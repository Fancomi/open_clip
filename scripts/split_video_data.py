#!/usr/bin/env python3
"""视频数据划分: 特征提取 → 去重 → FPS 划分 train/eval → 输出 split.json

用法:
    python scripts/split_video_data.py \
        --data-root /path/to/muscle_wiki \
        --model PE-Core-B-16 \
        --pretrained /path/to/open_clip_model.safetensors \
        --eval-ratio 0.15 --dedup-thresh 0.95
"""

import argparse, base64, io, json, logging, os, sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from open_clip import create_model_and_transforms, get_tokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
log = logging.getLogger(__name__)

VIEWS = ("front", "side")
CAPTION_KEY = "category_1_visual_description"


# ── 扫描数据 ────────────────────────────────────────────────────────────────
def scan_samples(root, max_side=768):
    """返回 list[dict] : {b64_path, caption, exercise_key}"""
    samples = []
    for dirpath, _, filenames in os.walk(root):
        for view in VIEWS:
            aug_name = f"augment_{view}_cn.json"
            if aug_name not in filenames:
                continue
            b64_path = os.path.join(dirpath, f"frames_{max_side}p", f"{view}.b64")
            if not os.path.isfile(b64_path):
                continue
            with open(os.path.join(dirpath, aug_name)) as f:
                caption = json.load(f).get(CAPTION_KEY, "")
            if not caption:
                continue
            # exercise_key: muscle/exercise (去掉 gender 前缀)
            rel = os.path.relpath(dirpath, root)  # gender/muscle/exercise
            parts = rel.replace("\\", "/").split("/")
            exercise_key = "/".join(parts[1:]) if len(parts) >= 3 else rel
            samples.append(dict(
                b64_path=b64_path, caption=caption,
                exercise_key=f"{exercise_key}/{view}",
            ))
    log.info(f"扫描完成: {len(samples)} 样本")
    return samples


# ── 特征提取 ────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_features(samples, model, preprocess, device, batch_size=64):
    """对每个视频取所有帧平均池化 → L2 normalized feature vector."""
    model.eval()
    features = []
    for s in tqdm(samples, desc="特征提取"):
        lines = open(s["b64_path"]).read().splitlines()
        imgs = []
        for line in lines:
            img = Image.open(io.BytesIO(base64.b64decode(line))).convert("RGB")
            imgs.append(preprocess(img))
        # 分 batch 前向
        all_feats = []
        for i in range(0, len(imgs), batch_size):
            batch = torch.stack(imgs[i:i + batch_size]).to(device)
            feat = model.encode_image(batch)  # [B, D]
            all_feats.append(feat)
        mean_feat = torch.cat(all_feats).mean(dim=0)
        features.append(F.normalize(mean_feat, dim=0))
    return torch.stack(features)  # [N, D]


# ── 去重 ────────────────────────────────────────────────────────────────────
def deduplicate(samples, features, thresh=0.95):
    """贪心去重: 按 exercise_key 聚类，cosine > thresh 的只保留首个."""
    N = len(samples)
    removed = set()
    # 按 exercise_key 分组，组内不同 gender 可能重复
    from collections import defaultdict
    groups = defaultdict(list)
    for i, s in enumerate(samples):
        groups[s["exercise_key"]].append(i)

    # 组内 cosine 检查
    for key, idxs in groups.items():
        if len(idxs) < 2:
            continue
        for j in range(1, len(idxs)):
            if idxs[j] in removed:
                continue
            sim = (features[idxs[0]] @ features[idxs[j]]).item()
            if sim > thresh:
                removed.add(idxs[j])

    # 跨组高相似度也去重（不同动作但视觉极近似）
    keep_mask = torch.ones(N, dtype=torch.bool)
    keep_mask[list(removed)] = False
    keep_idxs = keep_mask.nonzero(as_tuple=True)[0].tolist()
    kept_feats = features[keep_idxs]
    sim_matrix = kept_feats @ kept_feats.T
    sim_matrix.fill_diagonal_(0)
    cross_dups = set()
    for i in range(len(keep_idxs)):
        if i in cross_dups:
            continue
        for j in range(i + 1, len(keep_idxs)):
            if j in cross_dups:
                continue
            if sim_matrix[i, j] > thresh:
                cross_dups.add(j)
    # 映射回原始索引
    for local_j in cross_dups:
        removed.add(keep_idxs[local_j])

    log.info(f"去重: {len(removed)} 个样本被移除 (阈值={thresh})")
    keep = [i for i in range(N) if i not in removed]
    return keep


# ── FPS 最远点采样 ──────────────────────────────────────────────────────────
def farthest_point_sampling(features, n_select):
    """在 L2-normalized 特征上做 FPS，返回选中的索引列表."""
    N = features.shape[0]
    device = features.device
    # 用 cosine distance = 1 - sim
    selected = [torch.randint(N, (1,)).item()]
    min_dist = 1.0 - (features @ features[selected[0]])  # [N]
    for _ in range(n_select - 1):
        idx = min_dist.argmax().item()
        selected.append(idx)
        new_dist = 1.0 - (features @ features[idx])
        min_dist = torch.minimum(min_dist, new_dist)
    return selected


# ── 主流程 ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="视频数据划分")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model", default="PE-Core-B-16")
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--eval-ratio", type=float, default=0.15)
    parser.add_argument("--dedup-thresh", type=float, default=0.95)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    # 1. 扫描
    samples = scan_samples(args.data_root)

    # 2. 特征提取
    model, _, preprocess = create_model_and_transforms(args.model, pretrained=args.pretrained)
    model = model.to(args.device)
    features = extract_features(samples, model, preprocess, args.device)
    del model
    torch.cuda.empty_cache()

    # 3. 去重
    keep_idxs = deduplicate(samples, features, args.dedup_thresh)
    kept_samples = [samples[i] for i in keep_idxs]
    kept_features = features[keep_idxs]
    log.info(f"去重后: {len(kept_samples)} 样本")

    # 4. FPS 划分
    n_eval = max(1, int(len(kept_samples) * args.eval_ratio))
    eval_local_idxs = farthest_point_sampling(kept_features, n_eval)
    eval_set_local = set(eval_local_idxs)
    train_samples = [kept_samples[i] for i in range(len(kept_samples)) if i not in eval_set_local]
    eval_samples = [kept_samples[i] for i in eval_local_idxs]

    # 5. 输出
    def to_records(lst):
        return [{"b64_path": s["b64_path"], "caption": s["caption"]} for s in lst]

    out = {
        "train": to_records(train_samples),
        "eval": to_records(eval_samples),
        "stats": {
            "total_scanned": len(samples),
            "dedup_removed": len(samples) - len(kept_samples),
            "after_dedup": len(kept_samples),
            "train": len(train_samples),
            "eval": len(eval_samples),
            "dedup_thresh": args.dedup_thresh,
            "eval_ratio": args.eval_ratio,
        }
    }
    out_path = os.path.join(args.data_root, "split.json")
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log.info(f"输出: {out_path}")
    log.info(f"统计: {json.dumps(out['stats'], indent=2)}")


if __name__ == "__main__":
    main()
