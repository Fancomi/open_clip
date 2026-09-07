#!/usr/bin/env python3
"""用开源权重复现官方公开指标 —— 验证本项目评测管线的实现正确性。

原理：拿"官方公开过指标数字"的外部权重（openai CLIP B/16、SigLIP2 B/16、
DFN5B-CLIP H/14、PE-Core-B/16），用**本仓库的评测函数本体**（直接 import
eval_standard / eval_knn_probe 的实现）跑 IN-1k zero-shot、COCO 检索、
IN-1k k-NN，对照官方数字。若对上 → 我们的协议实现无 hidden bug；
若对不上 → 差值就是本仓库实现与业界协议的偏离量。

外部模型用 standard neg-mode（业界模型没在 projective 下训练）。
"""
import argparse
import os
import json
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "src"))

import open_clip

MODELS = {
    # key: (open_clip model name, 权重路径 or 预训练 tag, force_context_length)
    "clip_b16_openai": ("ViT-B-16-quickgelu",
                        "/root/paddlejob/workspace/env_run/penghaotian/models/timm/ViT-B-16-openai/ViT-B-16_sd.pt",
                        None),
    "siglip2_b16": ("ViT-B-16-SigLIP2",
                    "/root/paddlejob/workspace/env_run/penghaotian/models/timm/ViT-B-16-SigLIP2/open_clip_pytorch_model.bin",
                    None),
    "dfn5b_h14": ("ViT-H-14-378-quickgelu",
                  "/root/paddlejob/workspace/env_run/penghaotian/models/DFN5B-CLIP-ViT-H-14-378/open_clip_pytorch_model.bin",
                  None),  # 官方 eval_results.jsonl: IN-1k 84.218 (84.22)
    "pe_core_b16": ("PE-Core-B-16",
                    "/root/paddlejob/workspace/env_run/penghaotian/models/timm/PE-Core-B-16/open_clip_pytorch_model.bin",
                    32),  # PE 官方 ctx=32；本 fork 的 _pecfg 默认 256 是为自家训练改的，官方权重必须显式 32
}


def _strip_meta(sd):
    # openai JIT 提取的 sd 带 input_resolution/context_length/vocab_size 三个标量元数据
    return {k: v for k, v in sd.items()
            if k not in ("input_resolution", "context_length", "vocab_size")}


def _official_preprocess(weights):
    """读权重旁边的 open_clip_config.json 拿官方 preprocess_cfg。

    ⚠️ 这是本脚本最重要的一步。`create_model_and_transforms(pretrained=<本地路径>)`
    **拿不到** hf_hub 上的 preprocess_cfg，会静默回落到 open_clip 默认
    （openai CLIP 的 mean/std + shortest-resize + centercrop）。
    实测代价：SigLIP2（要 0.5/0.5 + squash）因此掉 16.9 点、PE-Core 掉 1.1 点，
    而 openai CLIP 恰好就是默认值所以 0 影响 —— 一个只在"非 CLIP 系"模型上
    发作的静默错配。
    """
    cfg_path = Path(weights).parent / "open_clip_config.json"
    if not cfg_path.exists():
        return {}
    pc = json.loads(cfg_path.read_text()).get("preprocess_cfg", {})
    out = {}
    if "mean" in pc:
        out["image_mean"] = tuple(pc["mean"])
    if "std" in pc:
        out["image_std"] = tuple(pc["std"])
    if "interpolation" in pc:
        out["image_interpolation"] = pc["interpolation"]
    if "resize_mode" in pc:
        out["image_resize_mode"] = pc["resize_mode"]
    return out


def load_external(name, device):
    mname, weights, fcl = MODELS[name]
    kw = _official_preprocess(weights)
    if fcl:
        kw["force_context_length"] = fcl
    model, _, val_tr = open_clip.create_model_and_transforms(
        mname, pretrained=weights, **kw)
    tok = open_clip.get_tokenizer(mname)
    if fcl:
        # PE 官方 ctx=32；本 fork 的 _pecfg 默认 256（为自家训练改的），tokenizer 必须对齐
        tok.context_length = fcl
    model = model.to(device).eval()
    if device.startswith("cuda") and os.environ.get("VERIFY_FP16", "1") == "1":
        model = model.half()  # VERIFY_FP16=0 可关（fp32 对照用）
    print(f"[{name}] loaded  preprocess={kw}")
    print(f"[{name}] val_tr={val_tr}")
    return model, tok, val_tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="clip_b16_openai,siglip2_b16")
    ap.add_argument("--tasks", default="in1k,coco,knn")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from eval_standard import eval_imagenet, eval_coco_retrieval
    from eval_knn_probe import extract_feats, knn_accuracy
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ek", _HERE / "eval_knn_probe.py")
    ek = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ek)

    results = {}
    for name in args.models.split(","):
        name = name.strip()
        if not name:
            continue
        device = args.device
        model, tok, val_tr = load_external(name, device)
        r = {}
        if "in1k" in args.tasks:
            t0 = time.time()
            r["in1k"] = eval_imagenet(model, tok, val_tr, device,
                                      neg_mode="standard")
            print(f"[{name}] IN-1k done in {time.time()-t0:.0f}s")
        if "coco" in args.tasks:
            t0 = time.time()
            r["coco"] = eval_coco_retrieval(model, tok, val_tr, device,
                                            neg_mode="standard")
            print(f"[{name}] COCO done in {time.time()-t0:.0f}s")
        if "knn" in args.tasks:
            t0 = time.time()
            bb, proj, labels = extract_feats(model, val_tr, device, 1000, 50,
                                             num_workers=12)
            if bb is not None:
                r["knn_bb"] = knn_accuracy(bb, labels, k=20, device=device)
            r["knn_proj"] = knn_accuracy(proj, labels, k=20, device=device)
            print(f"[{name}] k-NN bb={r.get('knn_bb', 'n/a')} "
                  f"proj={r['knn_proj']:.4f} ({time.time()-t0:.0f}s)")
        results[name] = r
        print(f"\n===== {name} 汇总 =====")
        print(json.dumps(r, indent=2, ensure_ascii=False))

    out = Path("/tmp/verify_open_weights.json")
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
