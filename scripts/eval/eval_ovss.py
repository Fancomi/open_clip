#!/usr/bin/env python3
"""开放词表语义分割（OVSS）mIoU —— 本项目第一个**局部表征**硬指标。

为什么需要它
------------
现有四个评测（COCO 5cap 检索 / IN-1k zero-shot / IN-1k k-NN / Urban-1k）全部是
**全局**表征口径：一张图 → 一个向量 → 比一次。而区域-短语监督改的恰恰是
**局部**表征（`roi_align` 出来的 patch 池化特征）。也就是说这条线此前一直在用
全局指标间接推断"图像塔的空间可区分性变好了"，从未直接测过。

mIoU 是这件事的直接测量：每个 patch 自己去和类名比，答对了才算。

口径声明（★必读，混比会得到错误结论★）
------------------------------------------
1. **dense 读出路径与训练时的区域分支完全一致**（`--dense-mode penult`，默认）：
   `trunk.forward_intermediates(indices=[-2], norm=True)` 取**倒数第二层** patch map
   → 每个位置过 `visual.head` 投到 1024d 文本空间 → 与类名嵌入比。
   这正是 `model.py:_roi_features` 的路径（只是把 RoIAlign 换成逐 patch），
   所以这个数字测的就是"区域损失直接优化的那个东西泛化得怎么样"。
   `--dense-mode last` 走最后一层，作为对照（最后一层已被全图对比损失拉平，
   预期更差 —— 这也是 FG-CLIP 取 `hidden_states[-2]` 的理由）。
2. **`--neg-mode` 必须与训练配方一致**，同 `eval_standard.py`。E 配方 = `projective`，
   相似度按 `|cos|` 排序。口径错配会让指标塌掉（见 `eval_protocol.md`）。
3. **主口径是 VOC-20**（只在 20 个前景类上算，背景像素当 ignore）——
   它没有任何阈值超参，是 SCLIP / ClearCLIP / NACLIP 系列的标准 benchmark 之一。
   VOC-21（含背景）需要一个背景阈值，本脚本对若干阈值都报一遍，不指定"官方值"，
   因为那个值一变排序就可能变。
4. **不做任何推理期改造**（不含 ClearCLIP 去残差 / SCLIP 相关自注意力 /
   NACLIP 邻域注意力）。那些改造是为"没有 dense 监督的原版 CLIP"救急的；
   本项目的问题是"区域监督到底有没有学到空间可区分性"，
   加了改造就分不清是改造的功劳还是训练的功劳。改造留作后续对照项。
5. 滑窗 224×224 / stride 112，与训练分辨率严格一致（避免 RoPE 外推带来的混淆项）。

用法
----
  python scripts/eval/eval_ovss.py --ckpt logs/.../epoch_10.pt --tag regw2.0
  python scripts/eval/eval_ovss.py --ckpt ... --tag foo --limit 50   # 冒烟
  python scripts/eval/eval_ovss.py --ckpt ... --tag foo --dense-mode last
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

VOC_ROOT = Path("/root/paddlejob/workspace/env_run/penghaotian/datas/voc2012/"
                "VOCdevkit/VOC2012")

# VOC2012 调色板标签：0=背景，1..20=前景，255=void（物体轮廓，官方要求忽略）
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "dining table", "dog", "horse", "motorbike", "person",
    "potted plant", "sheep", "sofa", "train", "tv monitor",
]


def load_model(ckpt_path, device):
    """与 eval_knn_probe / eval_standard 完全相同的加载路径，保证可比。"""
    from open_clip import create_model_and_transforms, get_tokenizer
    from open_clip.model import CLIPLeJEPA

    base, _, val_tr = create_model_and_transforms(
        "PE-Core-B-16-dinov3", "", precision="fp32", device="cpu",
        output_dict=True, force_context_length=256)
    model = CLIPLeJEPA(clip_model=base, sigreg_target="cls", output_dict=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  加载 {Path(ckpt_path).name}: missing={len(missing)} "
          f"unexpected={len(unexpected)}", flush=True)
    model = model.to(device).eval()
    if device == "cuda":
        model = model.half()
    return model, get_tokenizer("PE-Core-B-16-dinov3"), val_tr


def apply_neg_mode(sim, neg_mode):
    """与 eval_standard.py:79 同一套约定。projective 训练目标是 |cos|→1。"""
    if neg_mode == "standard":
        return sim
    if neg_mode in ("projective", "antipodal", "orthogonal"):
        return sim.abs()
    raise ValueError(neg_mode)


@torch.no_grad()
def build_classifier(model, tok, classnames, device):
    """[embed_dim, C] 的类名嵌入：80 个官方模板取平均后再归一化。"""
    from open_clip.zero_shot_metadata import OPENAI_IMAGENET_TEMPLATES
    cols = []
    for name in classnames:
        texts = tok([t(name) for t in OPENAI_IMAGENET_TEMPLATES]).to(device)
        e = model.encode_text(texts, normalize=True).float()
        e = F.normalize(e.mean(0), dim=-1)
        cols.append(e)
    return torch.stack(cols, dim=1)                        # [D, C]


@torch.no_grad()
def dense_embed(model, imgs, dense_mode):
    """[B,3,224,224] → [B, h, w, embed_dim]（未归一化）。

    penult：倒数第二层 patch map 逐位置过 visual.head —— 与训练时区域分支同路径。
    last  ：最后一层 patch token 逐位置过 visual.head —— 对照项。
    """
    visual = model.clip_model.visual
    trunk, head = visual.trunk, visual.head
    if dense_mode == "penult":
        inter = trunk.forward_intermediates(
            imgs, indices=[-2], return_prefix_tokens=False, norm=True,
            stop_early=True, intermediates_only=True)
        fmap = inter[0]                                     # [B, C, h, w]
        B, C, h, w = fmap.shape
        toks = fmap.permute(0, 2, 3, 1).reshape(B, h * w, C)
    else:
        feats = trunk.forward_features(imgs)                # [B, P+N, C]
        npt = getattr(trunk, "num_prefix_tokens", 1)
        toks = feats[:, npt:, :]
        B, N, C = toks.shape
        h = w = int(round(N ** 0.5))
        assert h * w == N, f"patch 数 {N} 不是完全平方"
    out = head(toks)                                        # [B, h*w, embed_dim]
    return out.reshape(B, h, w, out.shape[-1])


@torch.no_grad()
def seg_logits_one(model, img_t, classifier, neg_mode, dense_mode,
                   win=224, stride=112, batch=16):
    """滑窗推理，返回 [C, H, W] 的 logits（H,W = img_t 的尺寸）。"""
    _, H, W = img_t.shape
    Ct = classifier.shape[1]
    dev, dt = img_t.device, img_t.dtype
    # 短边不足一个窗口时补齐（VOC 里极少，但 --short-side 调小后会出现）
    pad_h, pad_w = max(0, win - H), max(0, win - W)
    if pad_h or pad_w:
        img_t = F.pad(img_t, (0, pad_w, 0, pad_h))
    _, Hp, Wp = img_t.shape
    ys = list(range(0, max(Hp - win, 0) + 1, stride))
    xs = list(range(0, max(Wp - win, 0) + 1, stride))
    if ys[-1] + win < Hp:
        ys.append(Hp - win)                                 # 保证覆盖右/下边缘
    if xs[-1] + win < Wp:
        xs.append(Wp - win)

    acc = torch.zeros(Ct, Hp, Wp, device=dev, dtype=torch.float32)
    cnt = torch.zeros(1, Hp, Wp, device=dev, dtype=torch.float32)
    coords = [(y, x) for y in ys for x in xs]
    for s in range(0, len(coords), batch):
        chunk = coords[s:s + batch]
        crops = torch.stack([img_t[:, y:y + win, x:x + win] for y, x in chunk])
        emb = dense_embed(model, crops.to(dtype=dt), dense_mode)   # [b,h,w,D]
        emb = F.normalize(emb.float(), dim=-1)
        sim = apply_neg_mode(emb @ classifier, neg_mode)     # [b,h,w,C]
        sim = sim.permute(0, 3, 1, 2)                        # [b,C,h,w]
        sim = F.interpolate(sim, size=(win, win), mode="bilinear",
                            align_corners=False)
        for i, (y, x) in enumerate(chunk):
            acc[:, y:y + win, x:x + win] += sim[i]
            cnt[:, y:y + win, x:x + win] += 1
    return (acc / cnt.clamp(min=1))[:, :H, :W]


def get_norm(val_tr):
    """从 val transform 里取出 Normalize 的 mean/std —— 不能猜，PE-Core 不是 OpenAI 常数。"""
    for t in getattr(val_tr, "transforms", []):
        if t.__class__.__name__ == "Normalize":
            return torch.tensor(t.mean).view(3, 1, 1), torch.tensor(t.std).view(3, 1, 1)
    raise RuntimeError("val transform 里找不到 Normalize")


def miou_from_conf(conf, ignore_first=False):
    """conf[gt, pred] → (mIoU, aAcc, per_class_iou)。ignore_first 用于 VOC-20。"""
    conf = conf.astype(np.float64)
    if ignore_first:
        conf = conf[1:, 1:]
    inter = np.diag(conf)
    union = conf.sum(1) + conf.sum(0) - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1e-9), np.nan)
    aacc = inter.sum() / max(conf.sum(), 1e-9)
    return float(np.nanmean(iou)), float(aacc), iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--neg-mode", default="projective",
                    choices=["standard", "projective", "antipodal", "orthogonal"],
                    help="必须与训练配方的 --neg-mode 一致（E 配方=projective）")
    ap.add_argument("--dense-mode", default="penult", choices=["penult", "last"],
                    help="penult=倒数第二层（与训练区域分支同路径，默认）")
    ap.add_argument("--short-side", type=int, default=336,
                    help="短边缩放到多少；窗口固定 224 与训练分辨率一致")
    ap.add_argument("--stride", type=int, default=112)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 张（冒烟用）")
    ap.add_argument("--bg-thd", type=float, nargs="*",
                    default=[0.3, 0.4, 0.5, 0.6],
                    help="VOC-21 的背景概率阈值，会各报一遍")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ids_file = VOC_ROOT / "ImageSets/Segmentation/val.txt"
    if not ids_file.exists():
        sys.exit(f"找不到 VOC2012：{ids_file}（先跑 tools/install_all.sh 的 data_voc2012）")
    ids = ids_file.read_text().split()
    if args.limit:
        ids = ids[:args.limit]
    scope = "★全量 1449★" if len(ids) >= 1449 else f"⚠️子集 {len(ids)} 张(不可与全量混比)"
    print(f"[{args.tag}] VOC-2012 val OVSS | device={device} "
          f"neg_mode={args.neg_mode} dense={args.dense_mode} "
          f"short={args.short_side} win=224/stride={args.stride} {scope}", flush=True)

    model, tok, val_tr = load_model(args.ckpt, device)
    mean, std = get_norm(val_tr)
    dt = torch.float16 if device == "cuda" else torch.float32
    mean, std = mean.to(device), std.to(device)
    classifier = build_classifier(model, tok, VOC_CLASSES, device)   # [D,20]
    print(f"  类名嵌入 {tuple(classifier.shape)}（20 类 × 80 模板平均）", flush=True)
    logit_scale = float(model.logit_scale.exp().item())

    C = len(VOC_CLASSES)
    conf20 = np.zeros((C, C), dtype=np.int64)               # 只前景，背景当 ignore
    conf21 = {t: np.zeros((C + 1, C + 1), dtype=np.int64) for t in args.bg_thd}
    t0 = time.time()
    for n, iid in enumerate(ids, 1):
        img = Image.open(VOC_ROOT / f"JPEGImages/{iid}.jpg").convert("RGB")
        gt = np.array(Image.open(VOC_ROOT / f"SegmentationClass/{iid}.png"))
        W0, H0 = img.size
        s = args.short_side / min(W0, H0)
        img = img.resize((max(1, round(W0 * s)), max(1, round(H0 * s))), Image.BICUBIC)
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().to(device) / 255.
        x = ((x - mean) / std).to(dt)

        lg = seg_logits_one(model, x, classifier, args.neg_mode, args.dense_mode,
                            win=224, stride=args.stride)          # [C,h,w]
        # 回到原图尺寸再取 argmax（先 argmax 再 resize 会引入插值伪影）
        lg = F.interpolate(lg[None], size=(H0, W0), mode="bilinear",
                           align_corners=False)[0]
        prob = (lg * logit_scale).softmax(0)
        pred_fg = prob.argmax(0).cpu().numpy()
        maxp = prob.max(0).values.cpu().numpy()

        # --- VOC-20：GT 背景(0) 与 void(255) 都忽略，只在前景像素上算
        m20 = (gt > 0) & (gt < 255)
        if m20.any():
            g = gt[m20].astype(np.int64) - 1
            p = pred_fg[m20].astype(np.int64)
            np.add.at(conf20, (g, p), 1)
        # --- VOC-21：背景是一个真实类别，靠最大前景概率过阈值判定
        m21 = gt < 255
        if m21.any():
            g21 = gt[m21].astype(np.int64)                  # 0=bg, 1..20
            for thd in args.bg_thd:
                p21 = np.where(maxp[m21] < thd, 0, pred_fg[m21] + 1)
                np.add.at(conf21[thd], (g21, p21.astype(np.int64)), 1)

        if n % 200 == 0 or n == len(ids):
            print(f"    ... {n}/{len(ids)}  ({time.time() - t0:.0f}s)", flush=True)

    miou20, aacc20, iou20 = miou_from_conf(conf20)
    print(f"  ★VOC-20 mIoU={miou20 * 100:.2f}  aAcc={aacc20 * 100:.2f}  "
          f"({len(ids)} 图, 20 类, 无阈值超参, {scope})", flush=True)
    for thd in args.bg_thd:
        m, a, _ = miou_from_conf(conf21[thd])
        print(f"   VOC-21 mIoU={m * 100:.2f}  aAcc={a * 100:.2f}  (bg_thd={thd})",
              flush=True)
    order = np.argsort(np.nan_to_num(iou20, nan=-1))
    worst = ", ".join(f"{VOC_CLASSES[i]} {iou20[i] * 100:.1f}" for i in order[:5])
    best = ", ".join(f"{VOC_CLASSES[i]} {iou20[i] * 100:.1f}" for i in order[::-1][:5])
    print(f"  最好 5 类: {best}", flush=True)
    print(f"  最差 5 类: {worst}", flush=True)
    print(f"  耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
