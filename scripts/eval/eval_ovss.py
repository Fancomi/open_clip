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
   `--dataset ade` 是第二个 benchmark（**ADE20K-150 val 2000 图**）：它没有背景类，
   GT 的 0 就是 ignore，所以同样没有阈值超参，走与 VOC-20 完全相同的代码路径。
   选它作为第二个是因为标注是 PNG label map、纯 PIL 可读、**零新依赖**
   （训练环境锁死 `torch==2.6.0+cu124`，不值得为 `pycocotools` / `detail-api` 去动）。
   ⚠️ **两个 benchmark 都只用朴素类名、不做同义词扩展**（VOC 20 个、ADE 取
   `objectInfo150.txt` 的第一个同义词）。SCLIP/ClearCLIP 用的 `cls_*.txt` 含扩展、
   通常更高 → **我们内部可比，但与论文数字不同条件**。
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
  python scripts/eval/eval_ovss.py --ckpt ... --tag foo --dataset ade  # ADE20K-150
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
ADE_ROOT = Path("/root/paddlejob/workspace/env_run/penghaotian/datas/ade20k/"
                "ADEChallengeData2016")

# VOC2012 调色板标签：0=背景，1..20=前景，255=void（物体轮廓，官方要求忽略）
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "dining table", "dog", "horse", "motorbike", "person",
    "potted plant", "sheep", "sofa", "train", "tv monitor",
]


def ade_classnames():
    """从官方 objectInfo150.txt 读 150 个类名，取**第一个同义词**。

    为什么不手抄：150 个名字手抄必出错，而这个文件就是标注的权威来源。
    为什么只取第一个同义词：与 VOC-20 用 20 个朴素类名保持一致 ——
    SCLIP/ClearCLIP 的 `cls_ade20k.txt` 含同义词扩展，通常更高，
    我们两个 benchmark 都不做扩展，内部可比，但**不与论文数字同条件**。
    """
    f = ADE_ROOT / "objectInfo150.txt"
    if not f.exists():
        sys.exit(f"找不到 {f}（先解压 ADEChallengeData2016.zip）")
    names = []
    for line in f.read_text().splitlines()[1:]:          # 首行是表头
        cols = line.split("\t")
        if len(cols) < 5:
            continue
        names.append(cols[4].split(",")[0].strip())
    assert len(names) == 150, f"objectInfo150 解析出 {len(names)} 类，应为 150"
    return names


def load_dataset(name, limit):
    """返回 (items, classnames, has_bg, scope, full_n)。

    GT 约定统一成 **0 = 背景/ignore，1..C = 类，255 = void**：
    VOC 原生就是这样；ADE 的 0 是 ignore、1..150 是类，没有 255。
    所以下游的 `mask = (gt>0)&(gt<255)`、`g = gt-1` 两个数据集共用一条代码路径。
    has_bg=False 时跳过"背景阈值"那一族指标（ADE 没有背景类，不存在这个超参）。
    """
    if name == "voc":
        ids_file = VOC_ROOT / "ImageSets/Segmentation/val.txt"
        if not ids_file.exists():
            sys.exit(f"找不到 VOC2012：{ids_file}"
                     f"（先跑 tools/install_all.sh 的 data_voc2012）")
        ids = ids_file.read_text().split()
        items = [(VOC_ROOT / f"JPEGImages/{i}.jpg",
                  VOC_ROOT / f"SegmentationClass/{i}.png") for i in ids]
        classes, has_bg, full_n = VOC_CLASSES, True, 1449
        title = "VOC-2012 val"
    elif name == "ade":
        d = ADE_ROOT / "images/validation"
        if not d.exists():
            sys.exit(f"找不到 ADE20K：{d}（先解压 ADEChallengeData2016.zip）")
        stems = sorted(p.stem for p in d.glob("ADE_val_*.jpg"))
        items = [(d / f"{s}.jpg", ADE_ROOT / f"annotations/validation/{s}.png")
                 for s in stems]
        classes, has_bg, full_n = ade_classnames(), False, 2000
        title = "ADE20K-150 val"
    else:
        raise ValueError(name)
    if limit:
        items = items[:limit]
    scope = (f"★全量 {full_n}★" if len(items) >= full_n
             else f"⚠️子集 {len(items)} 张(不可与全量混比)")
    return items, classes, has_bg, scope, title


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
    ap.add_argument("--dataset", default="voc", choices=["voc", "ade"],
                    help="voc=VOC-2012 val 1449 图 20 类；ade=ADE20K-150 val 2000 图 150 类")
    ap.add_argument("--short-side", type=int, default=336,
                    help="短边缩放到多少；窗口固定 224 与训练分辨率一致")
    ap.add_argument("--stride", type=int, default=112)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 张（冒烟用）")
    ap.add_argument("--bg-thd", type=float, nargs="*",
                    default=[0.3, 0.4, 0.5, 0.6],
                    help="VOC-21 的背景概率阈值，会各报一遍（ADE 无背景类，自动跳过）")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    items, CLASSES, has_bg, scope, title = load_dataset(args.dataset, args.limit)
    print(f"[{args.tag}] {title} OVSS | device={device} "
          f"neg_mode={args.neg_mode} dense={args.dense_mode} "
          f"short={args.short_side} win=224/stride={args.stride} {scope}", flush=True)

    model, tok, val_tr = load_model(args.ckpt, device)
    mean, std = get_norm(val_tr)
    dt = torch.float16 if device == "cuda" else torch.float32
    mean, std = mean.to(device), std.to(device)
    C = len(CLASSES)
    classifier = build_classifier(model, tok, CLASSES, device)        # [D,C]
    print(f"  类名嵌入 {tuple(classifier.shape)}（{C} 类 × 80 模板平均，"
          f"无同义词扩展）", flush=True)
    logit_scale = float(model.logit_scale.exp().item())

    conf_fg = np.zeros((C, C), dtype=np.int64)          # 只前景，背景/ignore 当 ignore
    conf_bg = ({t: np.zeros((C + 1, C + 1), dtype=np.int64) for t in args.bg_thd}
               if has_bg else {})
    t0 = time.time()
    for n, (ip, gp) in enumerate(items, 1):
        img = Image.open(ip).convert("RGB")
        gt = np.array(Image.open(gp))
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

        # --- 主口径：GT 的 0（背景/ignore）与 255（void）都忽略，只在类像素上算
        m = (gt > 0) & (gt < 255)
        if m.any():
            g = gt[m].astype(np.int64) - 1
            p = pred_fg[m].astype(np.int64)
            np.add.at(conf_fg, (g, p), 1)
        # --- 附带口径（只有 VOC 有）：背景是一个真实类别，靠最大前景概率过阈值判定
        if has_bg:
            m21 = gt < 255
            if m21.any():
                g21 = gt[m21].astype(np.int64)              # 0=bg, 1..C
                for thd in args.bg_thd:
                    p21 = np.where(maxp[m21] < thd, 0, pred_fg[m21] + 1)
                    np.add.at(conf_bg[thd], (g21, p21.astype(np.int64)), 1)

        if n % 200 == 0 or n == len(items):
            print(f"    ... {n}/{len(items)}  ({time.time() - t0:.0f}s)", flush=True)

    miou, aacc, iou = miou_from_conf(conf_fg)
    label = f"VOC-{C}" if args.dataset == "voc" else f"ADE-{C}"
    print(f"  ★{label} mIoU={miou * 100:.2f}  aAcc={aacc * 100:.2f}  "
          f"({len(items)} 图, {C} 类, 无阈值超参, {scope})", flush=True)
    for thd in args.bg_thd if has_bg else []:
        m_, a_, _ = miou_from_conf(conf_bg[thd])
        print(f"   VOC-{C + 1} mIoU={m_ * 100:.2f}  aAcc={a_ * 100:.2f}  (bg_thd={thd})",
              flush=True)
    order = np.argsort(np.nan_to_num(iou, nan=-1))
    worst = ", ".join(f"{CLASSES[i]} {iou[i] * 100:.1f}" for i in order[:5])
    best = ", ".join(f"{CLASSES[i]} {iou[i] * 100:.1f}" for i in order[::-1][:5])
    print(f"  最好 5 类: {best}", flush=True)
    print(f"  最差 5 类: {worst}", flush=True)
    # 全部类的逐类 IoU（按类下标固定顺序，方便跨 run 逐位对齐比较）。
    # 只报"最差 5 类"时无法做"某一组类是否被专门救回"这种分组比较（C5 的方向性
    # 预登记判据需要它，见 analysis/research/region_01_supervision.md §6.3）。
    # 行内不含 "mIoU"/"aAcc" 字样，不会污染各驱动脚本按这两个词做的 grep。
    allcls = ", ".join(f"{CLASSES[i]} {iou[i] * 100:.1f}" for i in range(len(CLASSES)))
    print(f"  逐类 IoU（{len(CLASSES)} 类，按类下标）: {allcls}", flush=True)
    print(f"  耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
