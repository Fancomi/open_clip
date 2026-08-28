#!/usr/bin/env python3
"""逐层 patch 间余弦：直接检验"区域监督提升空间可区分性"这个机制假说。

背景与为什么必须做
------------------
§5.1 的机制假说是：VISReg 只约束 CLS 的跨通道各向同性，实测会**升高** patch 间余弦
（`X_sum.patch_cos` 0.137→0.161，`diag_residual_dominance.py`），而区域-短语监督把
损失压在 `roi_align` 的局部 patch 特征上，逼不同位置分开 → patch 余弦应随
`region_weight` **单调下降**。

2026-08-27 的 OVSS 结果（`region_01_supervision.md` §5.8）让这个检验变成**可能推翻
假说**的实验：OVSS 直接测空间可区分性，而它在 `region_weight` 变大时是**下降**的。
于是有两种可能：
  (a) patch_cos 随 W 上升 → 假说的"方向"错了，但"patch_cos ↔ 可分割性"的联系还在；
  (b) patch_cos 随 W 下降，而 OVSS 也下降 → **"patch 余弦低 = 可分割性好"这一步本身是错的**，
      假说需要重写。
无论哪种，都比现在只有"指标涨了"要多知道一件事。

与 `diag_residual_dominance.py` 的分工
-------------------------------------
那个脚本只看**最后一个 block**、只比**两个** ckpt，且是在 trunk 的 768d 空间里看。
本脚本补三件它做不到的事：
  1. **逐层**（全部 blocks）—— 区域损失只压在**倒数第二层**，最后一层的读数不代表它。
  2. **任意多个 ckpt 一次跑完**（权重扫描有 5~6 个点，两两配对跑法要重复加载）。
  3. **`visual.head` 投影后的 1024d 空间也报一遍** —— OVSS 的判定就发生在这个空间里
     （patch 特征在 1024d 与类名嵌入比），trunk 空间里的可区分性未必传递过去。
  4. 同时报 `mean cos` 与 **`mean |cos|`** —— 训练用 `--neg-mode projective`，
     排序口径是 `|cos|`，所以 `|cos|` 才是与 OVSS 判定一致的那个量
     （两个 patch 余弦 −0.9 在 projective 口径下是"极其相似"，不是"可区分"）。

用法
----
  python scripts/tools/diag_patch_cos.py --n-images 128 \
      gt_base=logs/visreg_gemma_gt_gt_base_0811_1318/checkpoints/epoch_10.pt \
      W0.2=logs/visreg_gemma_regw0.2k12_projective_E_0820_2337/checkpoints/epoch_10.pt
纯推理，单卡约 3GB，可与训练共存。
"""
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def load_model(ckpt_path, device):
    """与 eval_ovss / eval_knn_probe 完全相同的加载路径，保证与那些数字可对照。"""
    from open_clip import create_model_and_transforms
    from open_clip.factory import context_length_from_checkpoint
    from open_clip.model import CLIPLeJEPA

    ctx = context_length_from_checkpoint(ckpt_path)
    base, _, val_tr = create_model_and_transforms(
        "PE-Core-B-16-dinov3", "", precision="fp32", device="cpu",
        output_dict=True, force_context_length=ctx)
    model = CLIPLeJEPA(clip_model=base, sigreg_target="cls", output_dict=True)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"    加载: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    return model.to(device).eval().half(), val_tr


def pair_cos(x):
    """x: [B, N, D] → (mean cos, mean |cos|, mean |cos| 去共同分量)，
    只算每图内部 patch 两两、去对角。

    第三个量是为什么加的
    -------------------
    `visual.head` 在本配方里就是**一层 `Linear(768→1024)`**
    （`PE-Core-B-16-dinov3.json`: `timm_pool=token` + `timm_proj=linear`，
    trunk 自带的 `attn_pool` 在 `timm_model.py:87` 被删掉了）。
    一层线性映射不会"丢掉结构"，但它**可以把所有 patch 共有的那个方向压掉**。
    §5.9 观察到 trunk L10 的 `|cos|` 0.44~0.47 过一层 head 掉到 0.17~0.19，
    如果这只是共同分量被压掉，那"head 丢结构"这条线索就是假的；
    同时 §5.9 说"区域监督把 trunk patch 余弦推高 2.7 倍"也需要确认
    **不是全体 patch 一起漂向同一个向量**造成的假象。
    去共同分量 = 每图先减掉自己的 patch 均值向量再算 —— 这正好是"head 是不是在做中心化"
    和"余弦升高是不是共同分量"这两个问题的同一个判据。
    """
    xn = F.normalize(x.float(), dim=-1)
    sim = torch.einsum("bnd,bmd->bnm", xn, xn)
    N = sim.shape[-1]
    off = ~torch.eye(N, dtype=torch.bool, device=sim.device)
    v = sim[:, off]
    xc = F.normalize(x.float() - x.float().mean(dim=1, keepdim=True), dim=-1)
    vc = torch.einsum("bnd,bmd->bnm", xc, xc)[:, off]
    return float(v.mean()), float(v.abs().mean()), float(vc.abs().mean())


def text_align(x, classifier):
    """patch↔文本对齐度。x: [B,N,D]，classifier: [D,C]（已归一化的类名嵌入）。

    返回 (top1, margin, nuniq, ent, nbr)：
      top1   = 每个 patch 到最近类名的 |cos|，对所有 patch 取平均。
      margin = top1 − top2，越大 = 这个"最近类"越明确。
      nuniq  = 每张图里 argmax 类别的种数（1~C），对图取平均。
      ent    = 每张图 argmax 直方图的熵（nats），对图取平均。
      nbr    = **4-邻域一致率**：相邻两个 patch 的 argmax 相同的比例。
    ⚠️ top1 / margin 是**幅度**类量，不看对不对。nuniq / ent 实测在 HEAD@-2 上
    饱和（gt_base 19.56 / W2.0 19.50，满值 20）—— 说明 gt_base 不是"整图一个类"，
    而是**近均匀随机撒标签**，与它 aAcc 10.2%≈随机一致。
    区分"随机撒"和"成片"的量是 `nbr`：随机撒 ≈ 1/C，成片 → 接近 1。
    mIoU 要的是成片，所以 nbr 才是与 OVSS 同向的候选变量。
    """
    xn = F.normalize(x.float(), dim=-1)
    sim = (xn @ classifier).abs()                          # [B,N,C]，projective 口径
    top2 = sim.topk(2, dim=-1).values
    arg = sim.argmax(-1)                                   # [B,N]
    C = classifier.shape[1]
    hist = F.one_hot(arg, C).float().mean(1)               # [B,C] 每图的类别占比
    ent = -(hist * (hist + 1e-12).log()).sum(-1)
    nuniq = (hist > 0).sum(-1).float()
    g = int(round(arg.shape[1] ** 0.5))
    assert g * g == arg.shape[1], f"patch 数 {arg.shape[1]} 不是完全平方，无法还原网格"
    a = arg.view(-1, g, g)
    nbr = torch.cat([(a[:, :, :-1] == a[:, :, 1:]).flatten(1),
                     (a[:, :-1] == a[:, 1:]).flatten(1)], dim=1).float().mean()
    return (float(top2[..., 0].mean()),
            float((top2[..., 0] - top2[..., 1]).mean()),
            float(nuniq.mean()), float(ent.mean()), float(nbr))


@torch.no_grad()
def analyze(model, images, device, classifier=None, batch=8):
    """返回 {层名: (cos, abscos)}。层名 L0..L{d-1} 为 trunk 归一化后 768d，
    HEAD@-2 / HEAD@-1 为过 visual.head 后的 1024d。
    classifier 给了的话，额外报 HEAD@-2/-1 的 patch↔文本对齐（TXT* 行）。"""
    visual = model.clip_model.visual
    trunk, head = visual.trunk, visual.head
    depth = len(trunk.blocks)
    idx = list(range(depth))
    acc, cnt = {}, 0
    for s in range(0, len(images), batch):
        chunk = images[s:s + batch].to(device=device, dtype=torch.float16)
        inter = trunk.forward_intermediates(
            chunk, indices=idx, return_prefix_tokens=False, norm=True,
            output_fmt="NLC", intermediates_only=True)
        b = chunk.shape[0]
        for i, toks in enumerate(inter):                  # [B, N, C]
            for key, val in zip((f"L{i}", f"L{i}|abs|", f"L{i}|ctr|"),
                                pair_cos(toks)):
                acc[key] = acc.get(key, 0.0) + val * b
        for tag, i in (("HEAD@-2", depth - 2), ("HEAD@-1", depth - 1)):
            proj = head(inter[i])                         # [B, N, 1024]
            for key, val in zip((tag, tag + "|abs|", tag + "|ctr|"),
                                pair_cos(proj)):
                acc[key] = acc.get(key, 0.0) + val * b
            if classifier is not None:
                names = ("TXTtop1", "TXTmarg", "TXTnuniq", "TXTent", "TXTnbr")
                for nm, val in zip(names, text_align(proj, classifier)):
                    k = f"{nm} {tag}"
                    acc[k] = acc.get(k, 0.0) + val * b
                # 去共同分量后再判一次。动机：`|cos|` 那两张表显示读出空间里
                # **原始值单调降、去共同分量后单调升** —— 也就是"全体 patch 共有的那个
                # 方向"占了读数的大头。类名判定用的是同一个 patch 向量，
                # 所以这个共同分量同样会污染 argmax。这里同表报一遍，
                # 若 `ctr` 版 `nbr` 明显高于原始版，说明"推理时先减掉每图 patch 均值"
                # 是一个零成本的读出改进 → 值得在 eval_ovss.py 上试真 mIoU。
                pc = proj.float() - proj.float().mean(dim=1, keepdim=True)
                for nm, val in zip(names, text_align(pc, classifier)):
                    k = f"{nm}ctr {tag}"
                    acc[k] = acc.get(k, 0.0) + val * b
        cnt += b
    return {k: v / cnt for k, v in acc.items()}


@torch.no_grad()
def build_voc_classifier(model, device):
    """[D, 20] 的 VOC 类名嵌入，与 eval_ovss.py:build_classifier 完全同一套做法
    （20 个类名 × 80 个 OpenAI 官方模板取平均后再归一化），保证两边数字可对照。"""
    from open_clip import get_tokenizer
    from open_clip.zero_shot_metadata import OPENAI_IMAGENET_TEMPLATES
    VOC = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
           "chair", "cow", "dining table", "dog", "horse", "motorbike", "person",
           "potted plant", "sheep", "sofa", "train", "tv monitor"]
    tok = get_tokenizer("PE-Core-B-16-dinov3")
    cols = []
    for name in VOC:
        texts = tok([t(name) for t in OPENAI_IMAGENET_TEMPLATES]).to(device)
        e = model.encode_text(texts, normalize=True).float()
        cols.append(F.normalize(e.mean(0), dim=-1))
    return torch.stack(cols, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="tag=ckpt路径，可给多个；按给定顺序出表")
    ap.add_argument("--n-images", type=int, default=128)
    ap.add_argument("--tsv", default="/root/paddlejob/workspace/env_run/penghaotian/"
                                    "datas/coco/annotations/karpathy_1cap.tsv")
    ap.add_argument("--voc-text", action="store_true",
                    help="额外报 patch↔VOC 类名的对齐度（TXT* 行）——"
                         "这是 patch↔patch 余弦看不到的那一半")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = []
    for sp in args.specs:
        assert "=" in sp, f"要写成 tag=路径：{sp}"
        t, c = sp.split("=", 1)
        assert Path(c).exists(), f"ckpt 不存在：{c}"
        pairs.append((t, c))

    # 图像批：COCO probe tsv（训练集外），与 diag_residual_dominance 同源
    import pandas as pd
    from PIL import Image
    from open_clip import create_model_and_transforms
    _, _, prep = create_model_and_transforms("PE-Core-B-16-dinov3", "",
                                             precision="fp32", device="cpu",
                                             force_context_length=256)
    df = pd.read_csv(args.tsv, sep="\t").head(args.n_images)
    col = "filepath" if "filepath" in df.columns else df.columns[0]
    imgs = []
    for p in df[col].tolist():
        try:
            imgs.append(prep(Image.open(str(p)).convert("RGB")))
        except Exception:
            continue
    images = torch.stack(imgs)
    print(f"=> {len(images)} 图 来自 {args.tsv}（{len(pairs)} 个 ckpt）", flush=True)

    res = {}
    for tag, ck in pairs:
        print(f"=> {tag}", flush=True)
        m, _ = load_model(ck, device)
        cls = build_voc_classifier(m, device) if args.voc_text else None
        res[tag] = analyze(m, images, device, classifier=cls)
        del m, cls
        torch.cuda.empty_cache()

    tags = [t for t, _ in pairs]
    keys = [k for k in res[tags[0]]
            if not k.endswith(("|abs|", "|ctr|")) and not k.startswith("TXT")]
    W = max(9, max(len(t) for t in tags) + 1)
    def table(title, rows, width):
        print("\n" + "=" * (width + W * len(tags)))
        print(title)
        print(f"{'层':<{width}}" + "".join(f"{t:>{W}}" for t in tags))
        print("-" * (width + W * len(tags)))
        for label, key in rows:
            print(f"{label:<{width}}" +
                  "".join(f"{res[t][key]:>{W}.4f}" for t in tags))

    table("patch↔patch mean cos（越小=方向越分散）",
          [(k, k) for k in keys], 12)
    table("patch↔patch mean |cos|（★与 projective 口径一致的那个量★）",
          [(k, k + "|abs|") for k in keys], 12)
    table("patch↔patch mean |cos| 去共同分量（每图先减 patch 均值向量）"
          "—— 与上一张表对比：差得多 = 上表主要是全体 patch 共漂的假象；"
          "\n     再与 HEAD@-* 行对比：若 trunk 的去共同分量值 ≈ HEAD 的原始值，"
          "说明 head 只是在做中心化，'head 丢结构'这条线索作废",
          [(k, k + "|ctr|") for k in keys], 12)
    if args.voc_text:
        table("patch↔VOC 类名（top1/marg = 幅度，是**反**指标；★nbr = 4-邻域 argmax 一致率，"
              "才是与 OVSS mIoU 同向的那个量★；\n     `*ctr` 行 = 每图先减 patch 均值向量"
              "再判，若 ctr 版 nbr 明显更高则'推理前先中心化'是零成本读出改进）",
              [(k, k) for k in res[tags[0]] if k.startswith("TXT")], 20)

    print("\n判读：区域损失只压在**倒数第二层**（= L{}，即 HEAD@-2 的输入）；"
          "OVSS 的判定发生在 HEAD@-* 这个 1024d 空间里。"
          .format(len(keys) - 4))
    print("      机制假说预测 patch 余弦随 region_weight 单调下降；"
          "若它下降而 OVSS 也下降，则假说里"
          "「patch 余弦低 = 可分割性好」这一步是错的。")


if __name__ == "__main__":
    main()
