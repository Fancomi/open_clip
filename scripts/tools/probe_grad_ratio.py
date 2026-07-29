#!/usr/bin/env python3
"""诊断：正则项对 backbone 的梯度，相对对比损失占多大比例。

回答「上升空间有多大」——若比值极小（如 1e-3），说明正则杠杆几乎没被拉动，
加大权重/换作用位置有很大空间；若接近 1，说明已在正常量级，空间有限。

同时对比 sigreg_target = cls (Identity) vs cls_proj (MLP projector) 两种位置，
以及不同权重下的比值，为后续权重扫描定量级。

用法:
  python scripts/tools/probe_grad_ratio.py --ckpt logs/<run>/checkpoints/epoch_7.pt
"""
import argparse
import math
import warnings

warnings.filterwarnings("ignore")

import torch
import pandas as pd
from PIL import Image

import open_clip
from open_clip.loss import SIGReg, VISReg, SigLipLoss
from open_clip.model import CLIPLeJEPA
from open_clip import get_tokenizer


def backbone_grad_norm(model, loss, retain=False):
    """反传 loss，返回其对 visual backbone 参数的梯度 L2 范数。"""
    model.zero_grad(set_to_none=True)
    loss.backward(retain_graph=retain)
    g2 = 0.0
    for p in model.clip_model.visual.parameters():
        if p.grad is not None:
            g2 += float(p.grad.detach().pow(2).sum())
    return math.sqrt(g2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tsv", default="/root/paddlejob/workspace/env_run/penghaotian/datas/cc3m-tsv/annotations/clip_train.tsv")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--steps", type=int, default=4, help="平均多少个 batch")
    ap.add_argument("--weight", type=float, default=1.83e-4, help="VISReg 标定权重")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)

    print(f"=> ckpt: {args.ckpt}")
    for target in ("cls", "cls_proj"):
        base = open_clip.create_model("PE-Core-B-16-dinov3", pretrained=None, output_dict=True)
        model = CLIPLeJEPA(clip_model=base, sigreg_target=target, proj_dim=512,
                           proj_layers=3, output_dict=True)
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
        sd = {k[len("module."):]: v for k, v in sd.items() if k.startswith("module.")}
        model.load_state_dict(sd, strict=False)
        model = model.to(dev).train()

        _, _, pp = open_clip.create_model_and_transforms("PE-Core-B-16-dinov3", pretrained=None)
        tok = get_tokenizer("PE-Core-B-16-dinov3")
        df = pd.read_csv(args.tsv, sep="\t", nrows=args.batch * args.steps * 3)

        vis = VISReg(num_slices=256, lambda_center=0.0, gather=False).to(dev)
        siglip = SigLipLoss(rank=0, world_size=1, neg_mode="projective")

        rows, i = [], 0
        for _ in range(args.steps):
            imgs, txts = [], []
            while len(imgs) < args.batch and i < len(df):
                try:
                    imgs.append(pp(Image.open(df["filepath"][i]).convert("RGB")))
                    txts.append(str(df["caption"][i]))
                except Exception:
                    pass
                i += 1
            x = torch.stack(imgs).to(dev)
            t = tok(txts).to(dev)

            out = model(image=x, text=t)
            ls = model.logit_scale.exp()
            lb = model.logit_bias

            g_con = backbone_grad_norm(
                model, siglip(out["image_features"], out["text_features"], ls, lb), retain=True)
            reg_raw = vis(out["image_proj"])
            g_reg1 = backbone_grad_norm(model, reg_raw, retain=True)      # weight=1
            rows.append((g_con, g_reg1, float(reg_raw.detach())))

        gc = sum(r[0] for r in rows) / len(rows)
        gr1 = sum(r[1] for r in rows) / len(rows)
        rv = sum(r[2] for r in rows) / len(rows)
        gr_w = gr1 * args.weight

        print(f"\n===== sigreg_target = {target} =====")
        print(f"  对比损失 → backbone 梯度范数      : {gc:.4e}")
        print(f"  VISReg(w=1) → backbone 梯度范数   : {gr1:.4e}   (裸 loss={rv:.5f})")
        print(f"  VISReg(w={args.weight:.2e}) 梯度   : {gr_w:.4e}")
        print(f"  ★ 梯度占比 = reg/contrastive      : {gr_w/gc:.3e}")
        print(f"    → 若要占比达到 0.1，权重需 ≈ {0.1*gc/gr1:.3e}  ({0.1*gc/gr1/args.weight:.0f}× 当前)")
        print(f"    → 若要占比达到 1.0，权重需 ≈ {1.0*gc/gr1:.3e}  ({1.0*gc/gr1/args.weight:.0f}× 当前)")

        del model, base
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
