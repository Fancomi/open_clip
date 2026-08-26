#!/usr/bin/env python3
"""区域损失对 backbone 的**梯度**占比 —— 不是损失值占比。

为什么必须量
------------
本项目此前只有**损失值**占比：`region_weight=2.0` 时 `Region_loss` 占总损失 68%，
W=4.0 时 83%。但损失值大不等于梯度大 —— 区域损失是 SigLIP 逐对口径、有效行数约
`B*K` ≈ 4340（W=2.0，K=12，掉框后），主对比损失是 4096 行；两者的 per-sample 尺度、
经过的 backbone 通路（`roi_align` 取倒数第二层 vs CLS 走完整 head）都不一样。

反面教训来自 VISReg 线：那条线实测 VISReg @1.83e-4 对 backbone 的梯度范数
**2.94e-05**，对比损失 **1.43e+02** → 占比 **2.06e-07**。也就是说，
**它此前跨 4 个数量级的所有权重/方向/目标形状扫描全部发生在死区里**，
那些"无差异"结论一个都不构成有效排除（`probe_grad_ratio.py`）。
区域这条线的 `region_weight` 已经扫到 4.0 还在涨，必须先确认它不是同一类错觉，
以及"还有多少空间"。

与 `probe_grad_ratio.py` 的分工
-------------------------------
那个脚本只管 SIGReg/VISReg 正则项，用的是纯图文 batch，构不出区域分支。
本脚本按 `params.txt` 复原**整条训练前向**（含 `roi_align` + 区域短语塔），
然后把 `create_loss` 返回的 loss dict **逐项单独反传**，各自量对
`clip_model.visual` 的梯度 L2 范数。已加权后的量才是真正进 backbone 的东西，
所以直接用 dict 里的值（`region_loss` 已乘过 `region_weight`）。

用法
----
  python scripts/tools/probe_region_grad_share.py \
      --run logs/visreg_gemma_regw2.0k12_projective_E_0826_1738 \
      --run logs/visreg_gemma_regw4.0k12_projective_E_0826_2347 --steps 4
`--run` 可给多个，按给定顺序出表。

⚠️ 口径限制（读数前必看）
------------------------
单进程跑，`world_size` 被强制打回 1。后果是**不对称的**：
  - `region_gather=local` → 区域损失本来就不跨卡，**读数与训练一致**。
  - 主对比损失（SigLIP）训练时会再累加 7 个 negative-only 分块
    （8 卡 bidir 交换），单进程只剩本地那一项 → **本脚本低估对比梯度，
    上限约 8 倍**，于是 region/contrastive 比值**最多高估 8 倍**。
本脚本要回答的是"1e-7 的死区还是 1e0 的活区"，差 8 倍不影响这个判断；
但**不要把这里的比值当精确值引用**。要精确值必须 8 卡 torchrun 起，等卡空。

显存：反传全模型，bs=128 约 20GB。默认 128，与训练的 per-GPU 512 不同 ——
比值对 batch 不敏感（两项都随 batch 线性走），绝对值敏感。
"""
import argparse
import ast
import math
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent / "src"))


def read_params(run: Path):
    """params.txt 是 `key: value` 逐行文本，用 literal_eval 尽量还原原类型。

    ⚠️ 不能对 value 做 strip()：`csv_separator` 的值是一个**真的 tab 字符**，
    strip 掉就变成空串，pandas 会报 "only single character unicode strings..."。
    """
    d = {}
    for line in (run / "params.txt").read_text().splitlines():
        if ": " not in line:
            continue
        k, v = line.split(": ", 1)
        try:
            d[k.strip()] = ast.literal_eval(v.strip())
        except (ValueError, SyntaxError):
            d[k.strip()] = v if v.strip() == "" else v.strip()
    return argparse.Namespace(**d)


def build(a, ckpt, device):
    """按 params.txt 复原训练用的 model / loss / preprocess_train，只走区域这条分支。"""
    from torchvision.transforms import Compose, InterpolationMode, Resize
    from open_clip import create_model_and_transforms, create_loss, get_tokenizer
    from open_clip.model import CLIPLeJEPA

    # ⚠️ 必须复刻 main.py:224-232 的 siglip 默认值。踩过的坑：只看 params.txt 里的
    # `init_logit_bias: None` 就不传，模型于是**没有 logit_bias 这个参数**
    # （加载时表现为 unexpected=['logit_bias','clip_model.logit_bias']，很容易忽略），
    # 于是 `_region_loss` 拿到 logit_bias=None → 负样本完全不被 −10 压下去 →
    # 区域损失虚高约 40 倍（实测 per-row 432 vs 训练日志 10.87），比值整个作废。
    kw = {}
    if getattr(a, "siglip", False):
        kw["init_logit_scale"] = math.log(10)
        kw["init_logit_bias"] = -10
    if getattr(a, "init_logit_scale", None) is not None:
        kw["init_logit_scale"] = a.init_logit_scale
    if getattr(a, "init_logit_bias", None) is not None:
        kw["init_logit_bias"] = a.init_logit_bias
    model, pre_train, pre_val = create_model_and_transforms(
        a.model, "", precision=a.precision, device="cpu",
        force_context_length=getattr(a, "force_context_length", None),
        output_dict=True, **kw)

    assert getattr(a, "image_resize_only", False), \
        "本脚本只覆盖 image_resize_only=True 的区域配方（region_crop_aug 未接）"
    sz = pre_val.transforms[0].size
    sz = sz if isinstance(sz, (tuple, list)) else (sz, sz)
    tail = [t for t in pre_val.transforms
            if type(t).__name__ not in ("Resize", "CenterCrop")]
    pre_train = Compose([Resize(sz, interpolation=InterpolationMode.BICUBIC,
                                antialias=True)] + tail)

    model = CLIPLeJEPA(
        clip_model=model, sigreg_target=a.sigreg_target,
        proj_dim=getattr(a, "sigreg_proj_dim", 512),
        proj_layers=getattr(a, "sigreg_proj_layers", 3), output_dict=True,
        pcm_dim=getattr(a, "pcm_dim", 0) if getattr(a, "pcm_weight", 0.0) > 0 else 0,
        region_own_scale=(a.region_weight > 0 and not getattr(a, "region_shared_scale", False)),
        region_boxtext_head=(a.region_weight > 0
                             and not getattr(a, "region_no_boxtext_head", False)),
    )
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"    加载: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    # 硬断言：本脚本要复刻训练前向，任何一个对不上的 key 都可能让损失量级整体跑偏
    # （logit_bias 那次就是这么错的）。宁可报错也不要出一张假表。
    assert not missing and not unexpected, \
        f"模型与 ckpt 不完全对齐 → 读数不可信。missing={missing} unexpected={unexpected}"

    # ⚠️ 单进程跑，必须把 world_size 打回 1，否则 SigLipLoss 会去做 bidir P2P 交换
    # 而进程组没初始化。代价见文件头「口径限制」：主对比损失只剩本地负样本。
    a.world_size, a.rank, a.local_rank = 1, 0, 0
    loss = create_loss(a)
    return model.to(device).train(), loss, pre_train, get_tokenizer(a.model)


def make_loader(a, pre_train, tokenizer, batch, n_batches, seed=0):
    from torch.utils.data import DataLoader
    from open_clip_train.data import CsvDataset
    ds = CsvDataset(a.train_data, pre_train, a.csv_img_key, a.csv_caption_key,
                    sep=a.csv_separator, tokenizer=tokenizer,
                    region_key=getattr(a, "csv_region_key", None),
                    max_region=getattr(a, "max_region", 12))
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch, shuffle=True, generator=g,
                      num_workers=8, drop_last=True, persistent_workers=False), \
        ds, n_batches


def grad_norm(model, loss_val, retain):
    """反传单项损失，返回它对 visual backbone 参数的梯度 L2 范数。"""
    model.zero_grad(set_to_none=True)
    loss_val.backward(retain_graph=retain)
    g2 = 0.0
    for p in model.clip_model.visual.parameters():
        if p.grad is not None:
            g2 += float(p.grad.detach().float().pow(2).sum())
    return math.sqrt(g2)


def probe(a, ckpt, device, batch, steps):
    model, loss_fn, pre_train, tok = build(a, ckpt, device)
    loader, _, _ = make_loader(a, pre_train, tok, batch, steps)
    # 训练用 amp_bf16；梯度比值对 dtype 敏感度低，但为可复现仍按 params.txt 走
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if "bf16" in str(a.precision) \
        else torch.autocast("cuda", enabled=False)

    acc, nseen, nvalid_sum = {}, 0, 0.0
    for bi, batch_data in enumerate(loader):
        if bi >= steps:
            break
        images, texts, rtexts, rboxes, rnvalid = batch_data
        images = images.to(device, non_blocking=True)
        texts, rtexts = texts.to(device), rtexts.to(device)
        rboxes, rnvalid = rboxes.to(device), rnvalid.to(device)
        nvalid_sum += float(rnvalid.float().mean())
        with amp:
            out = model(images, texts, None, rtexts, rboxes, rnvalid)
            losses = loss_fn(**out, output_dict=True)
        keys = list(losses.keys())
        for j, k in enumerate(keys):
            gn = grad_norm(model, losses[k], retain=(j < len(keys) - 1))
            acc[k] = acc.get(k, 0.0) + gn
            acc["VAL " + k] = acc.get("VAL " + k, 0.0) + float(losses[k].detach())
        nseen += 1
        print(f"    step {bi}: " + "  ".join(
            f"{k}|g|={acc[k]/nseen:.4g}" for k in keys), flush=True)
        # ⚠️ 必须在下一次前向之前把梯度和图都放掉。踩过的坑：只在 grad_norm 开头
        # zero_grad，于是最后一项损失的梯度会一直占着（全模型一份梯度，Gemma 文本塔
        # 很大），加上还活着的 out/losses 计算图 → step 1 前向直接 OOM，
        # 表现为「step 0 出数了，step 1 挂」。
        del out, losses
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()
    return {k: v / max(nseen, 1) for k, v in acc.items()}, nvalid_sum / max(nseen, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="logs/<run> 目录，可给多个；tag 取目录名里的 regw* 段")
    ap.add_argument("--epoch", default="epoch_10.pt")
    ap.add_argument("--batch", type=int, default=128,
                    help="默认 128 而非训练的 512（反传显存）；比值对 batch 不敏感但绝对值敏感")
    ap.add_argument("--steps", type=int, default=4)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    res, tags = {}, []
    for r in args.run:
        run = Path(r)
        ck = run / "checkpoints" / args.epoch
        assert ck.exists(), f"ckpt 不存在：{ck}"
        a = read_params(run)
        tag = run.name.split("_")[2] if len(run.name.split("_")) > 2 else run.name
        print(f"=> {tag}  region_weight={getattr(a, 'region_weight', 0.0)}", flush=True)
        res[tag], nv = probe(a, str(ck), device, args.batch, args.steps)
        res[tag]["n_valid/图"] = nv
        tags.append(tag)

    keys = sorted({k for t in tags for k in res[t]})
    W = max(11, max(len(t) for t in tags) + 2)
    print(f"\n{'=' * (26 + W * len(tags))}")
    print(f"backbone 梯度 L2 范数（batch={args.batch}, {args.steps} 步平均）"
          f"；VAL* 行是同一 batch 的损失值")
    print(f"{'量':<26}" + "".join(f"{t:>{W}}" for t in tags))
    print("-" * (26 + W * len(tags)))
    for k in keys:
        print(f"{k:<26}" + "".join(
            f"{res[t].get(k, float('nan')):>{W}.4g}" for t in tags))

    print("\n占比（区域两项的梯度 / 对比损失的梯度）—— 这是本脚本要的那个数：")
    for t in tags:
        c = res[t].get("contrastive_loss", float("nan"))
        rg = res[t].get("region_loss", 0.0)
        cc = res[t].get("region_cc_loss", 0.0)
        sg = res[t].get("sigreg_loss", 0.0)
        vc = res[t].get("VAL contrastive_loss", float("nan"))
        vr = res[t].get("VAL region_loss", 0.0)
        print(f"  {t:<12} region/contrastive 梯度 = {rg / c:.4g}"
              f"   region_cc/contrastive = {cc / c:.4g}"
              f"   visreg/contrastive = {sg / c:.4g}"
              f"   |  损失值占比 region/(总) = {vr / (vc + vr + cc + sg + 1e-12):.3f}")
    print("\n判读：VISReg 的对照数是 2.06e-07（死区）。区域这一项若在 1e-2~1e0 量级，"
          "说明它确实在有效范围内被拉动，\n      "
          "『W 还能往上开』有原理支撑；若已 ≫1，则 W=8.0 很可能只是让主损失让位。")


if __name__ == "__main__":
    main()
