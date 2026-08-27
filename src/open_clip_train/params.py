import argparse
import ast


def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            key, value = value.split('=')
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(value)  # fallback to string (avoid need to escape on command line)
        setattr(namespace, self.dest, kw)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to file(s) with training data. When using webdataset, "
        "multiple datasources can be combined using the `::` separator.",
    )
    parser.add_argument(
        "--train-data-upsampling-factors",
        type=str,
        default=None,
        help=(
            "When using multiple data sources with webdataset and sampling with replacement, this can be used to upsample specific data sources. "
            "Similar to --train-data, this should be a string with as many numbers as there are data sources, separated by `::` (e.g. 1::2::0.5) "
            "By default, datapoints are sampled uniformly regardless of the dataset sizes."
        )
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to file(s) with validation data",
    )
    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Required for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--val-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Useful for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["webdataset", "csv", "video_frame", "synthetic", "auto"],
        default="auto",
        help="Which type of dataset to process."
    )
    parser.add_argument(
        "--dataset-resampled",
        default=False,
        action="store_true",
        help="Whether to use sampling with replacement for webdataset shard selection."
    )
    parser.add_argument(
        "--csv-separator",
        type=str,
        default="\t",
        help="For csv-like datasets, which separator to use."
    )
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="filepath",
        help="For csv-like datasets, the name of the key for the image paths."
    )
    parser.add_argument(
        "--csv-caption-key",
        type=str,
        default="title",
        help="For csv-like datasets, the name of the key for the captions."
    )
    parser.add_argument(
        "--csv-caption2-key",
        type=str,
        default=None,
        help="Optional second caption column (DualTextCLIP: short+long text towers)."
    )
    parser.add_argument(
        "--imagenet-val",
        type=str,
        default=None,
        help="Path to imagenet val set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet-v2",
        type=str,
        default=None,
        help="Path to imagenet v2 for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Override system default cache path for model & tokenizer file downloads.",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="log files on local master, otherwise global master only.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--epochs-cooldown", type=int, default=None,
        help="When scheduler w/ cooldown used, perform cooldown from total_epochs - cooldown_epochs onwards."
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument("--momentum", type=float, default=None, help="Momentum (for timm optimizers).")
    parser.add_argument(
        "--warmup", type=int, default=10000, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--opt", type=str, default='adamw',
        help="Which optimizer to use. Choices are ['adamw', 'muon', or any timm optimizer 'timm/{opt_name}']. "
             "'muon' uses MuonWithAuxAdam: Muon for hidden weight matrices (ndim>=2, non-embed/bias) + AdamW for the rest."
    )
    parser.add_argument(
        "--muon-lr", type=float, default=None,
        help="Muon learning rate for hidden weight matrices. Defaults to --lr if not set. "
             "Muon lr semantics differ from Adam lr (spectral norm units); typical range 0.01-0.05."
    )
    parser.add_argument(
        "--muon-momentum", type=float, default=0.95,
        help="Muon momentum (default: 0.95)."
    )
    parser.add_argument(
        "--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.")
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default='cosine',
        help="LR scheduler. One of: 'cosine', 'const' (constant), "
        "'const-cooldown' (constant w/ cooldown). Default: cosine",
    )
    parser.add_argument(
        "--lr-cooldown-end", type=float, default=0.0,
        help="End learning rate for cooldown schedule. Default: 0"
    )
    parser.add_argument(
        "--lr-cooldown-power", type=float, default=1.0,
        help="Power for polynomial cooldown schedule. Default: 1.0 (linear decay)"
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=False,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--zeroshot-frequency", type=int, default=2, help="How often to run zero shot."
    )
    parser.add_argument(
        "--val-frequency", type=int, default=1, help="How often to run evaluation with val data."
    )
    parser.add_argument(
        "--val-num-captions-per-image", type=int, default=1,
        help="论文标准多caption eval: 设为5时使用 clip_val_coco5k.tsv（每张图5条caption），"
             "get_clip_metrics 自动处理 1-to-5 ground truth。默认 1 (1:1 匹配)。"
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "pure_bf16", "pure_fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RN50",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--pretrained",
        default='',
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--pretrained-image",
        default=False,
        action='store_true',
        help="Load imagenet pretrained weights for image tower backbone if available.",
    )
    parser.add_argument(
        "--pretrained-text-path",
        default=None,
        type=str,
        help="Path to checkpoint for text tower weights only. "
             "Supports full CLIP checkpoints (auto-strips 'text.' prefix).",
    )
    parser.add_argument(
        "--text-proj-type",
        default=None,
        type=str,
        choices=["linear", "mlp"],
        help="Override text projection type. 'mlp' replaces linear with 2-layer MLP (for reverse-LiT bridge).",
    )
    # ── Dual-teacher mode ──────────────────────────────────────────────────
    parser.add_argument(
        "--dual-teacher",
        default=False,
        action='store_true',
        help="Enable dual-teacher mode: one image encoder aligns with two frozen text encoders.",
    )
    parser.add_argument(
        "--dual-text",
        default=False,
        action='store_true',
        help="Enable DualTextCLIP: shared image tower + two trainable text towers (short+long), "
             "dual SigLIP alignment. Requires --csv-caption2-key (long caption column) "
             "and --tokenizer-secondary is auto-created.",
    )
    parser.add_argument(
        "--dual-cls",
        default=False,
        action='store_true',
        help="Use dual MAP pooling (two independent latent queries) instead of shared pool.",
    )
    parser.add_argument(
        "--teacher-pe-ckpt",
        default=None,
        type=str,
        help="Path to PE-Core teacher checkpoint (full CLIP model).",
    )
    parser.add_argument(
        "--teacher-sig-ckpt",
        default=None,
        type=str,
        help="Path to SigLIP2 teacher checkpoint (full CLIP model).",
    )
    parser.add_argument(
        "--teacher-sig-model",
        default=None,
        type=str,
        help="SigLIP2 model config name (for tokenizer resolution, e.g. 'local-dir:/path/to/ViT-B-16-SigLIP2').",
    )
    parser.add_argument(
        "--pretrained-image-init",
        default=None,
        type=str,
        help="Path to checkpoint for image backbone initialization in dual-teacher mode.",
    )

    # ── Multi-teacher mode ──────────────────────────────────────────────────
    parser.add_argument(
        "--multi-teacher",
        default=False,
        action='store_true',
        help="Enable multi-teacher mode: one image encoder aligns with N frozen text encoders.",
    )
    parser.add_argument(
        "--teachers",
        default=None,
        type=str,
        help="Comma-separated 'model_name::ckpt_path' pairs for multi-teacher mode.",
    )
    parser.add_argument(
        "--teacher-weights",
        default=None,
        type=str,
        help="Comma-separated loss weights per teacher. Default: equal weights (1.0 each).",
    )

    parser.add_argument(
        "--lock-image",
        default=False,
        action='store_true',
        help="Lock full image tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-image-unlocked-groups",
        type=int,
        default=0,
        help="Leave last n image tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-image-freeze-bn-stats",
        default=False,
        action='store_true',
        help="Freeze BatchNorm running stats in image tower for any locked layers.",
    )
    parser.add_argument(
        '--image-mean', type=float, nargs='+', default=None, metavar='MEAN',
        help='Override default image mean value of dataset')
    parser.add_argument(
        '--image-std', type=float, nargs='+', default=None, metavar='STD',
        help='Override default image std deviation of of dataset')
    parser.add_argument(
        '--image-interpolation',
        default=None, type=str, choices=['bicubic', 'bilinear', 'random'],
        help="Override default image resize interpolation"
    )
    parser.add_argument(
        '--image-resize-mode',
        default=None, type=str, choices=['shortest', 'longest', 'squash'],
        help="Override default image resize (& crop) mode during inference"
    )
    parser.add_argument('--aug-cfg', nargs='*', default={}, action=ParseKwargs)
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action='store_true',
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--local-loss",
        default=False,
        action="store_true",
        help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)"
    )
    parser.add_argument(
        "--gather-with-grad",
        default=False,
        action="store_true",
        help="enable full distributed gradient for feature gather"
    )
    parser.add_argument(
        '--force-context-length', type=int, default=None,
        help='Override default context length'
    )
    parser.add_argument(
        '--force-image-size', type=int, nargs='+', default=None,
        help='Override default image size'
    )
    parser.add_argument(
        "--force-quick-gelu",
        default=False,
        action='store_true',
        help="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )
    parser.add_argument(
        "--force-patch-dropout",
        default=None,
        type=float,
        help="Override the patch dropout during training, for fine tuning with no dropout near the end as in the paper",
    )
    parser.add_argument(
        "--force-custom-text",
        default=False,
        action='store_true',
        help="Force use of CustomTextCLIP model (separate text-tower).",
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action='store_true',
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )
    parser.add_argument(
        "--torchcompile",
        default=False,
        action='store_true',
        help="torch.compile() the model, requires pytorch 2.0 or later.",
    )
    parser.add_argument(
        "--trace",
        default=False,
        action='store_true',
        help="torch.jit.trace the model for inference / eval only",
    )
    parser.add_argument(
        "--accum-freq", type=int, default=1, help="Update the model every --acum-freq steps."
    )
    parser.add_argument(
        "--device", default="cuda", type=str, help="Accelerator to use."
    )
    # arguments for distributed training
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend",
        default=None,
        type=str,
        help="distributed backend. \"nccl\" for GPU, \"hccl\" for Ascend NPU"
    )
    parser.add_argument(
        "--report-to",
        default='',
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']"
    )
    parser.add_argument(
        "--wandb-notes",
        default='',
        type=str,
        help="Notes if logging with wandb"
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default='open-clip',
        help="Name of the project if logging with wandb.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log directory, and execute from there."
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training."
    )
    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action='store_true',
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc)."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )
    parser.add_argument(
        "--grad-clip-norm", type=float, default=None, help="Gradient clip."
    )
    parser.add_argument(
        "--lock-text",
        default=False,
        action='store_true',
        help="Lock full text tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-text-unlocked-layers",
        type=int,
        default=0,
        help="Leave last n text tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-text-freeze-layer-norm",
        default=False,
        action='store_true',
        help="Freeze LayerNorm running stats in text tower for any locked layers.",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=100,
        help="Log every n steps to tensorboard/console/wandb.",
    )
    parser.add_argument(
        "--coca-caption-loss-weight",
        type=float,
        default=2.0,
        help="Weight assigned to caption loss in CoCa."
    )
    parser.add_argument(
        "--coca-contrastive-loss-weight",
        type=float,
        default=1.0,
        help="Weight assigned to contrastive loss when training CoCa."
    )
    parser.add_argument(
        "--remote-sync",
        type=str,
        default=None,
        help="Optinoally sync with a remote path specified by this arg",
    )
    parser.add_argument(
        "--remote-sync-frequency",
        type=int,
        default=300,
        help="How frequently to sync to a remote directly if --remote-sync is not None.",
    )
    parser.add_argument(
        "--remote-sync-protocol",
        choices=["s3", "fsspec"],
        default="s3",
        help="How to do the remote sync backup if --remote-sync is not None.",
    )
    parser.add_argument(
        "--delete-previous-checkpoint",
        default=False,
        action="store_true",
        help="If true, delete previous checkpoint after storing a new one."
    )
    parser.add_argument(
        "--distill-model",
        default=None,
        help='Which model arch to distill from, if any.'
    )
    parser.add_argument(
        "--distill-pretrained",
        default=None,
        help='Which pre-trained weights to distill from, if any.'
    )
    parser.add_argument(
        "--use-bnb-linear",
        default=None,
        help='Replace the network linear layers from the bitsandbytes library. '
        'Allows int8 training/inference, etc.'
    )
    parser.add_argument(
        "--siglip",
        default=False,
        action="store_true",
        help='Use SigLip (sigmoid) loss.'
    )
    parser.add_argument(
        "--neg-mode",
        default='standard',
        type=str,
        choices=['standard', 'antipodal', 'orthogonal', 'projective'],
        help=(
            'Negative pair geometry in SigLIP loss. '
            '"standard": cos→+1 for pos, cos→-1 for neg. '
            '"antipodal": cos→-1 for pos, cos≠-1 for neg. '
            '"orthogonal": cos→+1 for pos, |cos|→0 (cos→0) for neg. '
            '"projective": |cos|→1 for pos (collinear), |cos|→0 for neg (orthogonal). '
            'Requires --siglip.'
        )
    )
    parser.add_argument(
        "--region-weight",
        default=0.0,
        type=float,
        help=(
            '区域-短语对比分支权重（FG-CLIP 式，ICML 2025）。>0 时启用：'
            '从倒数第二层 patch feature map 用 RoIAlign 抠区域特征，与短语做对比。'
            '需配合 --csv-region-key（JSON 列 [[phrase,x1,y1,x2,y2],...]，坐标归一化）'
            '与 --image-resize-only（区域坐标要求不做随机裁剪）。'
            'FG-CLIP 原论文用 0.1。'
        )
    )
    parser.add_argument(
        "--region-cc-weight", default=0.0, type=float,
        help=(
            '区域短语之间的类别对比权重（FG-CLIP2 hard_category_contrastive_loss，'
            '原论文 region_cc_loss_weight=0.1）。推开同 batch 内不同区域的短语 embedding，'
            '防止高频短语（eyes/nose/ears）塌成一团。'
        )
    )
    parser.add_argument(
        "--region-no-boxtext-head", default=False, action="store_true",
        help=(
            '区域文本不用独立投影头，与整句共用 text_projection。'
            '默认关（即用独立 head，对齐 FG-CLIP2 的 boxtext_head）。消融用。'
        )
    )
    parser.add_argument(
        "--region-gather", default="local", type=str, choices=["local", "gather"],
        help=(
            '区域分支负样本池范围。local=只在 rank 内（FG-CLIP 的做法，矩阵 n^2）；'
            'gather=跨卡收集文本侧扩大负样本池 world 倍（矩阵 n x n_total，按行分块算）。'
            '区域数是 batch 的 K 倍，gather 的矩阵按平方放大 —— K=12/B=512/world=8 时'
            '全量要 4.8GB，故分块。两者性能差异待实验。'
        )
    )
    parser.add_argument(
        "--region-shared-scale", default=False, action="store_true",
        help=(
            '区域分支与全图共用 logit_scale。默认关（用独立温度，对齐 FG-CLIP 的 '
            'logit_scale_finegrained）——区域对齐的难度与全图不同，共用可能互相拖累。'
        )
    )
    parser.add_argument(
        "--region-roi-grid", default=1, type=int,
        help=(
            'roi_align 的输出网格边长 S（区域特征取 S×S 个格子）。1=历史行为，'
            '**与已有全部 run 逐位相同**（单格的均值就是它自己）。'
            '只有配合 --region-roi-agg=mil 时 S>1 才改变目标函数。'
        )
    )
    parser.add_argument(
        "--region-roi-agg", default="mean", type=str, choices=["mean", "mil"],
        help=(
            'S×S 个格子怎么用。mean=先平均成一个区域向量（S>1 时只是采样密度变了，'
            '近似 no-op 对照臂）；mil=不平均，每格各自与短语算相似度、'
            '损失侧对格子取 max（"框内任一格子命中即算命中"）。'
            'mil 的动机：高 region_weight 牺牲的是细长结构类，'
            '1×1 均值池化把细长目标稀释进了框内背景。仅支持 --region-gather=local。'
        )
    )
    parser.add_argument(
        "--csv-region-key", default=None, type=str,
        help='TSV 里区域列的列名（如 regions）。见 scripts/data/build_region_tsv.py'
    )
    parser.add_argument(
        "--max-region", default=12, type=int,
        help='每图最多用多少个区域（实测 p90=11，成本随此值线性增长）'
    )
    parser.add_argument(
        "--region-crop-aug", default=False, action="store_true",
        help=(
            '区域监督下启用随机裁剪，框随裁剪同步变换（完全包含策略：裁出画面的框丢弃）。'
            '动机：关掉 RandomResizedCrop 实测代价 COCO i2t −1.70 / IN-1k −0.70（均超 2σ），'
            '这部分正则化收益不该白丢。优先级高于 --image-resize-only。'
        )
    )
    parser.add_argument(
        "--image-resize-only", default=False, action="store_true",
        help=(
            '训练图像变换只做 resize 到目标尺寸，不做 RandomResizedCrop。'
            '★ 用区域监督时必须开 ★ —— 区域坐标是相对原图归一化的，'
            '随机裁剪会让坐标失效（FG-CLIP 同样直接 resize）。'
        )
    )
    parser.add_argument(
        "--pcm-weight",
        default=0.0,
        type=float,
        help=(
            'Primary Component Matching (Long-CLIP, ECCV 2024) 短文本分支权重。'
            '>0 时启用：主分支用长文本(--csv-caption-key)对齐图像，短分支用 '
            'PCA(--pcm-dim) 降维后的图像特征对齐短文本(--csv-caption2-key)，'
            '让单塔同时保住长文本能力与短模板 zero-shot 能力。'
            '需配合 --pcm-dim 与双列 TSV（filepath/caption_dense/caption_short）。'
        )
    )
    parser.add_argument(
        "--pcm-dim",
        default=32,
        type=int,
        help='PCM 短分支的 PCA 保留维度（Long-CLIP 原论文用 32）。0 表示不降维。'
    )
    parser.add_argument(
        "--init-logit-scale",
        default=None,
        type=float,
        help='Override init logit_scale (log-space). Default: ln(10) for SigLIP, ln(1/0.07) for CLIP.'
    )
    parser.add_argument(
        "--init-logit-bias",
        default=None,
        type=float,
        help='Override init logit_bias. Default: -10 for SigLIP, None for CLIP.'
    )
    parser.add_argument(
        "--freeze-logit-params",
        default=False,
        action="store_true",
        help='Freeze logit_scale and logit_bias (non-learnable).'
    )
    parser.add_argument(
        "--neg-alpha",
        type=float,
        default=1.0,
        help=(
            'Blend factor between standard (1.0) and projective (0.0) similarity. '
            'logits = alpha * (scale*cos) + (1-alpha) * (scale*|cos|). '
            'alpha=0.5 is "half-orthogonal" (cos<0 neutral). Overrides --neg-mode when <1.0. '
            'Requires --siglip.'
        )
    )
    parser.add_argument(
        "--loss-dist-impl",
        default=None,
        type=str,
        help='A string to specify a specific distributed loss implementation.'
    )

    # ============ SIGReg 正则化参数 (https://arxiv.org/abs/2511.08544) ============
    parser.add_argument(
        "--sigreg-target",
        default="none",
        choices=["none", "clip", "clip_proj", "cls", "cls_proj"],
        help=(
            'SIGReg regularization target. '
            '"none": disabled. '
            '"clip": act on CLIP embedding [B, clip_dim] (Identity, no extra MLP). '
            '"clip_proj": act on MLP output built on top of CLIP embedding. '
            '"cls": act on CLS raw backbone embedding [B, backbone_dim] (Identity). '
            '"cls_proj": act on MLP output built on top of CLS raw. '
            'When --dinov3 is on, "cls"/"cls_proj" share the KoLeo position (pre-dino-head). '
            'Works with --siglip.'
        )
    )
    parser.add_argument(
        "--sigreg-weight",
        type=float,
        default=1e-4,
        help='SIGReg loss weight lambda. Literature range: 1e-4 to 1e-2.'
    )
    parser.add_argument(
        "--sigreg-proj-dim",
        type=int,
        default=512,
        help='Projector output dimension (only used with clip_proj / cls_proj targets).'
    )
    parser.add_argument(
        "--sigreg-proj-layers",
        type=int,
        default=3,
        help='Projector MLP depth (only used with clip_proj / cls_proj targets).'
    )
    # ── CLIP 空间 embedding 噪声（NOVIC 风格训练增强）────────────────────
    parser.add_argument(
        "--noise-scheme",
        default="",
        choices=["", "gausselem", "uniformangle", "gausselemuniformangle"],
        help='CLIP embedding noise scheme for contrastive training. '
             '"gausselem": element Gaussian (vec_norm); "uniformangle": rotate by angle in [min,max]; '
             '"gausselemuniformangle": mix by ratio. Empty = disabled (default).'
    )
    parser.add_argument(
        "--noise-vec-norm",
        type=float,
        default=3.25,
        help='GaussElem noise vector norm (NOVIC default 3.25).'
    )
    parser.add_argument(
        "--noise-angle-min",
        type=float,
        default=45.0,
        help='UniformAngle min rotation in degrees (NOVIC default 45).'
    )
    parser.add_argument(
        "--noise-angle-max",
        type=float,
        default=75.0,
        help='UniformAngle max rotation in degrees (NOVIC default 75).'
    )
    parser.add_argument(
        "--noise-mix-ratio",
        type=float,
        default=0.15,
        help='Mix ratio of UniformAngle in gausselemuniformangle (NOVIC default 0.15).'
    )
    parser.add_argument(
        "--noise-sides",
        default="both",
        choices=["both", "img", "txt"],
        help='Which side(s) get noise in CLIP space: both (default) / img / txt.'
    )
    parser.add_argument(
        "--sigreg-slices",
        type=int,
        default=256,
        help='Number of random slices for the SIGReg estimator.'
    )
    parser.add_argument(
        "--reg-method",
        default="sigreg",
        choices=["sigreg", "visreg"],
        help=(
            'Which regularizer to use at the sigreg-target position. '
            '"sigreg" (default): Sketched Isotropic Gaussian Regularization (LeJEPA, 2511.08544). '
            '"visreg": Variance-Invariance-Sketching Regularization (2606.02572) — scale+shape(SWD)+center. '
            'VISReg is batch-invariant, so --sigreg-weight must be re-calibrated (much larger).'
        )
    )
    parser.add_argument(
        "--visreg-lambda-scale",
        type=float,
        default=1.0,
        help='VISReg variance (scale) term weight. 0 disables it.'
    )
    parser.add_argument(
        "--visreg-lambda-shape",
        type=float,
        default=1.0,
        help='VISReg Sliced-Wasserstein (shape) term weight. 0 disables it.'
    )
    parser.add_argument(
        "--visreg-lambda-center",
        type=float,
        default=1.0,
        help='VISReg centering term weight. 0 disables it.'
    )
    parser.add_argument(
        "--visreg-topk-pool",
        type=int,
        default=0,
        help=(
            'VISReg shape: sample this many candidate directions, then keep the K worst '
            '(K = --sigreg-slices) for the gradient. 0 = disabled (pure random). '
            'Selection runs under no_grad; loss value becomes a biased SWD estimate, '
            'but gradients focus on genuinely deviating directions.'
        )
    )
    parser.add_argument(
        "--visreg-mixture",
        type=int,
        default=0,
        help=(
            'VISReg shape target = M-component equal-weight Gaussian mixture instead of a '
            'single standard Gaussian. 0/1 = standard Gaussian. Motivation: real CLIP '
            'features are multi-island (measured 66-68%% nearest-neighbour same-cluster rate); '
            'forcing a unimodal target at high weight destroys semantic clustering.'
        )
    )
    parser.add_argument(
        "--reg-sides",
        type=str, default='both', choices=['both', 'img', 'txt'],
        help=(
            'Which tower the regularizer acts on. "both" (default, current recipe -- note '
            'the text tower has been regularized all along), "img" / "txt" for ablation.'
        )
    )
    parser.add_argument(
        "--xmatch-weight",
        type=float, default=0.0,
        help=(
            'Cross-modal match auxiliary loss weight. Projects BOTH towers onto the SAME '
            'random directions. 0 = disabled. Requires --sigreg-target clip/clip_proj '
            '(both towers must share a dimension). Auxiliary only -- never a replacement '
            'for the contrastive loss.'
        )
    )
    parser.add_argument(
        "--xmatch-mode",
        type=str, default='pair', choices=['pair', 'dist'],
        help=(
            '"pair": per-sample alignment on projections, keeps pairing identity '
            '(= random-projection estimate of per-pair MSE, gentler than plain MSE). '
            '"dist": sorted-shape + std alignment -- permutation invariant, carries NO '
            'pairing signal, so it must stay a small auxiliary term.'
        )
    )
    parser.add_argument(
        "--visreg-mixture-sep",
        type=float,
        default=2.0,
        help='Spacing between mixture component centres, in units of sigma. Used with --visreg-mixture.'
    )

    # ============ Modality Gap ============
    parser.add_argument(
        "--modality-gap-weight",
        type=float,
        default=0.0,
        help=(
            'Batch gap loss weight λ: λ * ||mean(img_raw) - mean(txt_raw)||². '
            'Gradient flows through batch means. Applied pre-L2-norm. 0.0 = disabled.'
        )
    )
    parser.add_argument(
        "--within-modal-weight",
        type=float,
        default=0.0,
        help=(
            'Weight for within-modality SigLIP repulsion loss. '
            'Adds image-image and/or text-text all-negative sigmoid losses '
            '(diagonal masked). Pushes same-modality features apart on the '
            'hypersphere. Default 0.0 = disabled.'
        )
    )
    parser.add_argument(
        "--within-modal-sides",
        type=str,
        default='both',
        choices=['both', 'img', 'txt'],
        help=(
            'Which modality to apply within-modal repulsion to. '
            '"both" (default): image-image + text-text; '
            '"img": image-image only; '
            '"txt": text-text only.'
        )
    )
    parser.add_argument(
        "--within-modal-mode",
        type=str,
        default='replace',
        choices=['replace', 'auxiliary'],
        help=(
            'How within-modal loss interacts with cross-modal loss. '
            '"replace" (default): remove cross-modal negatives, use positive-only + within-modal. '
            '"auxiliary": keep full SigLIP (cross-modal pos+neg) and ADD within-modal as extra regularizer.'
        )
    )

    # ============ Positive-only (no negatives) ============
    parser.add_argument(
        "--pos-only",
        type=str,
        default='none',
        choices=['none', 'sigmoid', 'mse'],
        help=(
            'Positive-only cross-modal loss (no negatives). '
            '"sigmoid": -logsigmoid(scale*cos+bias) on matched pairs. '
            '"mse": (1-cos)^2 on matched pairs. '
            'Requires --sigreg-target for regularization.'
        )
    )
    parser.add_argument(
        "--sigreg-joint",
        default=False,
        action='store_true',
        help='SIGReg on concatenated [img;txt] (joint isotropy) instead of separately.'
    )

    # ============ Representation uniformity losses ============
    parser.add_argument(
        "--uniformity-weight",
        type=float,
        default=0.0,
        help=(
            'Weight for Wang & Isola (2020) uniformity loss on L2-normalized CLIP features. '
            'Applied to both image and text. 0.0 = disabled.'
        )
    )
    parser.add_argument(
        "--uniformity-t",
        type=float,
        default=2.0,
        help='Temperature for uniformity loss kernel: exp(-t * ||z_i - z_j||^2). Default 2.0.'
    )
    parser.add_argument(
        "--koleo-weight",
        type=float,
        default=0.0,
        help=(
            'Weight for KoLeo nearest-neighbor entropy loss on L2-normalized CLIP features. '
            'Applied to both image and text. 0.0 = disabled. '
            'Note: this is independent of the DINOv3 --koleo-loss-weight.'
        )
    )

    # ============ DINOv3 自蒸馏参数 ============
    parser.add_argument(
        "--dinov3",
        default=False,
        action="store_true",
        help="Enable DINOv3-style self-distillation (DINO + iBOT + KoLeo) alongside contrastive loss."
    )
    parser.add_argument(
        "--dino-loss-weight",
        type=float,
        default=1.0,
        help="Weight for DINO CLS token self-distillation loss."
    )
    parser.add_argument(
        "--ibot-loss-weight",
        type=float,
        default=1.0,
        help="Weight for iBOT masked patch token self-distillation loss."
    )
    parser.add_argument(
        "--koleo-loss-weight",
        type=float,
        default=0.1,
        help="Weight for KoLeo nearest-neighbor entropy regularizer."
    )
    parser.add_argument(
        "--dino-head-prototypes",
        type=int,
        default=65536,
        help="Number of prototypes in DINO head (out_dim)."
    )
    parser.add_argument(
        "--ibot-head-prototypes",
        type=int,
        default=65536,
        help="Number of prototypes in iBOT head (out_dim). Defaults to --dino-head-prototypes."
    )
    parser.add_argument(
        "--dino-head-nlayers",
        type=int,
        default=3,
        help="Number of MLP layers in DINO/iBOT projection heads."
    )
    parser.add_argument(
        "--dino-head-hidden-dim",
        type=int,
        default=2048,
        help="Hidden dimension of DINO/iBOT MLP heads."
    )
    parser.add_argument(
        "--dino-head-bottleneck-dim",
        type=int,
        default=256,
        help="Bottleneck dimension of DINO/iBOT MLP heads (before L2-norm)."
    )
    parser.add_argument(
        "--dino-student-temp",
        type=float,
        default=0.1,
        help="Student temperature for DINO/iBOT cross-entropy."
    )
    parser.add_argument(
        "--dino-teacher-temp",
        type=float,
        default=0.07,
        help="Final teacher temperature (after warmup)."
    )
    parser.add_argument(
        "--dino-warmup-teacher-temp",
        type=float,
        default=0.04,
        help="Starting teacher temperature during warmup."
    )
    parser.add_argument(
        "--dino-warmup-teacher-temp-epochs",
        type=int,
        default=30,
        help="Number of epochs to warm up teacher temperature."
    )
    parser.add_argument(
        "--dino-teacher-momentum",
        type=float,
        default=0.992,
        help="Starting EMA momentum for teacher (cosine-scheduled to 1.0)."
    )
    parser.add_argument(
        "--dino-n-global-crops",
        type=int,
        default=2,
        help="Number of global crops for DINO/iBOT. "
             "Set to 1 for local-to-global only DINO (no global-to-global pairs). "
             "iBOT always uses global crops regardless of this setting."
    )
    parser.add_argument(
        "--dino-local-crops-number",
        type=int,
        default=8,
        help="Number of local crops for DINO. Set 0 to disable local crops."
    )
    parser.add_argument(
        "--dino-local-crops-size",
        type=int,
        default=96,
        help="Pixel size of local crops."
    )
    parser.add_argument(
        "--dino-global-crops-scale",
        type=float,
        nargs=2,
        default=[0.32, 1.0],
        metavar=('MIN', 'MAX'),
        help="Scale range for global RandomResizedCrop."
    )
    parser.add_argument(
        "--dino-local-crops-scale",
        type=float,
        nargs=2,
        default=[0.05, 0.32],
        metavar=('MIN', 'MAX'),
        help="Scale range for local RandomResizedCrop."
    )
    parser.add_argument(
        "--ibot-mask-ratio-min",
        type=float,
        default=0.1,
        help="Minimum iBOT mask ratio (fraction of patches masked)."
    )
    parser.add_argument(
        "--ibot-mask-ratio-max",
        type=float,
        default=0.5,
        help="Maximum iBOT mask ratio."
    )
    parser.add_argument(
        "--ibot-mask-sample-prob",
        type=float,
        default=0.5,
        help="Fraction of samples per batch that receive an iBOT mask."
    )
    parser.add_argument(
        "--freeze-last-layer-epochs",
        type=int,
        default=1,
        help="Freeze the last linear layer of DINO/iBOT heads for this many epochs."
    )

    parser.add_argument(
        "--probe-data",
        type=str,
        default=None,
        help="TSV file (filepath/caption columns) for per-epoch feature probe."
    )
    parser.add_argument(
        "--probe-dir",
        type=str,
        default=None,
        help="Output dir for probe npz files (default: <checkpoint_path>/probe)."
    )
    parser.add_argument(
        "--probe-freq-steps",
        type=int,
        default=None,
        help="Run feature probe every N optimizer steps (inside epoch). "
             "If None, probe runs once per epoch as before. "
             "E.g. set to steps_per_epoch//4 for 4 probes per epoch."
    )

    # ============ Curriculum Learning ============
    parser.add_argument("--curriculum-strategy", type=str, default=None,
        choices=["fps", "fps_reverse", "density_high", "density_low", "curvature_high", "curvature_low"],
        help="Per-epoch sample ordering strategy. None=disabled.")
    parser.add_argument("--curriculum-init", type=str, default="self",
        choices=["dinov3", "pe_core", "pe_core_always", "self", "siglip2", "datacomp", "dfn2b", "eva02", "laion2b", "metaclip", "random_init"],
        help="Feature extractor for curriculum ordering.")
    parser.add_argument("--curriculum-k", type=int, default=50,
        help="kNN K for density/curvature metrics.")
    parser.add_argument("--curriculum-epochs", type=int, default=0,
        help="Number of initial epochs to apply curriculum. 0=all epochs.")

    args = parser.parse_args(args)

    if 'timm' not in args.opt:
        # set default opt params based on model name (only if timm optimizer not used)
        default_params = get_default_params(args.model)
        for name, val in default_params.items():
            if getattr(args, name) is None:
                setattr(args, name, val)

    return args
