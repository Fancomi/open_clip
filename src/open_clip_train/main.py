import copy
import glob
import logging
import math
import os
import re
import subprocess
import sys
import random
from datetime import datetime
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch import optim

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss
from open_clip.factory import attach_modality_modules
from open_clip.model import CLIPLeJEPA, CLIPWithDINO, DualTeacherCLIP, MultiTeacherCLIP, DualTextCLIP
from open_clip_train.data import get_data
from open_clip_train.distributed import is_master, init_distributed_device, broadcast_object
from open_clip_train.logger import setup_logging
from open_clip_train.params import parse_args
from open_clip_train.scheduler import cosine_lr, const_lr, const_lr_cooldown
from open_clip_train.train import train_one_epoch, evaluate
from open_clip_train.file_utils import pt_load, check_exists, start_sync_process, remote_sync


LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote: bool):
    """Get the latest checkpoint from a local or remote path."""
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def main(args):
    args = parse_args(args)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = '-'.join([
            date_str,
            f"model_{model_name_safe}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])

    resume_latest = args.resume == 'latest'
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not resume_latest:
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, args.name, "checkpoints")
            if args.save_most_recent:
                print('Error. Cannot use save-most-recent with remote_sync and resume latest.')
                return -1
            if args.remote_sync_protocol != 's3':
                print('Error. Sync protocol not supported when using resume latest.')
                return -1
        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if args.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)
            if resume_from:
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
        if args.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from

    if args.copy_codebase:
        copy_codebase(args)

    # start the sync proces if remote-sync is not None
    remote_sync_process = None
    if is_master(args) and args.remote_sync is not None:
        # first make sure it works
        result = remote_sync(
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('remote sync successful.')
        else:
            logging.info('Error: remote sync failed. Exiting.')
            return -1
        # if all looks good, start a process to do this every args.remote_sync_frequency seconds
        remote_sync_process = start_sync_process(
            args.remote_sync_frequency,
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        remote_sync_process.start()

    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    dist_model = None
    args.distill = args.distill_model is not None and args.distill_pretrained is not None
    if args.distill:
        #FIXME: support distillation with grad accum.
        assert args.accum_freq == 1
        #FIXME: support distillation with coca.
        assert 'coca' not in args.model.lower()

    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]
    random_seed(args.seed, 0)
    model_kwargs = {}
    if args.siglip:
        model_kwargs['init_logit_scale'] = np.log(10)  # different from CLIP
        model_kwargs['init_logit_bias'] = -10
    # CLI overrides for scale/bias init
    if args.init_logit_scale is not None:
        model_kwargs['init_logit_scale'] = args.init_logit_scale
    if args.init_logit_bias is not None:
        model_kwargs['init_logit_bias'] = args.init_logit_bias
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        force_context_length=args.force_context_length,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,  # only effective for inference
        aug_cfg=args.aug_cfg,
        pretrained_image=args.pretrained_image,
        pretrained_text_path=getattr(args, 'pretrained_text_path', None),
        output_dict=True,
        cache_dir=args.cache_dir,
        **model_kwargs,
    )

    # DualTextCLIP: 共享图像塔 + 双文本塔（短 gt + 长 dense），双 SigLIP 对齐
    if getattr(args, 'dual_text', False):
        from open_clip.model import CLIPTextCfg
        from open_clip.factory import get_model_config
        logging.info("=> Wrapping model with DualTextCLIP (short + long text towers)")
        _cfg = get_model_config(args.model)
        _text_cfg = dict(_cfg['text_cfg'])
        _text_cfg['context_length'] = args.force_context_length or _text_cfg.get('context_length', 256)
        _base_clip = model.clip_model if isinstance(model, CLIPLeJEPA) else model
        model = DualTextCLIP(
            image_backbone=_base_clip.visual,
            text_cfg=CLIPTextCfg(**_text_cfg),
            embed_dim=_base_clip.embed_dim if hasattr(_base_clip, 'embed_dim') else _cfg['embed_dim'],
            backbone_dim=_cfg['embed_dim'],  # _image_feat 用 visual.head 输出 [B, embed_dim]
            init_logit_scale=np.log(10) if getattr(args, 'siglip', False) else np.log(1 / 0.07),
            init_logit_bias=-10.0 if getattr(args, 'siglip', False) else None,
            output_dict=True,
        ).to(device)
        logging.info(f"   DualTextCLIP assembled (embed_dim={_cfg['embed_dim']})")

    # SIGReg (standalone, no DINOv3): wrap model to expose image_proj/text_proj
    sigreg_target = getattr(args, 'sigreg_target', 'none') or 'none'
    if sigreg_target != 'none' and not getattr(args, 'dinov3', False) and not getattr(args, 'dual_text', False):
        logging.info(f"=> Wrapping model with CLIPLeJEPA (sigreg_target={sigreg_target})")
        model = CLIPLeJEPA(
            clip_model=model,
            sigreg_target=sigreg_target,
            proj_dim=getattr(args, 'sigreg_proj_dim', 512),
            proj_layers=getattr(args, 'sigreg_proj_layers', 3),
            output_dict=True,
            noise_scheme=getattr(args, 'noise_scheme', ''),
            noise_vec_norm=getattr(args, 'noise_vec_norm', 3.25),
            noise_angle_min=getattr(args, 'noise_angle_min', 45.0),
            noise_angle_max=getattr(args, 'noise_angle_max', 75.0),
            noise_mix_ratio=getattr(args, 'noise_mix_ratio', 0.15),
            noise_sides=getattr(args, 'noise_sides', 'both'),
        )
        model = model.to(device)

    # DINOv3: 包装模型，增加 student-teacher 自蒸馏
    if getattr(args, 'dinov3', False):
        # 推断 visual backbone 的 embed_dim
        _visual = model.visual
        embed_dim = (
            getattr(_visual, 'output_dim', None)
            or getattr(model, 'embed_dim', None)
        )
        if embed_dim is None:
            if hasattr(_visual, 'trunk'):
                embed_dim = _visual.trunk.num_features
            else:
                raise RuntimeError("Cannot determine embed_dim for CLIPWithDINO wrapping.")
        ibot_out = getattr(args, 'ibot_head_prototypes', None) or getattr(args, 'dino_head_prototypes', 65536)
        logging.info(
            f"=> Wrapping model with CLIPWithDINO "
            f"(embed_dim={embed_dim}, dino_proto={args.dino_head_prototypes}, ibot_proto={ibot_out}, "
            f"sigreg_target={sigreg_target})"
        )
        model = CLIPWithDINO(
            clip_model=model,
            embed_dim=embed_dim,
            dino_head_out_dim=getattr(args, 'dino_head_prototypes', 65536),
            ibot_head_out_dim=ibot_out,
            dino_head_nlayers=getattr(args, 'dino_head_nlayers', 3),
            dino_head_hidden=getattr(args, 'dino_head_hidden_dim', 2048),
            dino_head_bottleneck=getattr(args, 'dino_head_bottleneck_dim', 256),
            n_global_crops=getattr(args, 'dino_n_global_crops', 2),
            sigreg_target=sigreg_target,
            sigreg_proj_dim=getattr(args, 'sigreg_proj_dim', 512),
            sigreg_proj_layers=getattr(args, 'sigreg_proj_layers', 3),
            output_dict=True,
        )
        model = model.to(device)

    # ── Dual-teacher mode ────────────────────────────────────────────────────
    tokenizer_secondary = None
    if getattr(args, 'dual_teacher', False):
        from open_clip import create_model_and_transforms as _create
        from safetensors.torch import load_file as _load_safetensors
        import torch.nn as nn

        logging.info("=> Setting up dual-teacher mode")

        # 1. Load PE-Core teacher text encoder
        pe_model, _, _ = _create(
            args.model, args.teacher_pe_ckpt,
            device=device, precision=args.precision, output_dict=True,
        )
        teacher_pe_text = pe_model.text
        teacher_pe_text.requires_grad_(False)
        teacher_pe_text.eval()
        pe_backbone_dim = pe_model.visual.trunk.num_features  # 768
        pe_dim = (
            pe_model.visual.trunk.head.out_features
            if hasattr(pe_model.visual.trunk.head, 'out_features')
            else pe_backbone_dim
        )
        del pe_model.visual  # free image tower memory
        logging.info(f"   PE teacher text loaded: output_dim={pe_dim}, backbone_dim={pe_backbone_dim}")

        # 2. Load SigLIP2 teacher text encoder
        sig_model_name = args.teacher_sig_model or "ViT-B-16-SigLIP2"
        sig_model, _, _ = _create(
            sig_model_name, args.teacher_sig_ckpt,
            device=device, precision=args.precision, output_dict=True,
        )
        teacher_sig_text = sig_model.text
        teacher_sig_text.requires_grad_(False)
        teacher_sig_text.eval()
        sig_dim = getattr(sig_model, 'embed_dim', 768) or 768
        del sig_model.visual
        logging.info(f"   SigLIP2 teacher text loaded: output_dim={sig_dim}")

        # 3. Image backbone from the already-created model (PE-Core architecture)
        image_backbone = model.visual

        # Optionally load pretrained image weights
        if getattr(args, 'pretrained_image_init', None):
            logging.info(f"   Loading pretrained image init from: {args.pretrained_image_init}")
            ckpt = _load_safetensors(args.pretrained_image_init)
            # Extract visual.* keys
            vis_keys = {k.replace('visual.', ''): v for k, v in ckpt.items() if k.startswith('visual.')}
            if not vis_keys:
                vis_keys = ckpt
            image_backbone.load_state_dict(vis_keys, strict=False)

        # 4. Assemble DualTeacherCLIP
        dual_cls = getattr(args, 'dual_cls', False)
        model = DualTeacherCLIP(
            image_backbone=image_backbone,
            teacher_pe_text=teacher_pe_text,
            teacher_sig_text=teacher_sig_text,
            backbone_dim=pe_backbone_dim,
            pe_dim=pe_dim,
            sig_dim=sig_dim,
            dual_cls=dual_cls,
            output_dict=True,
        ).to(device)

        # 5. Secondary tokenizer for SigLIP2
        tokenizer_secondary = get_tokenizer(
            sig_model_name, cache_dir=args.cache_dir,
            context_length=args.force_context_length,
        )
        logging.info(f"   DualTeacherCLIP assembled: dual_cls={dual_cls}, backbone_dim={pe_backbone_dim}, "
                     f"pe_dim={pe_dim}, sig_dim={sig_dim}")

    # DualTextCLIP 的长塔 tokenizer（与短塔同款，context 对齐模型）
    if getattr(args, 'dual_text', False):
        tokenizer_secondary = get_tokenizer(
            args.model, cache_dir=args.cache_dir,
            context_length=args.force_context_length or 256,
        )

    # ── Multi-teacher mode ───────────────────────────────────────────────────
    tokenizer_list = None
    if getattr(args, 'multi_teacher', False):
        logging.info("=> Setting up multi-teacher mode")
        assert args.teachers, "--teachers is required for --multi-teacher"
        teacher_specs = [s.strip().split('::') for s in args.teachers.split(',')]

        teacher_configs = []
        tokenizer_list = []
        image_backbone = model.visual
        backbone_dim = image_backbone.trunk.num_features  # 768

        for spec in teacher_specs:
            t_model_name = spec[0]
            t_ckpt = spec[1] if len(spec) > 1 else None

            if t_model_name.startswith('local-dir:'):
                t_model, _, _ = create_model_and_transforms(
                    t_model_name, pretrained='',
                    device=device, precision=args.precision, output_dict=True,
                )
            else:
                t_model, _, _ = create_model_and_transforms(
                    t_model_name, pretrained=t_ckpt or '',
                    device=device, precision=args.precision, output_dict=True,
                )

            t_model.requires_grad_(False)
            t_model.eval()
            if hasattr(t_model, 'visual'):
                del t_model.visual

            tok_i = get_tokenizer(
                t_model_name, cache_dir=args.cache_dir,
                context_length=args.force_context_length,
            )
            tokenizer_list.append(tok_i)

            with torch.no_grad():
                dummy_tokens = tok_i("a photo").to(device)
                t_embed_dim = t_model.encode_text(dummy_tokens, normalize=True).shape[-1]

            is_siglip = 'siglip' in t_model_name.lower()
            teacher_configs.append({
                'name': t_model_name,
                'model': t_model,
                'embed_dim': t_embed_dim,
                'siglip_style': is_siglip,
            })
            logging.info(f"   Teacher '{t_model_name}': embed_dim={t_embed_dim}, siglip={is_siglip}")

        model = MultiTeacherCLIP(
            image_backbone=image_backbone,
            teacher_configs=teacher_configs,
            backbone_dim=backbone_dim,
            output_dict=True,
        ).to(device)

        args._n_teachers = len(teacher_configs)
        logging.info(f"   MultiTeacherCLIP assembled: {len(teacher_configs)} teachers, backbone_dim={backbone_dim}")

    if args.distill:
        # FIXME: currently assumes the model you're distilling from has the same tokenizer & transforms.
        dist_model, _, _ = create_model_and_transforms(
            args.distill_model, 
            args.distill_pretrained,
            device=device,
            precision=args.precision,
            output_dict=True,
            cache_dir=args.cache_dir,
        )
    if args.use_bnb_linear is not None:
        print('=> using a layer from bitsandbytes.\n'
              '   this is an experimental feature which requires two extra pip installs\n'
              '   pip install bitsandbytes triton'
              '   please make sure to use triton 2.0.0')
        import bitsandbytes as bnb
        from open_clip.utils import replace_linear
        print(f'=> replacing linear layers with {args.use_bnb_linear}')
        linear_replacement_cls = getattr(bnb.nn.triton_based_modules, args.use_bnb_linear)
        replace_linear(model, linear_replacement_cls)
        model = model.to(device)

    random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    # Replace text projection with MLP if requested (reverse-LiT bridge)
    text_proj_override = getattr(args, 'text_proj_type', None)
    if text_proj_override == 'mlp':
        text_module = getattr(model, 'text', model)
        width = text_module.width
        output_dim = text_module.output_dim
        # Remove existing Parameter/Linear before registering new Module
        if hasattr(text_module, 'text_projection'):
            del text_module.text_projection
        mlp = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.LayerNorm(width * 4),
            nn.Linear(width * 4, output_dim),
        ).to(device)
        text_module.text_projection = mlp
        logging.info(f"Replaced text_projection with MLP: {width} -> {width*4} -> {output_dim}")

    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats)
    if args.lock_text:
        model.lock_text_tower(
            unlocked_layers=args.lock_text_unlocked_layers,
            freeze_layer_norm=args.lock_text_freeze_layer_norm)
        # Unfreeze MLP text projection if it was just frozen by lock_text_tower
        if text_proj_override == 'mlp':
            text_module = getattr(model, 'text', model)
            for p in text_module.text_projection.parameters():
                p.requires_grad = True
            logging.info("Unfroze MLP text_projection (trainable bridge on frozen text encoder)")

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    # Freeze logit_scale / logit_bias if requested
    if getattr(args, 'freeze_logit_params', False):
        from open_clip import get_model_config  # noqa: F811
        m_ = model.module if hasattr(model, 'module') else model
        for name in ('logit_scale', 'logit_bias'):
            p = getattr(m_, name, None)
            if p is not None and isinstance(p, torch.nn.Parameter):
                p.requires_grad = False
                logging.info(f"Froze {name} = {p.data.item():.4f}")

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            ddp_args['static_graph'] = True
        if getattr(args, 'dinov3', False):
            ddp_args['static_graph'] = True
            ddp_args.pop('find_unused_parameters', None)
        elif getattr(args, 'pos_only', 'none') != 'none':
            ddp_args['static_graph'] = True
        if getattr(args, 'dual_teacher', False) or getattr(args, 'dual_text', False):
            ddp_args['static_graph'] = True
        if getattr(args, 'multi_teacher', False):
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)
    
        if args.distill:
            dist_model = torch.nn.parallel.DistributedDataParallel(dist_model, device_ids=[device], **ddp_args)

    # create optimizer and scaler
    optimizer = None
    scaler = None

    if args.train_data or args.dataset_type == "synthetic":
        assert not args.trace, 'Cannot train with traced model'

        opt = getattr(args, 'opt', 'adamw').lower()
        if opt.startswith('timm/'):
            from timm.optim import create_optimizer_v2
            timm_opt = opt.split('timm/')[-1]
            opt_kwargs = {}
            assert (args.beta1 is None) == (args.beta2 is None), \
                'When using timm optimizer, BOTH beta1 and beta2 must be specified (or not specified).'
            if args.beta1 is not None:
                opt_kwargs['betas'] = (args.beta1, args.beta2)
            if args.momentum is not None:
                opt_kwargs['momentum'] = args.momentum
            optimizer = create_optimizer_v2(
                model,
                timm_opt,
                lr=args.lr,
                weight_decay=args.wd,
                eps=args.eps,
                **opt_kwargs,
            )
        else:
            # If some params are not passed, we use the default values based on model name.
            exclude = lambda n, p: (
                p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n
                or 'logit_scale' in n or 'logit_bias' in n
            )
            include = lambda n, p: not exclude(n, p)

            named_parameters = list(model.named_parameters())
            gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
            rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

            if opt == 'adamw':
                optimizer = optim.AdamW(
                    [
                        {"params": gain_or_bias_params, "weight_decay": 0.},
                        {"params": rest_params, "weight_decay": args.wd},
                    ],
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                )
            elif opt == 'muon':
                # Muon: hidden weight matrices (ndim>=2, non-embed/non-bias/non-norm) → Muon
                #        everything else (embed, bias, norm, logit_scale/bias)           → AdamW
                from open_clip_train.muon import MuonWithAuxAdam
                muon_lr = getattr(args, 'muon_lr', None) or args.lr
                muon_momentum = getattr(args, 'muon_momentum', 0.95)

                # Re-partition with Muon-specific rules:
                # is_muon: ndim>=2 AND not embedding-like AND not norm/bias/logit
                is_muon = lambda n, p: (
                    p.ndim >= 2
                    and "embed" not in n
                    and "bn" not in n
                    and "ln" not in n
                    and "bias" not in n
                    and "logit_scale" not in n
                    and "logit_bias" not in n
                )
                muon_params = [
                    p for n, p in named_parameters if is_muon(n, p) and p.requires_grad]
                adam_params_wd = [
                    p for n, p in named_parameters
                    if not is_muon(n, p) and not exclude(n, p) and p.requires_grad]
                adam_params_nowd = [
                    p for n, p in named_parameters
                    if not is_muon(n, p) and exclude(n, p) and p.requires_grad]

                param_groups = [
                    dict(params=adam_params_nowd, lr=args.lr, betas=(args.beta1, args.beta2),
                         eps=args.eps, weight_decay=0., use_muon=False),
                    dict(params=adam_params_wd,   lr=args.lr, betas=(args.beta1, args.beta2),
                         eps=args.eps, weight_decay=args.wd, use_muon=False),
                    dict(params=muon_params, lr=muon_lr, momentum=muon_momentum,
                         weight_decay=args.wd, use_muon=True),
                ]
                optimizer = MuonWithAuxAdam(param_groups)
            else:
                assert False, f'Unknown optimizer {opt}'

        if is_master(args):
            defaults = copy.deepcopy(optimizer.defaults)
            defaults['weight_decay'] = args.wd
            defaults = ', '.join([f'{k}: {v}' for k, v in defaults.items()])
            logging.info(
                f'Created {type(optimizer).__name__} ({args.opt}) optimizer: {defaults}'
            )

        if args.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        scaler = None
        if args.precision == "amp":
            try:
                scaler = torch.amp.GradScaler(device=device)
            except (AttributeError, TypeError) as e:
                scaler = torch.cuda.amp.GradScaler()

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

    # initialize datasets
    # If pretrained weights are loaded from a local file, use the same directory for the
    # tokenizer (avoids HuggingFace network requests in offline environments).
    _tokenizer_model = args.model
    if args.pretrained and os.path.isfile(args.pretrained):
        _local_model_dir = os.path.dirname(args.pretrained)
        if os.path.isfile(os.path.join(_local_model_dir, 'tokenizer.json')):
            _tokenizer_model = f'local-dir:{_local_model_dir}'
            logging.info(f"Using local tokenizer from: {_local_model_dir}")
    tokenizer = get_tokenizer(_tokenizer_model, cache_dir=args.cache_dir,
                              context_length=args.force_context_length)
    _model_clip = getattr(model, 'clip_model', model)  # 解包 CLIPLeJEPA / CLIPWithDINO
    _model_ctx = getattr(_model_clip, 'context_length', None)
    if args.force_context_length is None and getattr(tokenizer, 'context_length', None) != _model_ctx:
        # 对齐训练时 PE 系"随机初始化从头训练"的默认 256：tokenizer 与模型必须同窗口。
        # 显式 --force-context-length 时两者天然一致，不重复设置。
        logging.info(f"Aligning tokenizer context_length {getattr(tokenizer, 'context_length', None)} "
                     f"-> model.context_length {_model_ctx}")
        tokenizer = get_tokenizer(_tokenizer_model, cache_dir=args.cache_dir,
                                  context_length=_model_ctx)
    data = get_data(
        args,
        (preprocess_train, preprocess_val),
        epoch=start_epoch,
        tokenizer=tokenizer,
        tokenizer_secondary=tokenizer_secondary,
        tokenizer_list=tokenizer_list,
    )
    assert len(data), 'At least one train or eval dataset must be specified.'

    # create scheduler if train
    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
        if args.lr_scheduler == "cosine":
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const":
            scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const-cooldown":
            assert args.epochs_cooldown is not None, \
                "Please specify the number of cooldown epochs for this lr schedule."
            cooldown_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
            scheduler = const_lr_cooldown(
                optimizer, args.lr, args.warmup, total_steps,
                cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
        else:
            logging.error(
                f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
            exit(1)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume='auto' if args.resume == "latest" else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    # For compatibility, we save state_dict() of the original model, which shares the
    # weights without the prefix.
    original_model = model
    if args.torchcompile:
        logging.info('Compiling model...')

        if args.grad_checkpointing and args.distributed:
            logging.info('Disabling DDP dynamo optimizer when grad checkpointing enabled.')
            # As of now (~PyTorch 2.4/2.5), compile + grad checkpointing work, but DDP optimizer must be disabled
            torch._dynamo.config.optimize_ddp = False

        filter_prefixes = (
            "torch._dynamo",
            "torch._inductor",
            "torch._functorch",
            "torch._utils_internal",
            "torch.fx",
        )

        for name in logging.root.manager.loggerDict:
            if name.startswith(filter_prefixes):
                logging.getLogger(name).setLevel(logging.WARNING)

        model = torch.compile(original_model)

    if 'train' not in data:
        # If using int8, convert to inference mode.
        if args.use_bnb_linear is not None:
            from open_clip.utils import convert_int8_model_to_inference_mode
            convert_int8_model_to_inference_mode(model)
        # Evaluate.
        evaluate(model, data, start_epoch, args, tb_writer=writer, tokenizer=tokenizer)
        return

    loss = create_loss(args)
    loss = loss.to(device)

    attach_modality_modules(model, args)

    # DINOv3 调度：teacher temperature warmup + EMA momentum cosine schedule
    dino_schedules = None
    if getattr(args, 'dinov3', False) and 'train' in data:
        total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
        steps_per_epoch = data["train"].dataloader.num_batches // args.accum_freq
        warmup_temp_epochs = getattr(args, 'dino_warmup_teacher_temp_epochs', 30)
        warmup_temp_steps  = warmup_temp_epochs * steps_per_epoch
        start_temp = getattr(args, 'dino_warmup_teacher_temp', 0.04)
        end_temp   = getattr(args, 'dino_teacher_temp', 0.07)
        start_mom  = getattr(args, 'dino_teacher_momentum', 0.992)

        def _teacher_temp_schedule(step):
            if step < warmup_temp_steps:
                return start_temp + (end_temp - start_temp) * step / max(warmup_temp_steps, 1)
            return end_temp

        def _ema_momentum_schedule(step):
            # cosine from start_mom to 1.0 over total_steps
            return 1.0 - (1.0 - start_mom) * (
                math.cos(math.pi * step / max(total_steps, 1)) * 0.5 + 0.5
            )

        dino_schedules = {
            'teacher_temp': _teacher_temp_schedule,
            'ema_momentum': _ema_momentum_schedule,
        }
        logging.info(
            f"DINOv3 schedules: teacher_temp {start_temp}->{end_temp} over {warmup_temp_epochs} epochs, "
            f"EMA momentum {start_mom}->1.0 over {args.epochs} epochs"
        )

    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        # ---- Curriculum learning: 按特征空间度量重排样本顺序 ----
        if getattr(args, 'curriculum_strategy', None):
            from open_clip_train.curriculum import apply_curriculum, restore_default_order
            cur_epochs = getattr(args, 'curriculum_epochs', 0)
            if cur_epochs == 0 or epoch < cur_epochs:
                apply_curriculum(original_model, data, epoch, args, preprocess_val, device)
            else:
                restore_default_order(data, args, epoch)

        train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args,
                        tb_writer=writer, dino_schedules=dino_schedules,
                        original_model=original_model, preprocess_val=preprocess_val)
        completed_epoch = epoch + 1

        if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
            evaluate(model, data, epoch, args, tb_writer=writer, tokenizer=tokenizer)
            # sync to avoid some processes advancing/exiting while rank 0 finishes eval
            if args.distributed:
                if args.horovod:
                    hvd.join()
                else:
                    torch.distributed.barrier()

        if is_master(args) and not getattr(args, 'probe_freq_steps', None):
            # step-based probing handled inside train_one_epoch; skip epoch-end probe
            from open_clip_train.probe_hook import run_probe
            run_probe(original_model, completed_epoch, args, preprocess_val)

        # Saving checkpoints.
        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": original_model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            if args.delete_previous_checkpoint:
                previous_epoch = completed_epoch - args.save_frequency
                if previous_epoch > 0:
                    previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{previous_epoch}.pt")
                    if os.path.exists(previous_checkpoint):
                        os.remove(previous_checkpoint)

            if args.save_most_recent:
                # try not to corrupt the latest checkpoint if save fails
                tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                torch.save(checkpoint_dict, tmp_save_path)
                os.replace(tmp_save_path, latest_save_path)

        # keep nodes in sync during checkpointing
        if args.distributed:
            if args.horovod:
                hvd.join()
            else:
                torch.distributed.barrier()

    if args.wandb and is_master(args):
        wandb.finish()

    # run a final sync.
    if remote_sync_process is not None:
        logging.info('Final remote sync.')
        remote_sync_process.terminate()
        result = remote_sync(
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol
        )
        if result:
            logging.info('Final remote sync successful.')
        else:
            logging.info('Final remote sync failed.')

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main(sys.argv[1:])
