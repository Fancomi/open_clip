import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP
from open_clip.model import CLIPLeJEPA, CLIPWithDINO
from open_clip_train.distributed import is_master
from open_clip_train.zero_shot import zero_shot_eval
from open_clip_train.precision import get_autocast


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args,
                    tb_writer=None, dino_schedules=None,
                    original_model=None, preprocess_val=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    # Detect DINOv3 mode
    _unwrapped = unwrap_model(model)
    is_dinov3 = isinstance(_unwrapped, CLIPWithDINO)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1 and not is_dinov3:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        # ---- Batch unpacking: DINOv3 vs standard ----
        if is_dinov3:
            # batch = (batch_dict, texts)
            # batch_dict: {global_crops, local_crops, collated_masks, masks_weight, mask_indices}
            batch_dict, texts = batch
            global_crops = batch_dict["global_crops"].to(device=device, dtype=input_dtype, non_blocking=True)
            local_crops  = batch_dict["local_crops"].to(device=device, dtype=input_dtype, non_blocking=True) \
                if batch_dict["local_crops"].numel() > 0 else None
            collated_masks = batch_dict["collated_masks"].to(device=device, non_blocking=True)
            masks_weight   = batch_dict["masks_weight"].to(device=device, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)

            # teacher temperature and EMA momentum from schedules
            teacher_temp = dino_schedules['teacher_temp'](step) if dino_schedules else 0.07
            ema_momentum = dino_schedules['ema_momentum'](step) if dino_schedules else 0.992

            # freeze last layer of DINO/iBOT heads for first N epochs
            # NOTE: we do NOT use requires_grad_(False) here because that changes
            # the DDP computation graph, which is incompatible with static_graph=True.
            # Instead we zero out the gradient *after* backward (see below, after backward()).
            freeze_epochs = getattr(args, 'freeze_last_layer_epochs', 1)
            freeze_last = (epoch < freeze_epochs)
        else:
            images, texts = batch
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            texts  = texts.to(device=device, non_blocking=True)
            teacher_temp = None
            ema_momentum = None

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                if is_dinov3:
                    model_out = model(
                        global_crops=global_crops,
                        local_crops=local_crops,
                        texts=texts,
                        student_masks=collated_masks,
                        masks_weight=masks_weight,
                    )
                    model_out["teacher_temp"] = teacher_temp
                else:
                    model_out = model(images, texts)
                logit_scale = model_out["logit_scale"]
                logit_bias = model_out.get("logit_bias", None)
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(images, texts)
                    model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                losses = loss(**model_out, output_dict=True)

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)

            # freeze last layer: zero grads after backward so the graph stays
            # constant across epochs (static_graph=True requires unchanged graph).
            if is_dinov3 and freeze_last:
                for head in [_unwrapped.student_dino_head, _unwrapped.student_ibot_head]:
                    if head.last_layer.weight.grad is not None:
                        head.last_layer.weight.grad.zero_()

            # === DIAG: print key grad/feature stats for first 5 steps ===
            if is_master(args) and i_accum < 5:
                m = unwrap_model(model)
                lb = getattr(m, 'logit_bias', None)
                ls = getattr(m, 'logit_scale', None)
                img_f = model_out.get("image_features", None)
                txt_f = model_out.get("text_features", None)
                lb_grad = lb.grad.item() if (lb is not None and lb.grad is not None) else None
                ls_grad = ls.grad.item() if (ls is not None and ls.grad is not None) else None
                logging.info(
                    f"  [diag2] logit_bias.grad={lb_grad}  logit_scale.grad={ls_grad}"
                    + (f"  img_norm={img_f.norm(dim=-1).mean().item():.4f}"
                       f"  txt_norm={txt_f.norm(dim=-1).mean().item():.4f}"
                       if img_f is not None else "")
                    + (f"  sim_pos={( (img_f * txt_f).sum(-1).mean().item()):.4f}" if img_f is not None else "")
                )
        else:
            # accum_freq > 1 path: not supported for DINOv3 multi-crop
            # (is_dinov3 always uses accum_freq=1 path above)
            assert not is_dinov3, "accum_freq > 1 is not supported with --dinov3"
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    logit_bias = None
                    if "logit_bias" in model_out:
                        logit_bias = model_out["logit_bias"]
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                # diag3: read true grads after unscale, before step
                if is_master(args) and i_accum < 5:
                    m = unwrap_model(model)
                    lb = getattr(m, 'logit_bias', None)
                    ls = getattr(m, 'logit_scale', None)
                    vp = next(iter(m.visual.parameters()))
                    _vp_before = vp.data.float().clone()
                    _lb_grad = lb.grad.item() if (lb is not None and lb.grad is not None) else None
                    _ls_grad = ls.grad.item() if (ls is not None and ls.grad is not None) else None
                    logging.info(f"  [diag3] lb.grad={_lb_grad}  ls.grad={_ls_grad}"
                                 f"  visual[0].norm={vp.data.norm().item():.4f}")
                scaler.step(optimizer)
                # diag3 cont: measure param update magnitude after step
                if is_master(args) and i_accum < 5:
                    delta = (vp.data.float() - _vp_before).norm().item()
                    logging.info(f"  [diag3] visual[0] param_delta={delta:.6f}")
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            # diag3: param delta (amp_bf16 path, scaler=None)
            if is_master(args) and i_accum < 5:
                m = unwrap_model(model)
                vp = next(iter(m.visual.parameters()))
                _vp_before = vp.data.float().clone()
                lb = getattr(m, 'logit_bias', None)
                ls = getattr(m, 'logit_scale', None)
                _lb_grad = lb.grad.item() if (lb is not None and lb.grad is not None) else None
                _ls_grad = ls.grad.item() if (ls is not None and ls.grad is not None) else None
                logging.info(f"  [diag3] lb.grad={_lb_grad}  ls.grad={_ls_grad}"
                             f"  visual[0].norm={vp.data.norm().item():.4f}")
            optimizer.step()
            if is_master(args) and i_accum < 5:
                delta = (vp.data.float() - _vp_before).norm().item()
                logging.info(f"  [diag3] visual[0] param_delta={delta:.6f}")

        # reset gradient accum, if enabled
        if args.accum_freq > 1 and not is_dinov3:
            accum_images, accum_texts, accum_features = [], [], {}

        # DINOv3: EMA update of teacher after optimizer step
        if is_dinov3 and ema_momentum is not None:
            unwrap_model(model).update_ema(ema_momentum)

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = args.batch_size if is_dinov3 else len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size

            # Grad norm diagnostics (first 3 steps only to avoid overhead)
            if batch_count <= 3:
                grad_norms = {n: p.grad.norm().item() for n, p in model.named_parameters()
                              if p.grad is not None}
                total_gnorm = sum(v**2 for v in grad_norms.values()) ** 0.5
                logging.info(f"  [diag] total_grad_norm={total_gnorm:.4f} "
                             f"n_params_with_grad={len(grad_norms)}")
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            logit_bias_scalar = logit_bias.item() if logit_bias is not None else None
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:.6f} ({loss_m.avg:.6f})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
                + (f"Logit Bias: {logit_bias_scalar:.3f} " if logit_bias_scalar is not None else "")
                + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            if logit_bias_scalar is not None:
                log_data["bias"] = logit_bias_scalar
            log_data.update({name:val.val for name,val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)
            
            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)
            
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()

        # ── Step-granularity feature probe ────────────────────────────────
        probe_freq = getattr(args, 'probe_freq_steps', None)
        if (probe_freq and is_master(args)
                and getattr(args, 'probe_data', None)
                and original_model is not None
                and preprocess_val is not None
                and (step + 1) % probe_freq == 0):
            from open_clip_train.probe_hook import run_probe
            run_probe(original_model, epoch, args, preprocess_val, step=step + 1)
    # end for


def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        # unwrap DDP for single process eval
        if args.distributed and not args.horovod:
            model = model.module
        # For CLIPWithDINO, eval uses the inner CLIP model (standard image/text interface)
        if isinstance(model, CLIPWithDINO):
            model = model.clip_model
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        logging.info(f"Eval: Starting validation on {samples_per_val} samples...")
        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()

                    # Use the same loss function as training
                    if args.siglip:
                        logits_per_image = logit_scale * image_features @ text_features.t()
                        logits_per_text = logits_per_image.t()
                        # 5cap 协议：同一张图的 n_caps 条 caption 互为正样本，
                        # label matrix 为块对角；1cap 时退化为 eye
                        n_caps_val = getattr(args, 'val_num_captions_per_image', 1) or 1
                        if n_caps_val > 1 and batch_size % n_caps_val == 0:
                            n_img_batch = batch_size // n_caps_val
                            block = torch.ones(n_caps_val, n_caps_val, device=device)
                            labels_binary = torch.block_diag(*([block] * n_img_batch))
                        else:
                            labels_binary = torch.eye(batch_size, device=device)
                        loss_i2t = F.binary_cross_entropy_with_logits(
                            logits_per_image, labels_binary, reduction='mean')
                        loss_t2i = F.binary_cross_entropy_with_logits(
                            logits_per_text, labels_binary, reduction='mean')
                        total_loss = (loss_i2t + loss_t2i) / 2
                    else:
                        # CLIP contrastive loss
                        total_loss = (
                            F.cross_entropy(logits_per_image, labels) +
                            F.cross_entropy(logits_per_text, labels)
                        ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % args.log_every_n_steps) == 0:
                    loss_name = "SigLIP Loss" if args.siglip else "Clip Loss"
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"{loss_name}: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            logging.info(f"Eval: Computing retrieval metrics on {len(all_image_features)} batches...")
            n_caps = getattr(args, 'val_num_captions_per_image', 1) or 1
            all_img = torch.cat(all_image_features)   # [n_img * n_caps, d]  or  [n_img, d]
            all_txt = torch.cat(all_text_features)    # [n_img * n_caps, d]
            if n_caps > 1:
                # 每 n_caps 行对应同一张图，取第 0 行代表该图的特征
                all_img = all_img[::n_caps]           # [n_img, d]
            val_metrics = get_clip_metrics(
                image_features=all_img,
                text_features=all_txt,
                logit_scale=logit_scale.cpu(),
                num_captions_per_image=n_caps,
            )
            logging.info(f"Eval:   Step 3/3 - Metrics computed.")
            loss = cumulative_loss / num_samples
            loss_key = "siglip_val_loss" if args.siglip else "clip_val_loss"
            metrics.update(
                {**val_metrics, loss_key: loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale,
                     num_captions_per_image: int = 1):
    """计算检索指标，支持 1:1 和 1:N（论文标准 COCO 5cap）两种协议。

    num_captions_per_image=1  : 标准 1:1，ground_truth[i] = i
    num_captions_per_image=N  : 1:N，TSV 行排列为 img0_cap0..capN-1, img1_cap0...
                                 - I2T: 每张图对应第 i*N..(i+1)*N-1 条文本
                                 - T2I: 每条文本对应第 i//N 张图
    """
    metrics = {}
    n_caps = num_captions_per_image
    n_img  = len(image_features)          # 图数 = total_pairs / n_caps
    n_txt  = len(text_features)           # 文本数 = n_img * n_caps

    logging.info(f"Eval:   Step 1/3 - Computing similarity matrix "
                 f"[{n_img} imgs, {n_txt} texts, {n_caps} caps/img]...")
    # image_features: [n_img, d]  text_features: [n_txt, d]
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    # [n_img, n_txt]
    logits_per_text  = logits_per_image.t().detach().cpu()
    # [n_txt, n_img]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ---------- I2T ----------
    logging.info("Eval:     Computing ranks 1/2 - image_to_text...")
    logit_gpu = logits_per_image.to(device)   # [n_img, n_txt]
    if n_caps == 1:
        gt_scores = logit_gpu.diagonal().unsqueeze(1)          # [n_img, 1]
        ranks_i2t = (logit_gpu > gt_scores).sum(dim=1).cpu().numpy()
    else:
        # 每张图的正确 caption 是列 [i*n_caps .. (i+1)*n_caps - 1]
        # rank = 有多少条非正确文本的分数 > 该图所有正确文本中最高分
        ranks_i2t = np.zeros(n_img, dtype=np.int64)
        for i in range(n_img):
            pos_cols = slice(i * n_caps, (i + 1) * n_caps)
            best_pos_score = logit_gpu[i, pos_cols].max()
            # 非正确文本中有多少分数更高
            neg_mask = torch.ones(n_txt, dtype=torch.bool, device=device)
            neg_mask[pos_cols] = False
            ranks_i2t[i] = (logit_gpu[i][neg_mask] > best_pos_score).sum().item()
    del logit_gpu

    metrics["image_to_text_mean_rank"]   = ranks_i2t.mean() + 1
    metrics["image_to_text_median_rank"] = np.floor(np.median(ranks_i2t)) + 1
    for k in [1, 5, 10]:
        metrics[f"image_to_text_R@{k}"] = np.mean(ranks_i2t < k)

    # ---------- T2I ----------
    logging.info("Eval:     Computing ranks 2/2 - text_to_image...")
    logit_gpu = logits_per_text.to(device)    # [n_txt, n_img]
    if n_caps == 1:
        gt_scores = logit_gpu.diagonal().unsqueeze(1)          # [n_txt, 1]
        ranks_t2i = (logit_gpu > gt_scores).sum(dim=1).cpu().numpy()
    else:
        # 每条 caption 的正确图是第 i//n_caps 张
        gt_img_idx = torch.arange(n_txt, device=device) // n_caps  # [n_txt]
        gt_scores  = logit_gpu[torch.arange(n_txt, device=device), gt_img_idx].unsqueeze(1)
        ranks_t2i  = (logit_gpu > gt_scores).sum(dim=1).cpu().numpy()
    del logit_gpu

    metrics["text_to_image_mean_rank"]   = ranks_t2i.mean() + 1
    metrics["text_to_image_median_rank"] = np.floor(np.median(ranks_t2i)) + 1
    for k in [1, 5, 10]:
        metrics[f"text_to_image_R@{k}"] = np.mean(ranks_t2i < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
