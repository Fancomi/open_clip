from typing import Optional

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        if logit_bias is not None:
            logits_per_image += logit_bias
            logits_per_text += logit_bias

        return logits_per_image, logits_per_text

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            logit_bias=None,
            output_dict=False,
    ):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(
            image_features,
            text_features,
            logit_scale,
            logit_bias=logit_bias,
        )

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss
        else:
            clip_loss = torch.tensor(0, device=logits.device)

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self,
            cache_labels: bool = False,
            rank: int = 0,
            world_size: int = 1,
            dist_impl: Optional[str] = None,
            antipodal: bool = False,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.antipodal = antipodal
        self.dist_impl = dist_impl or 'bidir'  # default to bidir exchange for now, this will likely change
        assert self.dist_impl in ('bidir', 'shift', 'reduce', 'gather')

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if self.antipodal:
            logits = -logits
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias, output_dict=False):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        if self.world_size > 1:
            if self.dist_impl == 'bidir':
                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )
                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_right
                    )
                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            elif self.dist_impl == "shift":
                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_right,
                    )
                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left
            elif self.dist_impl == "reduce":
                for i in range(self.world_size):
                    text_from_other = torch.distributed.nn.all_reduce(
                        text_features * (self.rank == i),
                        torch.distributed.ReduceOp.SUM,
                    )
                    loss += float(i != self.rank) * self._loss(
                        image_features,
                        text_from_other,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            elif self.dist_impl == "gather":
                all_text = torch.distributed.nn.all_gather(text_features)
                for i in range(self.world_size):
                    loss += float(i != self.rank) * self._loss(
                        image_features,
                        all_text[i],
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                assert False

        return {"siglip_loss": loss} if output_dict else loss


class DualSigLipLoss(nn.Module):
    """Sum of two SigLipLoss for dual-teacher training."""

    def __init__(self, cache_labels=False, rank=0, world_size=1, dist_impl=None):
        super().__init__()
        self.loss_pe = SigLipLoss(cache_labels, rank, world_size, dist_impl)
        self.loss_sig = SigLipLoss(cache_labels, rank, world_size, dist_impl)

    def forward(self, image_features_pe, text_features_pe, logit_scale_pe,
                image_features_sig, text_features_sig, logit_scale_sig, logit_bias_sig,
                output_dict=False, **kwargs):
        lp = self.loss_pe(image_features_pe, text_features_pe, logit_scale_pe, None, output_dict=True)
        ls = self.loss_sig(image_features_sig, text_features_sig, logit_scale_sig, logit_bias_sig, output_dict=True)
        loss = lp["siglip_loss"] + ls["siglip_loss"]
        if output_dict:
            return {"dual_loss": loss, "loss_pe": lp["siglip_loss"], "loss_sig": ls["siglip_loss"]}
        return loss


class MultiTeacherLoss(nn.Module):
    """Sum of N SigLipLoss instances for multi-teacher training."""

    def __init__(self, n_teachers, weights=None, cache_labels=False, rank=0, world_size=1, dist_impl=None):
        super().__init__()
        self.n_teachers = n_teachers
        self.weights = weights or [1.0] * n_teachers
        self.losses = nn.ModuleList([
            SigLipLoss(cache_labels, rank, world_size, dist_impl)
            for _ in range(n_teachers)
        ])

    def forward(self, n_teachers=0, output_dict=False, **kwargs):
        n = n_teachers or self.n_teachers
        total = 0.0
        result = {}
        for i in range(n):
            img = kwargs[f'image_features_{i}']
            txt = kwargs[f'text_features_{i}']
            scale = kwargs[f'logit_scale_{i}']
            bias = kwargs.get(f'logit_bias_{i}', None)
            li = self.losses[i](img, txt, scale, bias, output_dict=True)['siglip_loss']
            total = total + self.weights[i] * li
            result[f'loss_{i}'] = li
        result['multi_teacher_loss'] = total
        if output_dict:
            return result
        return total


def _dist_all_reduce_avg(x):
    """跨 GPU 平均归约，未初始化时直接返回。"""
    if has_distributed and dist.is_available() and dist.is_initialized():
        torch.distributed.nn.functional.all_reduce(x, torch.distributed.ReduceOp.AVG)
    return x


def _dist_world_size():
    if has_distributed and dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization (LeJEPA, https://arxiv.org/abs/2511.08544)

    随机切片 + Epps-Pulley 特征函数检验，约束 embeddings 服从各向同性高斯分布。
    输入特征应为 unnormalized（不在超球面上），否则统计量为常数。
    """

    def __init__(self, knots: int = 17, num_slices: int = 256):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        phi = (-t.square() / 2.0).exp()  # N(0,1) 特征函数: exp(-t²/2)

        self.num_slices = num_slices
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)  # 梯形权重 × φ(t)
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))
        # generator 缓存（避免每次重建）
        self._gen: Optional[torch.Generator] = None
        self._gen_device = None

    def _get_generator(self, device, seed: int) -> torch.Generator:
        if self._gen is None or self._gen_device != device:
            self._gen = torch.Generator(device=device)
            self._gen_device = device
        self._gen.manual_seed(seed)
        return self._gen

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, D]，unnormalized 特征
        Returns:
            Epps-Pulley 统计量（标量），越小越接近各向同性高斯
        """
        N = x.size(0)
        world_size = _dist_world_size()

        with torch.no_grad():
            # 同步 global_step seed（保证各 rank 投影方向一致）
            step = self.global_step.clone().to(x.device)
            _dist_all_reduce_avg(step)  # AVG on identical values = same value
            seed = int(step.item()) * 2  # ×2 避免与其他地方 seed 冲突
            g = self._get_generator(x.device, seed)

            A = torch.randn(x.size(-1), self.num_slices, device=x.device, dtype=x.dtype, generator=g)
            A /= A.norm(p=2, dim=0, keepdim=True)
            self.global_step.add_(1)

        t = self.t.to(x)          # 同步 device + dtype
        phi = self.phi.to(x)
        weights = self.weights.to(x)

        # 投影: [N, num_slices]，再扩展积分维度: [N, num_slices, knots]
        x_t = (x @ A).unsqueeze(-1) * t

        cos_mean = x_t.cos().mean(0)   # [num_slices, knots]
        sin_mean = x_t.sin().mean(0)

        # 跨 GPU 平均（等价于全局 batch 均值）
        _dist_all_reduce_avg(cos_mean)
        _dist_all_reduce_avg(sin_mean)

        # |φ̂(t) - φ(t)|² = (cos_mean - φ)² + sin_mean²
        err = (cos_mean - phi).square() + sin_mean.square()

        # 梯形数值积分，乘以全局样本数
        return (err @ weights).mean() * N * world_size



# ============================================================
# DINOv3 Self-distillation Losses
# Ported from Meta AI DINOv3, removing dinov3.distributed deps.
# ============================================================

def _dino_all_reduce(x: torch.Tensor) -> torch.Tensor:
    """In-place all-reduce sum across GPUs. No-op if not distributed."""
    if has_distributed and dist.is_available() and dist.is_initialized():
        dist.all_reduce(x)
    return x


def _dino_world_size() -> int:
    if has_distributed and dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


class SinkhornKnopp(nn.Module):
    """Sinkhorn-Knopp centering for teacher outputs (used by DINO and iBOT).

    Converts raw teacher logits to soft assignment probabilities via iterative
    row/column normalization, ensuring balanced cluster assignments.
    """

    @torch.no_grad()
    def forward(
        self,
        teacher_output: torch.Tensor,
        teacher_temp: float,
        n_samples: Optional[torch.Tensor] = None,
        n_iterations: int = 3,
    ) -> torch.Tensor:
        """
        Args:
            teacher_output: [B, K] teacher logits (before temperature).
            teacher_temp:   Teacher sharpening temperature.
            n_samples:      Total number of samples across all GPUs (scalar tensor).
                            If None, uses local batch size * world_size.
            n_iterations:   Number of Sinkhorn iterations.

        Returns:
            Q: [B, K] soft assignment matrix (rows sum to ~1).
        """
        teacher_output = teacher_output.float()
        Q = torch.exp(teacher_output / teacher_temp).t()  # [K, B]
        world_size = _dino_world_size()
        B = Q.shape[1] * world_size if n_samples is None else n_samples.float()
        K = Q.shape[0]

        sum_Q = Q.sum()
        _dino_all_reduce(sum_Q)
        Q /= sum_Q

        for _ in range(n_iterations):
            sum_rows = Q.sum(dim=1, keepdim=True)
            _dino_all_reduce(sum_rows)
            Q /= sum_rows
            Q /= K
            Q /= Q.sum(dim=0, keepdim=True)
            Q /= B

        Q *= B
        return Q.t()  # [B, K]


class DINOClsTokenLoss(nn.Module):
    """CLS-token self-distillation loss (DINO-style).

    Student learns to match Sinkhorn-Knopp-centered teacher outputs.

    Args:
        out_dim:          Number of prototypes (head output dim).
        student_temp:     Student temperature (default: 0.1).
        center_momentum:  EMA momentum for center (default: 0.9).
    """

    def __init__(
        self,
        out_dim: int,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        n_global_crops: int = 2,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.n_global_crops = n_global_crops
        self.sinkhorn = SinkhornKnopp()
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(
        self,
        student_cls_tokens: torch.Tensor,
        teacher_cls_tokens: torch.Tensor,
        teacher_temp: float,
        ignore_diagonal: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            student_cls_tokens: [n_student_crops * B, out_dim] student head logits.
            teacher_cls_tokens: [n_teacher_crops * B, out_dim] teacher head logits (raw).
            teacher_temp:       Current teacher temperature (scheduled externally).
            ignore_diagonal:    Exclude same-view pairs (student_i vs teacher_i).

        Returns:
            Scalar loss.
        """
        B_s = student_cls_tokens.shape[0]
        B_t = teacher_cls_tokens.shape[0]

        # B_t must evenly divide B_s (n_student_crops >= n_teacher_crops)
        assert B_s % B_t == 0, f"B_s={B_s} must be divisible by B_t={B_t}"
        local_B = B_t  # per-crop batch size = total teacher tokens
        n_s = B_s // local_B  # number of student crops per image... wait
        # Actually: B_t = n_t * BS, B_s = n_s * BS, so local_B = BS
        # We need to find BS.  B_t = n_t * BS. Simplest: GCD approach.
        # In practice n_t=2 (global crops), so BS = B_t // 2.
        # But we don't want to hardcode n_t.  Use: iterate over teacher crops.
        # Split teacher into n_t crops of size BS each:
        #   teacher_probs[i*BS:(i+1)*BS] corresponds to teacher crop i
        # Then for each teacher crop t, average cross-entropy over all student crops.

        # Sinkhorn-Knopp centering on teacher
        teacher_probs = self.sinkhorn(teacher_cls_tokens - self.center, teacher_temp)  # [B_t, K]

        # Student log-softmax
        student_logsoft = F.log_softmax(student_cls_tokens.float() / self.student_temp, dim=-1)  # [B_s, K]

        # Cross-entropy: iterate over teacher crops to avoid [B_s, B_t, K] expansion.
        # B_t = n_t * BS,  B_s = n_s * BS.  We don't know BS, but GCD(B_s,B_t)=BS*gcd(n_s,n_t).
        # Safest: infer BS as B_t (treat each teacher token as one "crop"), iterate over each.
        # This is equivalent to the reference DINOv2 implementation which loops over teacher views.
        #
        # Reference pattern (DINOv2 facebookresearch):
        #   total_loss = 0
        #   for t in teacher_crops:
        #       for s in student_crops (skip same-view):
        #           total_loss += -sum(t * log(s))
        #
        # Here B_t rows = n_t * BS, so we iterate chunk-wise.
        # We need BS to know chunk boundaries.
        n_t = self.n_global_crops
        BS = B_t // n_t
        n_s_crops = B_s // BS

        total_loss = 0.0
        n_pairs = 0
        for t_idx in range(n_t):
            t_probs = teacher_probs[t_idx * BS: (t_idx + 1) * BS]  # [BS, K]
            for s_idx in range(n_s_crops):
                if ignore_diagonal and s_idx == t_idx:
                    continue  # skip same-view pair
                s_logsoft = student_logsoft[s_idx * BS: (s_idx + 1) * BS]  # [BS, K]
                total_loss += -(t_probs * s_logsoft).sum(dim=-1).mean()
                n_pairs += 1

        return total_loss / max(n_pairs, 1)

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor) -> None:
        """EMA update of the centering buffer from teacher CLS tokens."""
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        _dino_all_reduce(batch_center)
        batch_center /= _dino_world_size()
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class iBOTPatchLoss(nn.Module):
    """Masked patch token self-distillation loss (iBOT-style).

    Computes cross-entropy between student masked patches and teacher patches,
    weighted by per-sample inverse mask count.

    Args:
        out_dim:          Head output dim (number of prototypes).
        student_temp:     Student temperature (default: 0.1).
        center_momentum:  EMA momentum for center (default: 0.9).
    """

    def __init__(
        self,
        out_dim: int,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.sinkhorn = SinkhornKnopp()
        self.register_buffer("center", torch.zeros(1, 1, out_dim))

    def forward(
        self,
        student_patch_tokens: torch.Tensor,
        teacher_patch_tokens: torch.Tensor,
        student_masks: torch.Tensor,
        teacher_temp: float,
        masks_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            student_patch_tokens:  [B, N, out_dim] student head logits on all patches.
            teacher_patch_tokens:  [B, N, out_dim] teacher head logits (raw) on all patches.
            student_masks:         [B, N] bool mask (True = masked in student, predict these).
            teacher_temp:          Current teacher temperature.
            masks_weight:          [n_masked] optional per-token weight; if None, computed from masks.

        Returns:
            Scalar loss.
        """
        B, N, K = student_patch_tokens.shape

        # Teacher centering (subtract center, then Sinkhorn)
        # Flatten [B, N, K] -> [B*N, K] for SK, then reshape back
        teacher_flat = teacher_patch_tokens.reshape(B * N, K)
        teacher_centered = teacher_flat - self.center.reshape(1, K)
        teacher_probs_flat = self.sinkhorn(teacher_centered, teacher_temp)  # [B*N, K]
        teacher_probs = teacher_probs_flat.reshape(B, N, K)

        # Student log-softmax
        student_logsoft = F.log_softmax(
            student_patch_tokens.float() / self.student_temp, dim=-1
        )  # [B, N, K]

        # Per-token cross-entropy, then mask and normalize per sample
        per_token_ce = -(teacher_probs * student_logsoft).sum(dim=-1)  # [B, N]

        if masks_weight is not None:
            # Use pre-computed flat weights for masked tokens
            masked_ce = per_token_ce[student_masks]  # [n_masked]
            loss = (masked_ce * masks_weight).sum() / B
        else:
            n_masked_per_sample = student_masks.float().sum(dim=-1).clamp(min=1.0)  # [B]
            loss = (per_token_ce * student_masks.float()).sum(dim=-1)  # [B]
            loss = (loss / n_masked_per_sample).mean()

        return loss

    def forward_masked(
        self,
        student_patch_tokens_masked: torch.Tensor,
        teacher_patch_tokens_masked: torch.Tensor,
        student_masks: torch.Tensor,
        teacher_temp: float,
        n_masked_patches: Optional[int] = None,
        masks_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Efficient variant operating on pre-gathered masked tokens.

        Args:
            student_patch_tokens_masked:  [n_masked, K] student tokens at mask positions.
            teacher_patch_tokens_masked:  [n_masked, K] teacher tokens at mask positions.
            student_masks:                [B, N] bool mask (used for weight computation if needed).
            teacher_temp:                 Current teacher temperature.
            n_masked_patches:             If set, truncate to first n_masked_patches tokens.
            masks_weight:                 [n_masked] per-token weight.

        Returns:
            Scalar loss.
        """
        B = student_masks.shape[0]

        teacher_centered = teacher_patch_tokens_masked - self.center.reshape(1, teacher_patch_tokens_masked.shape[-1])
        n_samples = torch.tensor(
            teacher_patch_tokens_masked.shape[0] * _dino_world_size(),
            dtype=torch.long, device=teacher_patch_tokens_masked.device
        )
        teacher_probs = self.sinkhorn(teacher_centered, teacher_temp, n_samples=n_samples)  # [n_masked, K]
        student_logsoft = F.log_softmax(
            student_patch_tokens_masked.float() / self.student_temp, dim=-1
        )  # [n_masked, K]

        loss = -(teacher_probs * student_logsoft).sum(dim=-1)  # [n_masked]

        if n_masked_patches is not None:
            loss = loss[:n_masked_patches]

        if masks_weight is None:
            masks_weight = (
                (1.0 / student_masks.float().sum(dim=-1).clamp(min=1.0))
                .unsqueeze(-1)
                .expand_as(student_masks)[student_masks]
            )
        loss = (loss * masks_weight).sum() / B
        return loss

    @torch.no_grad()
    def update_center(self, teacher_patch_tokens: torch.Tensor) -> None:
        """EMA update of the patch centering buffer.

        Args:
            teacher_patch_tokens: [B, N, K] or [B*N, K] teacher patch head outputs.
        """
        if teacher_patch_tokens.dim() == 3:
            batch_center = teacher_patch_tokens.mean(dim=1).mean(dim=0, keepdim=True)  # [1, K]
        else:
            batch_center = teacher_patch_tokens.mean(dim=0, keepdim=True)
        _dino_all_reduce(batch_center)
        batch_center /= _dino_world_size()
        self.center = self.center * self.center_momentum + batch_center.unsqueeze(0) * (1 - self.center_momentum)


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko nearest-neighbor entropic regularizer.

    Encourages uniform spreading of embeddings by penalizing
    small distances to the nearest neighbor.

    Reference: Sablayrolles et al. 2018 "Spreading vectors for similarity search"
    """

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def _pairwise_nn_inner(self, x: torch.Tensor) -> torch.Tensor:
        """Find nearest neighbor indices via max inner product (for L2-normalized vectors)."""
        dots = torch.mm(x, x.t())  # [N, N]
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # fill diagonal with -1
        _, indices = torch.max(dots, dim=1)
        return indices

    def forward(self, student_output: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Args:
            student_output: [B, D] backbone (pre-head) CLS token features.

        Returns:
            Scalar KoLeo loss.
        """
        with torch.autocast("cuda", enabled=False):
            x = F.normalize(student_output.float(), p=2, dim=-1, eps=eps)
            indices = self._pairwise_nn_inner(x)
            distances = self.pdist(x, x[indices])  # [B]
            loss = -torch.log(distances + eps).mean()
        return loss


class ModalityGapLoss(nn.Module):
    """Batch-level modality-gap regularizer (pre-L2-norm features).

    L_gap = || mean(img_raw) - mean(txt_raw) ||²

    Gradient flows through the current-batch means back to the features.
    No EMA — pure batch statistics. Applied pre-L2-norm.
    """

    def forward(
        self,
        image_features: torch.Tensor,  # [B, D] pre-L2-norm, gradient required
        text_features: torch.Tensor,   # [B, D] pre-L2-norm, gradient required
    ) -> torch.Tensor:
        batch_img = image_features.mean(dim=0)
        batch_txt = text_features.mean(dim=0)
        if has_distributed and dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_img, op=dist.ReduceOp.AVG)
            dist.all_reduce(batch_txt, op=dist.ReduceOp.AVG)
        return (batch_img - batch_txt).pow(2).sum()


class UniformityLoss(nn.Module):
    """Wang & Isola (2020) uniformity loss on the hypersphere.

    L_uniform = log(mean(exp(-t * ||z_i - z_j||^2)))
             = log(mean(exp(-2t * (1 - cos_ij))))   for L2-normalized z

    Lower values = more uniform distribution. Minimum at perfect uniformity.
    """

    def __init__(self, t: float = 2.0):
        super().__init__()
        self.t = t

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [N, D], assumed L2-normalized
        # ||z_i - z_j||^2 = 2 - 2*cos_ij for unit vectors
        cos_sim = features @ features.T  # [N, N]
        sq_dists = 2.0 - 2.0 * cos_sim
        # Exclude diagonal (self-pairs)
        n = features.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=features.device)
        # logsumexp for numerical stability, then subtract log(count) to get log(mean)
        return torch.logsumexp(-self.t * sq_dists[mask], dim=0) - math.log(n * (n - 1))


class SIGRegContrastiveLoss(nn.Module):
    """SIGReg 正则化 + CLIP/SigLIP 主损失 (+ optional modality-gap regularizer)

    L = L_clip
      + λ_sigreg  × (SIGReg(image_proj) + SIGReg(text_proj))
      + modality_gap_loss   (pre-computed by model, pre-L2-norm)

    image_proj / text_proj 由 CLIPLeJEPA 模型提供（unnormalized）：
    - sigreg_target in {clip, cls}:      Identity，直接作用于 [B, D] embedding
    - sigreg_target in {clip_proj, cls_proj}: MLP projector 输出，间接约束 backbone
    """

    def __init__(
            self,
            sigreg_weight: float = 1e-4,
            sigreg_knots: int = 17,
            sigreg_num_slices: int = 256,
            use_siglip: bool = False,
            local_loss: bool = False,
            gather_with_grad: bool = False,
            cache_labels: bool = False,
            rank: int = 0,
            world_size: int = 1,
            use_horovod: bool = False,
            dist_impl=None,
            within_modal_weight: float = 0.0,
            within_modal_sides: str = 'both',   # 'both' | 'img' | 'txt'
            within_modal_mode: str = 'replace',  # 'replace' | 'auxiliary'
            uniformity_weight: float = 0.0,
            uniformity_t: float = 2.0,
            koleo_weight: float = 0.0,
            antipodal: bool = False,
    ):
        super().__init__()
        self.sigreg_weight = sigreg_weight
        self.within_modal_weight = within_modal_weight
        assert within_modal_sides in ('both', 'img', 'txt'), \
            f"within_modal_sides must be 'both'/'img'/'txt', got {within_modal_sides!r}"
        self.within_modal_sides = within_modal_sides
        assert within_modal_mode in ('replace', 'auxiliary'), \
            f"within_modal_mode must be 'replace'/'auxiliary', got {within_modal_mode!r}"
        self.within_modal_mode = within_modal_mode
        self.rank = rank
        self.world_size = world_size
        self.gather_with_grad = gather_with_grad
        self.antipodal = antipodal

        if use_siglip:
            assert not use_horovod, "Horovod not supported for SigLip"
            self.main_loss = SigLipLoss(rank=rank, world_size=world_size, dist_impl=dist_impl, antipodal=antipodal)
        else:
            self.main_loss = ClipLoss(
                local_loss=local_loss, gather_with_grad=gather_with_grad,
                cache_labels=cache_labels, rank=rank, world_size=world_size, use_horovod=use_horovod,
            )
        self.sigreg = SIGReg(knots=sigreg_knots, num_slices=sigreg_num_slices)

        # Representation uniformity losses
        self.uniformity_weight = uniformity_weight
        self.uniformity_loss = UniformityLoss(t=uniformity_t) if uniformity_weight > 0 else None
        self.koleo_weight = koleo_weight
        self.koleo_loss = KoLeoLoss() if koleo_weight > 0 else None

    def _cross_modal_positive_only(
        self,
        image_features: torch.Tensor,   # [B, D] local rank, L2-normalised
        text_features: torch.Tensor,    # [B, D] local rank, L2-normalised
        logit_scale: torch.Tensor,
        logit_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Cross-modal loss using ONLY positive pairs (diagonal).

        No cross-modal negatives → removes the incentive for modality separation.
        Alignment is maintained purely through positive pairs.

        No distributed exchange needed: positive pairs are always co-located
        on the same rank.

        L = -mean_i log σ(scale · img_i⊤txt_i  + bias)
        Normalisation: sum / B  (same convention as SigLipLoss._loss).
        """
        B = image_features.shape[0]
        # Diagonal similarities: element-wise dot product of matched pairs
        pos_logits = logit_scale * (image_features * text_features).sum(dim=-1)  # [B]
        if self.antipodal:
            pos_logits = -pos_logits
        if logit_bias is not None:
            pos_logits = pos_logits + logit_bias
        return -F.logsigmoid(pos_logits).sum() / B

    def _within_modal_siglip(
        self,
        features: torch.Tensor,       # [B, D] local rank features, L2-normalised
        logit_scale: torch.Tensor,
        logit_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Within-modality SigLIP repulsion loss.

        Reuses the exact same formula, logit_scale, logit_bias, and sum/B
        normalisation as cross-modal SigLipLoss, so the two losses are
        naturally on the same scale.

        Labels: all off-diagonal pairs = -1 (negatives).
        Diagonal is masked to 0 (self-similarity carries no signal).

        With logit_bias ≈ -10.4 the loss is near-zero when within-modal
        cosines are small (features spread out), and grows automatically as
        features cluster into a modality cloud — self-calibrating.

        With world_size > 1 features are gathered globally so every rank
        sees the full same-modality negatives (consistent with SigLip gather).

        Gradient scaling reference (λ=0.5):
          0.5 * (L_wm_img + L_wm_txt) contributes the same total number of
          negative pairs as the cross-modal SigLIP: 0.5*2*B*(B-1) = B*(B-1).
        """
        if self.world_size > 1:
            if self.gather_with_grad:
                all_feats = torch.cat(torch.distributed.nn.all_gather(features), dim=0)
            else:
                gathered = [torch.zeros_like(features) for _ in range(self.world_size)]
                dist.all_gather(gathered, features)
                gathered[self.rank] = features   # keep grad on local slice
                all_feats = torch.cat(gathered, dim=0)
        else:
            all_feats = features

        N = all_feats.shape[0]
        logits = logit_scale * (all_feats @ all_feats.T)
        if logit_bias is not None:
            logits = logits + logit_bias

        # All off-diagonal pairs are negatives: -logsigmoid(-logit) = softplus(logit)
        loss = F.softplus(logits)

        # Mask diagonal (self-similarity = 1.0, no learning signal)
        eye = torch.eye(N, device=logits.device, dtype=torch.bool)
        loss = loss.masked_fill(eye, 0.0)

        # Normalise by B (same convention as SigLipLoss._loss: sum / B)
        return loss.sum() / N

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            logit_bias=None,
            image_proj=None,
            text_proj=None,
            modality_gap_loss: Optional[torch.Tensor] = None,
            output_dict: bool = False,
    ):
        # SIGReg 作用在 unnormalized proj 上（由 CLIPLeJEPA 提供）
        reg = sum(self.sigreg(f) for f in (image_proj, text_proj) if f is not None)
        weighted_reg = self.sigreg_weight * reg

        if self.within_modal_weight > 0:
            # ── within-modal 模式 ─────────────────────────────────────────────
            sides = self.within_modal_sides

            if self.within_modal_mode == 'auxiliary':
                # auxiliary 模式：保留完整 SigLIP（含 cross-neg），叠加 within-modal
                wm_img = self._within_modal_siglip(image_features, logit_scale, logit_bias) \
                    if sides in ('both', 'img') else None
                wm_txt = self._within_modal_siglip(text_features, logit_scale, logit_bias) \
                    if sides in ('both', 'txt') else None
                if sides == 'both':
                    wm_loss = self.within_modal_weight * (wm_img + wm_txt) * 0.5
                elif sides == 'img':
                    wm_loss = self.within_modal_weight * wm_img
                else:
                    wm_loss = self.within_modal_weight * wm_txt
                main_loss = self.main_loss(
                    image_features, text_features, logit_scale, logit_bias, output_dict=False
                )
                losses = {
                    "contrastive_loss":  main_loss,
                    "sigreg_loss":       weighted_reg,
                    "within_modal_loss": wm_loss,
                }
            else:
                # replace 模式（原始行为）：去掉 cross-neg，仅保留 cross-pos
                wm_img = self._within_modal_siglip(image_features, logit_scale, logit_bias) \
                    if sides in ('both', 'img') else None
                wm_txt = self._within_modal_siglip(text_features, logit_scale, logit_bias) \
                    if sides in ('both', 'txt') else None
                if sides == 'both':
                    wm_loss = self.within_modal_weight * (wm_img + wm_txt) * 0.5
                elif sides == 'img':
                    wm_loss = self.within_modal_weight * wm_img
                else:
                    wm_loss = self.within_modal_weight * wm_txt
                cross_loss = self._cross_modal_positive_only(
                    image_features, text_features, logit_scale, logit_bias
                )
                losses = {
                    "contrastive_loss":  cross_loss,
                    "sigreg_loss":       weighted_reg,
                    "within_modal_loss": wm_loss,
                }
        else:
            # ── 标准模式（baseline 行为，完全不变）────────────────────────────
            main_loss = self.main_loss(
                image_features, text_features, logit_scale, logit_bias, output_dict=False
            )
            losses = {"contrastive_loss": main_loss, "sigreg_loss": weighted_reg}

        # modality_gap_loss pre-computed by model (pre-L2-norm), just accumulate
        if modality_gap_loss is not None:
            losses["modality_gap_loss"] = modality_gap_loss

        # Representation uniformity losses (on L2-normalized features in CLIP space)
        if self.uniformity_weight > 0 or self.koleo_weight > 0:
            # Gather features across GPUs (same pattern as _within_modal_siglip)
            if self.world_size > 1:
                if self.gather_with_grad:
                    all_img = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
                    all_txt = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
                else:
                    gathered_img = [torch.zeros_like(image_features) for _ in range(self.world_size)]
                    gathered_txt = [torch.zeros_like(text_features) for _ in range(self.world_size)]
                    dist.all_gather(gathered_img, image_features)
                    dist.all_gather(gathered_txt, text_features)
                    gathered_img[self.rank] = image_features
                    gathered_txt[self.rank] = text_features
                    all_img = torch.cat(gathered_img, dim=0)
                    all_txt = torch.cat(gathered_txt, dim=0)
            else:
                all_img = image_features
                all_txt = text_features

            if self.uniformity_weight > 0:
                uni_img = self.uniformity_loss(all_img)
                uni_txt = self.uniformity_loss(all_txt)
                losses["uniformity_loss"] = self.uniformity_weight * 0.5 * (uni_img + uni_txt)

            if self.koleo_weight > 0:
                koleo_img = self.koleo_loss(all_img)
                koleo_txt = self.koleo_loss(all_txt)
                losses["koleo_loss"] = self.koleo_weight * 0.5 * (koleo_img + koleo_txt)

        if output_dict:
            return losses
        return sum(losses.values())


# backwards-compat alias
ClipLeJEPALoss = SIGRegContrastiveLoss


class CLIPWithDINOLoss(nn.Module):
    """Combined contrastive + self-distillation (+ optional SIGReg) loss for CLIPWithDINO.

    L = w_contrast * L_siglip
      + w_dino    * L_dino      (CLS token self-distillation)
      + w_ibot    * L_ibot      (masked patch token self-distillation)
      + w_koleo   * L_koleo     (nearest-neighbor entropy regularizer)
      + w_sigreg  * L_sigreg    (optional: Sketched Isotropic Gaussian Regularizer)

    SIGReg inputs (sigreg_feat / sigreg_feat_text) are unnormalized features.
    When dinov3+sigreg cls/cls_proj: sigreg_feat = student_cls_tokens_raw [B, backbone_dim].
    When dinov3+sigreg clip/clip_proj: sigreg_feat = unnormalized CLIP image embedding [B, clip_dim].
    """

    def __init__(
        self,
        dino_out_dim: int,
        ibot_out_dim: int,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        dino_loss_weight: float = 1.0,
        ibot_loss_weight: float = 1.0,
        koleo_loss_weight: float = 0.1,
        sigreg_weight: float = 0.0,
        sigreg_num_slices: int = 256,
        modality_gap_weight: float = 0.0,
        modality_gap_ema: float = 0.999,
        use_siglip: bool = True,
        rank: int = 0,
        world_size: int = 1,
        dist_impl: Optional[str] = None,
        n_global_crops: int = 2,
        antipodal: bool = False,
    ):
        super().__init__()
        self.dino_loss_weight = dino_loss_weight
        self.ibot_loss_weight = ibot_loss_weight
        self.koleo_loss_weight = koleo_loss_weight
        self.sigreg_weight = sigreg_weight
        self.modality_gap_weight = modality_gap_weight

        if use_siglip:
            self.contrastive_loss = SigLipLoss(
                rank=rank, world_size=world_size, dist_impl=dist_impl, antipodal=antipodal
            )
        else:
            self.contrastive_loss = ClipLoss(
                rank=rank, world_size=world_size, cache_labels=True
            )

        self.dino_loss = DINOClsTokenLoss(
            out_dim=dino_out_dim,
            student_temp=student_temp,
            center_momentum=center_momentum,
            n_global_crops=n_global_crops,
        )
        self.ibot_loss = iBOTPatchLoss(
            out_dim=ibot_out_dim,
            student_temp=student_temp,
            center_momentum=center_momentum,
        )
        self.koleo_loss = KoLeoLoss()

        if sigreg_weight > 0:
            self.sigreg = SIGReg(num_slices=sigreg_num_slices)
        else:
            self.sigreg = None

        if modality_gap_weight > 0:
            self.gap_loss = ModalityGapLoss(ema_momentum=modality_gap_ema)
        else:
            self.gap_loss = None

    def forward(
        self,
        # contrastive inputs
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
        logit_bias: Optional[torch.Tensor] = None,
        # DINO inputs
        student_cls_tokens: Optional[torch.Tensor] = None,
        teacher_cls_tokens: Optional[torch.Tensor] = None,
        student_cls_tokens_raw: Optional[torch.Tensor] = None,
        # iBOT inputs
        student_patch_tokens: Optional[torch.Tensor] = None,
        teacher_patch_tokens: Optional[torch.Tensor] = None,
        student_masks: Optional[torch.Tensor] = None,
        masks_weight: Optional[torch.Tensor] = None,
        # SIGReg inputs (unnormalized features for regularization)
        sigreg_feat: Optional[torch.Tensor] = None,
        sigreg_feat_text: Optional[torch.Tensor] = None,
        # modality gap loss pre-computed by model (pre-L2-norm)
        modality_gap_loss: Optional[torch.Tensor] = None,
        # teacher temperature (scheduled outside)
        teacher_temp: float = 0.07,
        output_dict: bool = False,
    ) -> dict:
        losses = {}

        # 1. Contrastive loss (SigLIP / CLIP)
        contrast = self.contrastive_loss(
            image_features, text_features, logit_scale, logit_bias, output_dict=True
        )
        losses.update(contrast)

        # 2. DINO CLS token loss
        if student_cls_tokens is not None and teacher_cls_tokens is not None:
            dino = self.dino_loss(student_cls_tokens, teacher_cls_tokens, teacher_temp)
            self.dino_loss.update_center(teacher_cls_tokens)
            losses["dino_loss"] = self.dino_loss_weight * dino

        # 3. iBOT patch token loss
        if (
            student_patch_tokens is not None
            and teacher_patch_tokens is not None
            and student_masks is not None
        ):
            ibot = self.ibot_loss.forward_masked(
                student_patch_tokens,
                teacher_patch_tokens,
                student_masks,
                teacher_temp=teacher_temp,
                masks_weight=masks_weight,
            )
            self.ibot_loss.update_center(teacher_patch_tokens)
            losses["ibot_loss"] = self.ibot_loss_weight * ibot

        # 4. KoLeo loss on student CLS token (pre-head backbone features)
        if student_cls_tokens_raw is not None and self.koleo_loss_weight > 0:
            koleo = self.koleo_loss(student_cls_tokens_raw)
            losses["koleo_loss"] = self.koleo_loss_weight * koleo

        # 5. SIGReg loss (optional; acts on unnormalized features provided by CLIPWithDINO)
        if self.sigreg is not None and self.sigreg_weight > 0:
            reg = sum(self.sigreg(f) for f in (sigreg_feat, sigreg_feat_text) if f is not None)
            losses["sigreg_loss"] = self.sigreg_weight * reg

        # 6. Modality gap loss pre-computed by model (pre-L2-norm), just accumulate
        if modality_gap_loss is not None:
            losses["modality_gap_loss"] = modality_gap_loss

        if output_dict:
            return losses
        return sum(losses.values())
