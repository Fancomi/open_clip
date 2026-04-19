"""Multi-crop augmentation and block masking for DINOv3 self-distillation.

Ported from Meta AI DINOv3, adapted to:
  - Remove dinov3-library dependencies
  - Use torchvision.transforms.v2
  - Accept model-level mean/std from preprocess config

Provides:
  DataAugmentationDINO   -- per-image transform returning a dict of crops
  MaskingGenerator       -- blockwise random masking for iBOT
  collate_dino_batch     -- collate function for DataLoader
"""

import logging
import math
import random

import numpy as np
import torch
import torchvision.transforms.v2 as v2
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Default normalisation constants (ImageNet)
# ------------------------------------------------------------------ #
_IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_DEFAULT_STD  = (0.229, 0.224, 0.225)


class GaussianBlur:
    """Apply Gaussian blur with a given probability."""

    def __init__(self, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if random.random() < self.p:
            radius = random.uniform(self.radius_min, self.radius_max)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


# ------------------------------------------------------------------ #
# DataAugmentationDINO
# ------------------------------------------------------------------ #

class DataAugmentationDINO:
    """Multi-crop augmentation for DINOv3 self-distillation.

    For each image produces:
      - 2 global crops (large, diverse augmentations)
      - N local crops  (small, for local-context losses)

    Returns a dict with keys:
      "global_crops"   : list of 2 tensors [C, H_g, W_g]
      "local_crops"    : list of N tensors [C, H_l, W_l]

    Args:
        global_crops_scale:  Scale range for global RandomResizedCrop.
        local_crops_scale:   Scale range for local RandomResizedCrop.
        local_crops_number:  Number of local crops.
        global_crops_size:   Pixel size of global crops (default 224).
        local_crops_size:    Pixel size of local crops (default 96).
        mean, std:           Normalization constants (from model preprocess config).
    """

    def __init__(
        self,
        global_crops_scale=(0.32, 1.0),
        local_crops_scale=(0.05, 0.32),
        local_crops_number: int = 8,
        global_crops_size: int = 224,
        local_crops_size: int = 96,
        mean=_IMAGENET_DEFAULT_MEAN,
        std=_IMAGENET_DEFAULT_STD,
    ):
        self.local_crops_number = local_crops_number

        # Normalization
        normalize = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=list(mean), std=list(std)),
        ])

        # Geometric: global
        geo_global = v2.Compose([
            v2.RandomResizedCrop(
                global_crops_size,
                scale=global_crops_scale,
                interpolation=v2.InterpolationMode.BICUBIC,
            ),
            v2.RandomHorizontalFlip(p=0.5),
        ])

        # Geometric: local
        geo_local = v2.Compose([
            v2.RandomResizedCrop(
                local_crops_size,
                scale=local_crops_scale,
                interpolation=v2.InterpolationMode.BICUBIC,
            ),
            v2.RandomHorizontalFlip(p=0.5),
        ])

        # Color augmentations
        color_jitter = v2.Compose([
            v2.RandomApply([
                v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            ], p=0.8),
            v2.RandomGrayscale(p=0.2),
        ])

        # Global crop 1: strong blur
        self.global_transfo1 = v2.Compose([
            geo_global,
            color_jitter,
            # GaussianBlur is PIL-based; applied in __call__
        ])
        self._blur1 = GaussianBlur(p=1.0)
        self._normalize = normalize

        # Global crop 2: mild blur + solarize
        self.global_transfo2 = v2.Compose([
            geo_global,
            color_jitter,
        ])
        self._blur2 = GaussianBlur(p=0.1)
        self._solarize = v2.RandomSolarize(threshold=128, p=0.2)

        # Local crops
        self.local_transfo = v2.Compose([
            geo_local,
            color_jitter,
        ])
        self._blur_local = GaussianBlur(p=0.5)

        logger.info(
            f"DataAugmentationDINO: global_crops_scale={global_crops_scale}, "
            f"local_crops_scale={local_crops_scale}, n_local={local_crops_number}, "
            f"global_size={global_crops_size}, local_size={local_crops_size}"
        )

    def _apply_global1(self, img):
        img = self.global_transfo1(img)
        img = self._blur1(img)
        return self._normalize(img)

    def _apply_global2(self, img):
        img = self.global_transfo2(img)
        img = self._blur2(img)
        img = self._solarize(img)
        return self._normalize(img)

    def _apply_local(self, img):
        img = self.local_transfo(img)
        img = self._blur_local(img)
        return self._normalize(img)

    def __call__(self, image):
        """
        Args:
            image: PIL Image.

        Returns:
            dict with "global_crops" (list of 2 tensors) and
                       "local_crops"  (list of N tensors).
        """
        global_crop_1 = self._apply_global1(image)
        global_crop_2 = self._apply_global2(image)
        local_crops = [self._apply_local(image) for _ in range(self.local_crops_number)]
        return {
            "global_crops": [global_crop_1, global_crop_2],
            "local_crops": local_crops,
        }


# ------------------------------------------------------------------ #
# MaskingGenerator
# ------------------------------------------------------------------ #

class MaskingGenerator:
    """Blockwise random mask generator for iBOT.

    Generates a 2D boolean mask over patch grids by randomly placing
    rectangular blocks, then filling remaining targets randomly.

    Args:
        input_size:  (H, W) in patches, e.g. (14, 14) for 224px / patch_size=16.
        min_num_patches:  Minimum block size.
        max_num_patches:  Maximum number of patches per block.
        min_aspect:  Minimum aspect ratio of blocks.
        max_aspect:  Maximum aspect ratio of blocks.
    """

    def __init__(
        self,
        input_size,
        num_masking_patches=None,
        min_num_patches: int = 4,
        max_num_patches=None,
        min_aspect: float = 0.3,
        max_aspect=None,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size)
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches
        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches
        max_aspect = max_aspect or 1.0 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def _place_block(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect)))
            w = int(round(math.sqrt(target_area / aspect)))
            if w < self.width and h < self.height:
                top  = random.randint(0, self.height - h)
                left = random.randint(0, self.width  - w)
                num_already = mask[top:top+h, left:left+w].sum()
                if 0 < h * w - num_already <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if not mask[i, j]:
                                mask[i, j] = True
                                delta += 1
                    if delta > 0:
                        break
        return delta

    def __call__(self, num_masking_patches: int = 0) -> np.ndarray:
        """Generate a boolean mask.

        Args:
            num_masking_patches:  Target number of masked patches.

        Returns:
            Boolean array of shape (height * width,).
        """
        mask = np.zeros((self.height, self.width), dtype=bool)
        count = 0
        while count < num_masking_patches:
            remaining = num_masking_patches - count
            max_block = min(remaining, self.max_num_patches)
            delta = self._place_block(mask, max_block)
            if delta == 0:
                break
            count += delta

        # Fill remaining randomly
        flat = mask.flatten()
        remaining = num_masking_patches - flat.sum()
        if remaining > 0:
            candidates = np.where(~flat)[0]
            chosen = np.random.choice(candidates, size=int(remaining), replace=False)
            flat[chosen] = True
        return flat.reshape(self.height, self.width)


# ------------------------------------------------------------------ #
# collate_dino_batch
# ------------------------------------------------------------------ #

def collate_dino_batch(
    samples_list,
    mask_ratio_tuple=(0.1, 0.5),
    mask_probability: float = 0.5,
    n_tokens: int = 196,
    mask_generator: MaskingGenerator = None,
    dtype=torch.float32,
):
    """Collate a list of (dino_dict, text_tokens) samples into a batch.

    Args:
        samples_list:        List of ((dino_output_dict, text_token), ...) tuples.
                             dino_output_dict has keys "global_crops", "local_crops".
        mask_ratio_tuple:    (min_ratio, max_ratio) for iBOT masking.
        mask_probability:    Fraction of samples that get masked.
        n_tokens:            Total number of patch tokens per image.
        mask_generator:      MaskingGenerator instance.
        dtype:               Target dtype for image tensors.

    Returns:
        (batch_dict, texts) where:
          batch_dict has keys:
            "global_crops"    : [n_global * B, C, H, W]
            "local_crops"     : [n_local * B, C, H, W]
            "collated_masks"  : [B, N] bool
            "masks_weight"    : [n_masked] float
            "mask_indices"    : [n_masked] long (flat indices into [B*N])
          texts : [B, L]
    """
    # samples_list: list of ((dict, text), )  -- from webdataset batch
    # after wds.batched() it comes as a list of tuples; we handle both shapes
    if isinstance(samples_list[0], (list, tuple)) and len(samples_list[0]) == 2:
        dino_dicts = [s[0] for s in samples_list]
        texts      = torch.stack([s[1] for s in samples_list])
    else:
        # fallback: assume samples_list IS the dino_dicts (texts handled elsewhere)
        dino_dicts = samples_list
        texts = None

    n_global = len(dino_dicts[0]["global_crops"])
    n_local  = len(dino_dicts[0]["local_crops"])
    B = len(dino_dicts)

    # Stack crops: [n_crops * B, C, H, W]  with samples interleaved per crop index
    global_crops = torch.stack(
        [dino_dicts[b]["global_crops"][i] for i in range(n_global) for b in range(B)]
    ).to(dtype)
    local_crops = torch.stack(
        [dino_dicts[b]["local_crops"][i] for i in range(n_local) for b in range(B)]
    ).to(dtype) if n_local > 0 else torch.zeros(0, dtype=dtype)

    # iBOT masking (applied to global crops, shape indexed over B samples)
    n_samples_masked = int(B * mask_probability)
    masks_list = []

    if mask_generator is not None and n_samples_masked > 0:
        probs = torch.linspace(mask_ratio_tuple[0], mask_ratio_tuple[1], n_samples_masked + 1)
        for i in range(n_samples_masked):
            prob = probs[i + 1].item()
            masks_list.append(torch.from_numpy(mask_generator(int(n_tokens * prob))))
        for _ in range(n_samples_masked, B):
            masks_list.append(torch.from_numpy(mask_generator(0)))
        random.shuffle(masks_list)
    else:
        for _ in range(B):
            masks_list.append(torch.zeros(
                int(math.sqrt(n_tokens)), int(math.sqrt(n_tokens)), dtype=torch.bool
            ))

    collated_masks = torch.stack(masks_list).flatten(1)   # [B, N]
    mask_indices   = collated_masks.flatten().nonzero(as_tuple=False).squeeze(-1)  # [n_masked]
    n_masked_per   = collated_masks.float().sum(dim=-1).clamp(min=1.0)  # [B]
    masks_weight   = (1.0 / n_masked_per).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    batch = {
        "global_crops":   global_crops,
        "local_crops":    local_crops,
        "collated_masks": collated_masks,
        "masks_weight":   masks_weight,
        "mask_indices":   mask_indices,
    }
    return batch, texts
