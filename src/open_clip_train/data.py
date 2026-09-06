import ast
import json
import logging
import math
import os
import random
import sys
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


class RandomResizedCropWithBoxes:
    """RandomResizedCrop + 同步变换归一化框（区域监督专用）。

    动机：区域坐标是相对**原图**归一化的，默认的 RandomResizedCrop 会让框失效，
    所以区域组此前只能用 resize-only。而实测随机裁剪值 COCO i2t 1.70 / IN-1k 0.70
    （gt_s1 有裁剪 23.16/22.95 vs A' 无裁剪 21.46/22.25，同 commit，均超 2σ）——
    这部分正则化收益不该白丢。本类把裁剪参数取出来，让框跟着变换。

    删减策略有两档，由 `keep_area_thr` 选：

    - `keep_area_thr <= 0`（默认，历史行为）：**完全包含** —— 框必须整体落在裁剪区内，
      否则丢弃。当初的理由是我们的框绑定短语级语义（"orange equilateral triangle"），
      切掉一半后短语与区域不再对应；检测任务里 clip 一个 person 框还是 person，
      容忍度不同。
    - `keep_area_thr > 0`：**clip + 保留面积比阈值** —— 把框裁到 [0,1] 内，
      只要「裁剪后面积 / 裁剪前面积 ≥ 阈值」就保留（保留的是裁过的框）。
      动机：完全包含在 scale=(0.9,1.0) 下丢掉四分之一以上的框，等于把区域监督的
      信号量削掉一大块 —— 而 H 组那次 crop-aug 判有害（k-NN −0.55 / COCO t2i −0.40）时，
      我们错把原因归给「保框约束削弱了增强」，实际 H 用的 scale 与 gt_base 完全相同，
      真正的差别就是这笔框损失。改 clip + 面积阈值后丢框率大幅下降，
      而语义损失被阈值兜住（thr=0.8 ⇒ 最多切掉 20% 面积）。

    实测丢框率（k24 表 3000 图 / 20138 框，K=12，scale=(0.9,1.0)，ratio=(3/4,4/3)，
    见 /tmp/verify_keep_area.py）：

    | thr | 0（完全包含） | 0.9 | 0.8 | 0.7 | 0.6 | 0.5 | 0.3 |
    |---|---|---|---|---|---|---|---|
    | 丢框率 | 26.2% | 12.1% | **6.9%** | 3.0% | 1.8% | 1.2% | 0.7% |
    | 零框图 | 6.0% | 3.1% | **1.7%** | 0.9% | 0.8% | 0.7% | 0.6% |

    ⚠️ 上表测在 `clip_train_region_k24.tsv` 的**前 3000 行**上，而那一段每图只有
    8.29 个候选框、全表是 18.71（>K 的行 23.9% vs 全表 77.5%）—— **不具代表性**。
    在 `clip_train_region.tsv` 的 1/100 代表性切片（2 万图，见 /tmp/measure_misalign.py）
    上重测：thr=0.8 丢框 1.37%、thr=0.6 丢框 0.23%、thr=0（完全包含）丢框 **39.92%**。
    引用丢框率一律用后一组。

    ⚠️⚠️ 删框会把短语配对打乱（`--region-crop-fix-align` 之前的历史行为）：
    `kept = b[keep]` 把保留的框**压到前面**，而调用方取短语时用的是 `phrases[:n_valid]`
    ——「取前 n 个」。只要被丢的框不在末尾，从它往后每个 slot 的短语都错位一格。
    同一切片实测「保留框中错配比例」：thr=0.8 **2.41%**、thr=0.6 0.27%、
    thr=0 **53.95%**（有错配的图分别 6.52% / 1.03% / 65.34%）。
    框按面积降序 ⇒ 大框在前、也最容易被裁出画面 ⇒ 错配偏向发生在**开头**、影响最大。
    """

    def __init__(self, size, scale, ratio, interpolation, tail, keep_area_thr=0.0,
                 fix_align=False):
        self.size = size if isinstance(size, (tuple, list)) else (size, size)
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.tail = tail            # ToTensor + Normalize 等后续变换
        self.keep_area_thr = float(keep_area_thr)
        self.fix_align = bool(fix_align)   # 见 --region-crop-fix-align

    def __call__(self, img, boxes, n_valid):
        """img: PIL；boxes: [K,4] 归一化；n_valid: int。

        返回 (tensor, boxes, n_valid, kept_idx)。`kept_idx` 是保留下来的框在**输入
        顺序**里的下标列表，供调用方挑对应短语；`fix_align=False` 时返回 None，
        调用方退回历史行为（取前 n_valid 个短语 —— 有丢框时会错配，见该开关的说明）。
        """
        from torchvision.transforms import RandomResizedCrop
        from torchvision.transforms.functional import resized_crop
        W, H = img.size
        i, j, h, w = RandomResizedCrop.get_params(img, list(self.scale), list(self.ratio))
        out = self.tail(resized_crop(img, i, j, h, w, list(self.size),
                                     interpolation=self.interpolation, antialias=True))
        if n_valid == 0 or w <= 0 or h <= 0:
            return out, boxes, n_valid, (list(range(n_valid)) if self.fix_align else None)
        # 归一化(原图) → 像素 → 减裁剪原点 → 除裁剪尺寸 = 归一化(裁剪后)
        b = boxes[:n_valid].clone()
        b[:, 0] = (b[:, 0] * W - j) / w
        b[:, 2] = (b[:, 2] * W - j) / w
        b[:, 1] = (b[:, 1] * H - i) / h
        b[:, 3] = (b[:, 3] * H - i) / h
        if self.keep_area_thr <= 0:
            # 完全包含：四边都在 [0,1] 内，且仍有面积
            keep = ((b[:, 0] >= 0) & (b[:, 1] >= 0) & (b[:, 2] <= 1) & (b[:, 3] <= 1)
                    & (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1]))
        else:
            # clip 到画面内，按「保留面积 / 原面积」过阈值决定留不留（留裁过的框）
            area0 = ((b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0))
            b = b.clamp(0.0, 1.0)
            area1 = ((b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0))
            keep = (area0 > 0) & (area1 > 0) & ((area1 / area0.clamp(min=1e-8)) >= self.keep_area_thr)
        kept = b[keep]
        newb = torch.zeros_like(boxes)
        nk = int(kept.shape[0])
        if nk:
            newb[:nk] = kept
        kept_idx = [x for x, v in enumerate(keep.tolist()) if v] if self.fix_align else None
        return out, newb, nk, kept_idx


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None,
                 caption2_key=None, tokenizer2=None, region_key=None, max_region=12,
                 region_select="order", region_seed=0, shared_epoch=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        # caption_key 缺失时回退到 'caption'（val 单列数据集兼容 dual-text 配置）
        if caption_key not in df.columns:
            caption_key = 'caption'
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer
        # 可选第二文本列（DualTextCLIP 用）：caption2_key + tokenizer2
        self.caption2_key = caption2_key if caption2_key and caption2_key in df.columns else None
        self.captions2 = df[caption2_key].tolist() if self.caption2_key else None
        self.tokenize2 = tokenizer2

        # 可选区域列（FG-CLIP 式区域-短语对比）：JSON [[phrase,x1,y1,x2,y2], ...]，坐标已归一化
        self.region_key = region_key if region_key and region_key in df.columns else None
        self.regions = df[region_key].tolist() if self.region_key else None
        self.max_region = max_region
        self.region_select = region_select
        self.region_seed = region_seed
        self.shared_epoch = shared_epoch

    def _select_regions(self, items, idx, K):
        """候选框多于 K 时挑 K 个（见 --region-select）。

        order        取前 K 个（建表按面积降序 ⇒ 与历史 run 逐位一致）
        random       每图固定的随机 K 个（种子 = seed × 行号，跨 epoch 不变）
        random-epoch 每 epoch 重挑（种子额外含 epoch，由 SharedEpoch 传进 worker）
        dedup        仍按面积降序，但短语重复的框排到最后 ⇒ 优先填满不同短语

        四种取法都返回**保持原有面积降序**的子集，且框数一律 = min(K, 候选数)；
        候选数 ≤ K 时四者完全一致。
        """
        if self.region_select == "order" or len(items) <= K:
            return items[:K]
        if self.region_select == "dedup":
            seen, first, dup = set(), [], []
            for i, it in enumerate(items):
                p = str(it[0])
                (dup if p in seen else first).append(i)
                seen.add(p)
            return [items[i] for i in sorted((first + dup)[:K])]
        ep = 0
        if self.region_select == "random-epoch" and self.shared_epoch is not None:
            ep = self.shared_epoch.get_value()
        rng = random.Random((self.region_seed * 1000003 + idx) * 1009 + ep)
        return [items[i] for i in sorted(rng.sample(range(len(items)), K))]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        box_aware = isinstance(self.transforms, RandomResizedCropWithBoxes)
        try:
            pil = Image.open(str(self.images[idx]))
            if box_aware:
                pil = pil.convert("RGB")        # box-aware 分支自己管 convert
                images = None                    # 稍后与框一起变换
            else:
                images = self.transforms(pil)
        except (OSError, IOError):
            # 损坏图片: 返回随机邻居替代
            return self.__getitem__((idx + 1) % len(self))
        texts = self.tokenize([str(self.captions[idx])])[0]

        if self.regions is not None:
            # 区域模式：返回 (image, text, region_texts[K], boxes[K,4], n_valid)
            # 若同时有 caption2（PCM），返回 6 元组 (image, text, text2, rtexts, boxes, n_valid)
            # K 固定为 max_region，不足处补零并由 n_valid 标记有效数（collate 需定长）
            K = self.max_region
            try:
                items = self._select_regions(json.loads(self.regions[idx]), idx, K)
            except (TypeError, ValueError):
                items = []
            phrases = [str(it[0]) for it in items]
            boxes = torch.zeros(K, 4, dtype=torch.float32)
            for j, it in enumerate(items):
                boxes[j] = torch.tensor([float(it[1]), float(it[2]), float(it[3]), float(it[4])])
            n_valid = len(items)
            # 短语 tokenize；空位用空串占位（前向时按 n_valid 掩掉，不参与损失）
            pad = K - n_valid
            region_texts = self.tokenize(phrases + [""] * pad) if K else torch.zeros(0, dtype=torch.long)
            if box_aware:
                # 图像与框同步变换；n_valid 可能因框被裁出画面而减少
                images, boxes, n_valid, kept_idx = self.transforms(pil, boxes, n_valid)
                if n_valid < K:                  # 被删的框对应短语置空（mask 会掩掉）
                    pad2 = K - n_valid
                    # kept_idx 非 None（--region-crop-fix-align）时按保留下标挑短语；
                    # None 时退回历史行为「取前 n_valid 个」—— 丢框不在末尾就会错配
                    sel = phrases[:n_valid] if kept_idx is None else [phrases[i] for i in kept_idx]
                    region_texts = self.tokenize(sel + [""] * pad2)
            nv = torch.tensor(n_valid, dtype=torch.long)
            if self.captions2 is not None:
                texts2 = self.tokenize2([str(self.captions2[idx])])[0]
                return images, texts, texts2, region_texts, boxes, nv
            return images, texts, region_texts, boxes, nv

        if images is None:      # box_aware 但无区域列：退化为纯裁剪
            images, _, _, _ = self.transforms(pil, torch.zeros(1, 4), 0)
        if self.captions2 is not None:
            texts2 = self.tokenize2([str(self.captions2[idx])])[0]
            return images, texts, texts2
        return images, texts


class VideoFrameDataset(Dataset):
    """从 b64 帧缓存中随机采帧，配对视觉描述文本。

    支持两种初始化模式:
      1. split="train"/"eval": 从 {root}/split.json 加载预划分子集
      2. split=None: 全量扫描目录（兼容旧用法）
    """

    VIEWS = ("front", "side")
    CAPTION_KEY = "category_1_visual_description"

    def __init__(self, root, transforms, tokenizer, max_side=768, split=None):
        import io, base64 as _b64
        self._io, self._b64 = io, _b64
        self.transforms = transforms
        self.tokenize = tokenizer

        root = os.path.normpath(root)
        split_file = os.path.join(root, "split.json")
        if split and os.path.isfile(split_file):
            with open(split_file) as f:
                split_data = json.load(f)
            records = split_data[split]
            self.samples = [(r["b64_path"], r["caption"]) for r in records]
            logging.info(f'VideoFrameDataset [{split}]: {len(self.samples)} samples')
        else:
            self.samples = []
            for dirpath, _, filenames in os.walk(root):
                for view in self.VIEWS:
                    aug_name = f"augment_{view}_cn.json"
                    if aug_name not in filenames:
                        continue
                    b64_path = os.path.join(dirpath, f"frames_{max_side}p", f"{view}.b64")
                    if not os.path.isfile(b64_path):
                        continue
                    with open(os.path.join(dirpath, aug_name)) as f:
                        caption = json.load(f).get(self.CAPTION_KEY, "")
                    if caption:
                        self.samples.append((b64_path, caption))
            logging.info(f'VideoFrameDataset: {len(self.samples)} samples from {root}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        b64_path, caption = self.samples[idx]
        # 读取所有帧并随机选 1 帧
        lines = open(b64_path, "r").read().splitlines()
        frame_b64 = random.choice(lines)
        img = Image.open(self._io.BytesIO(self._b64.b64decode(frame_b64))).convert("RGB")
        return self.transforms(img), self.tokenize([caption])[0]


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist), \
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights), \
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None, **kwargs):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    # DINOv3 multi-crop mode: use DataAugmentationDINO as transform and custom collate
    use_dinov3 = is_train and getattr(args, 'dinov3', False)
    if use_dinov3:
        from open_clip_train.dino_transform import DataAugmentationDINO, MaskingGenerator, collate_dino_batch
        from open_clip.model import get_model_preprocess_cfg
        # Retrieve mean/std from preprocess config (attached to model.visual by factory)
        # preprocess_img is the standard transform; extract mean/std from it
        _mean = getattr(preprocess_img, 'image_mean', None)
        _std  = getattr(preprocess_img, 'image_std',  None)
        if _mean is None:
            # fallback: extract from Normalize in transforms list
            for t in getattr(preprocess_img, 'transforms', []):
                if hasattr(t, 'mean') and hasattr(t, 'std'):
                    _mean = tuple(t.mean)
                    _std  = tuple(t.std)
                    break
        if _mean is None:
            _mean = (0.485, 0.456, 0.406)
            _std  = (0.229, 0.224, 0.225)

        # Determine global crop size from preprocess_img
        global_size = 224
        for t in getattr(preprocess_img, 'transforms', []):
            if hasattr(t, 'size'):
                sz = t.size
                global_size = sz[0] if isinstance(sz, (list, tuple)) else sz
                break

        dino_transform = DataAugmentationDINO(
            global_crops_scale=tuple(getattr(args, 'dino_global_crops_scale', [0.32, 1.0])),
            local_crops_scale=tuple(getattr(args, 'dino_local_crops_scale', [0.05, 0.32])),
            local_crops_number=getattr(args, 'dino_local_crops_number', 8),
            n_global_crops=getattr(args, 'dino_n_global_crops', 2),
            global_crops_size=global_size,
            local_crops_size=getattr(args, 'dino_local_crops_size', 96),
            mean=_mean,
            std=_std,
        )
        n_tokens = (global_size // 16) ** 2  # assumes patch_size=16
        mask_gen = MaskingGenerator(
            input_size=global_size // 16,
            num_masking_patches=int(n_tokens * getattr(args, 'ibot_mask_ratio_max', 0.5)),
        )
        _mask_ratio = (
            getattr(args, 'ibot_mask_ratio_min', 0.1),
            getattr(args, 'ibot_mask_ratio_max', 0.5),
        )
        _mask_prob = getattr(args, 'ibot_mask_sample_prob', 0.5)

        def _collate_dino(samples):
            return collate_dino_batch(
                samples,
                mask_ratio_tuple=_mask_ratio,
                mask_probability=_mask_prob,
                n_tokens=n_tokens,
                mask_generator=mask_gen,
            )

        preprocess_img = dino_transform

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0 

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled, (
            "--train_data_upsampling_factors is only supported when sampling"
            " with replacement (with --dataset-resampled)."
        )
    
    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            epoch=shared_epoch,
        )]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    tokenizer_secondary = kwargs.get('tokenizer_secondary', None)
    tokenizer_list = kwargs.get('tokenizer_list', None)
    if tokenizer_list is not None:
        def _multi_tokenize(sample):
            raw_text = sample['text']
            for i, tok in enumerate(tokenizer_list):
                sample[f'text_{i}'] = tok(raw_text)[0]
            return sample
        text_fields = tuple(f'text_{i}' for i in range(len(tokenizer_list)))
        pipeline.extend([
            wds.select(filter_no_caption_or_no_image),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(image="jpg;png;jpeg;webp", text="txt"),
            wds.map_dict(image=preprocess_img),
            wds.map(_multi_tokenize),
            wds.to_tuple("image", *text_fields),
            wds.batched(args.batch_size, partial=not is_train,
                        collation_fn=_collate_dino if use_dinov3 else wds.filters.default_collation_fn)
        ])
    elif tokenizer_secondary is not None:
        def _dual_tokenize(sample):
            sample['text2'] = tokenizer_secondary(sample['text'])[0]
            sample['text'] = tokenizer(sample['text'])[0]
            return sample
        pipeline.extend([
            wds.select(filter_no_caption_or_no_image),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(image="jpg;png;jpeg;webp", text="txt"),
            wds.map_dict(image=preprocess_img),
            wds.map(_dual_tokenize),
            wds.to_tuple("image", "text", "text2"),
            wds.batched(args.batch_size, partial=not is_train,
                        collation_fn=_collate_dino if use_dinov3 else wds.filters.default_collation_fn)
        ])
    else:
        pipeline.extend([
            wds.select(filter_no_caption_or_no_image),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(image="jpg;png;jpeg;webp", text="txt"),
            wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
            wds.to_tuple("image", "text"),
            wds.batched(args.batch_size, partial=not is_train,
                        collation_fn=_collate_dino if use_dinov3 else wds.filters.default_collation_fn)
        ])

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None, **kwargs):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    # 区域选框策略：只有 random-epoch 需要把 epoch 同步进 worker 进程
    region_select = getattr(args, 'region_select', 'order') if is_train else 'order'
    shared_epoch = SharedEpoch(epoch=epoch) if region_select == 'random-epoch' else None
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer,
        caption2_key=getattr(args, 'csv_caption2_key', None),
        tokenizer2=kwargs.get('tokenizer_secondary', None),
        region_key=getattr(args, 'csv_region_key', None) if is_train else None,
        max_region=getattr(args, 'max_region', 12),
        region_select=region_select,
        region_seed=getattr(args, 'seed', 0),
        shared_epoch=shared_epoch,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler, shared_epoch)


class SyntheticDataset(Dataset):

    def __init__(
            self,
            transform=None,
            image_size=(224, 224),
            caption="Dummy caption",
            dataset_size=100,
            tokenizer=None,
            tokenizer_secondary=None,
            tokenizer_list=None,
    ):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]
        self.preprocess_txt2 = (lambda text: tokenizer_secondary(text)[0]) if tokenizer_secondary else None
        self.tokenizer_list = tokenizer_list

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        if self.tokenizer_list is not None:
            texts = tuple(tok(self.caption)[0] for tok in self.tokenizer_list)
            return (image, *texts)
        if self.preprocess_txt2 is not None:
            return image, self.preprocess_txt(self.caption), self.preprocess_txt2(self.caption)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None, **kwargs):
    image_size = preprocess_fn.transforms[0].size
    tokenizer_secondary = kwargs.get('tokenizer_secondary', None)
    tokenizer_list = kwargs.get('tokenizer_list', None)
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples,
        tokenizer=tokenizer, tokenizer_secondary=tokenizer_secondary,
        tokenizer_list=tokenizer_list)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_video_frame_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None, **kwargs):
    input_path = args.train_data if is_train else args.val_data
    assert input_path
    dataset = VideoFrameDataset(
        root=input_path,
        transforms=preprocess_fn,
        tokenizer=tokenizer,
        max_side=getattr(args, 'video_max_side', 768),
        split="train" if is_train else "eval",
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "video_frame":
        return get_video_frame_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns, epoch=0, tokenizer=None, tokenizer_secondary=None, tokenizer_list=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.dataset_type == "synthetic":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer,
            tokenizer_secondary=tokenizer_secondary, tokenizer_list=tokenizer_list)

    if args.val_data:
        # video_frame: val 走 split.json 中的 eval 子集（同一根目录）
        if args.dataset_type == 'video_frame' and os.path.isfile(os.path.join(args.val_data, 'split.json')):
            val_dataset_type = 'video_frame'
        elif args.dataset_type == 'webdataset':
            val_dataset_type = 'auto'
        else:
            val_dataset_type = args.dataset_type
        data["val"] = get_dataset_fn(args.val_data, val_dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data
