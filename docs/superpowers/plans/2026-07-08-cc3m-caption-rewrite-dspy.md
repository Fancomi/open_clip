# CC3M Caption 改写实验 (dspy + GEPA) 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 dspy+GEPA 优化 caption 改写 prompt，把 CC3M caption 里 CLIP-BPE 稀有 token 换成常用词，保真优先。

**Architecture:** `bpe_freq.py` 用 CLIP 文本塔 BPE 全量统计 token 频次并冻结（分布图定阈值 N）；只挑含稀有 token 的 caption 进 GEPA；student(gemma/qwen) 改写，teacher(Opus 走厂内代理) 做语义保真硬闸 + GEPA 反思；评分 = 保真达标后算稀有词降幅减编辑距离惩罚。本次只做优化+小样本验证，全量重写留 `--apply` 接口。

**Tech Stack:** dspy 3.2.1 (GEPA), open_clip SimpleTokenizer (BPE), 本地 sglang/vllm (gemma/qwen), Opus 4.8 via litellm, pytest。

**约定:** 全程 `source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate` + `export PYTHONPATH=./src:$PYTHONPATH`，工作目录 `open_clip/`。测试 `pytest caption_rewrite/tests/ -v`。

---

## 文件结构

```
open_clip/caption_rewrite/
├── README.md            # 用法与实验说明
├── lab_lm.py            # student(本地vLLM) + make_teacher(Opus代理) 接线
├── bpe_freq.py          # CLIP BPE 全量 token 频次统计 + is_rare/count_rare + 分布图
├── rewrite_program.py   # dspy Module: caption → rewritten_caption
├── metric.py            # 逐句评分: 保真硬闸(Opus) + 稀有词降幅 − 编辑距离惩罚
├── sample_data.py       # 从 CC3M wds 采样并筛"含稀有token"的句子 → train/val jsonl
├── optimize.py          # GEPA 优化主入口 (student gemma/qwen 切换, --apply 预留)
├── serve_models.sh      # tmux 起 gemma×4 + qwen×4 端点
├── tests/
│   ├── test_bpe_freq.py
│   └── test_metric.py
├── outputs/             # 冻结频次表/分布图/优化prompt/报告 (gitignore)
└── data/                # 采样 jsonl (gitignore)
```

---

## Task 0: 目录与 gitignore

**Files:**
- Create: `caption_rewrite/__init__.py` (空), `caption_rewrite/tests/__init__.py` (空)
- Modify: `.gitignore`

- [ ] **Step 1: 建目录骨架**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/vision_encoder/open_clip
mkdir -p caption_rewrite/tests caption_rewrite/outputs caption_rewrite/data
touch caption_rewrite/__init__.py caption_rewrite/tests/__init__.py
```

- [ ] **Step 2: gitignore 忽略产物与数据**

在 `.gitignore` 末尾追加：
```
caption_rewrite/outputs/
caption_rewrite/data/
```

- [ ] **Step 3: Commit**

```bash
git add caption_rewrite/__init__.py caption_rewrite/tests/__init__.py .gitignore
git commit -m "feat(caption_rewrite): 目录骨架与 gitignore"
```

---

## Task 1: BPE 频次统计与稀有词判定 (bpe_freq.py)

**Files:**
- Create: `caption_rewrite/bpe_freq.py`
- Test: `caption_rewrite/tests/test_bpe_freq.py`

**接口约定** (后续 Task 复用，签名固定):
- `get_tokenizer()` → 返回 `open_clip.get_tokenizer('PE-Core-B-16-dinov3')` 的内层 `SimpleTokenizer`
- `encode_ids(caption: str) -> list[int]` → 纯 BPE id 列表 (用 `tokenizer.encode`，不含 SOT/EOT/pad)
- `count_tar_tokens(tars: list[str]) -> tuple[Counter, dict]` → (token_id→count, stats)
- `load_freq(path) -> dict[int,int]` → 从 json 读回 `{token_id: count}`
- `rare_ids(freq: dict, n: int) -> set[int]` → 频次 < n 的 token id 集合
- `count_rare(caption: str, rare_set: set[int]) -> int` → caption 中稀有 token 出现次数

- [ ] **Step 1: 写失败测试**

```python
# caption_rewrite/tests/test_bpe_freq.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from caption_rewrite import bpe_freq


def test_encode_ids_pure_bpe():
    ids = bpe_freq.encode_ids("a red zebra")
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
    # 不含 SOT(49406)/EOT(49407)
    assert 49406 not in ids and 49407 not in ids
    assert len(ids) >= 3


def test_rare_ids_and_count_rare():
    freq = {10: 100, 20: 5, 30: 1}
    rs = bpe_freq.rare_ids(freq, n=10)   # <10 → {20,30}
    assert rs == {20, 30}
    # 构造一个已知 caption 的稀有计数: 用其真实 ids 造 freq 表
    ids = bpe_freq.encode_ids("zebra")
    freq2 = {i: 1 for i in ids}          # 全部设为频次1
    rs2 = bpe_freq.rare_ids(freq2, n=5)  # 全稀有
    assert bpe_freq.count_rare("zebra", rs2) == len(ids)
    assert bpe_freq.count_rare("zebra", set()) == 0
```

- [ ] **Step 2: 运行验证失败**

Run: `pytest caption_rewrite/tests/test_bpe_freq.py -v`
Expected: FAIL (ModuleNotFoundError / AttributeError: bpe_freq 无该函数)

- [ ] **Step 3: 实现 bpe_freq.py**

```python
"""CLIP 文本塔 BPE token 频次统计 + 稀有词判定。

冻结的 token 频次表是评分锚: token 频次 < N 视为稀有。
用 open_clip 的 SimpleTokenizer.encode() 取纯 BPE id (不含 SOT/EOT/pad),
与 CLIP 训练时真正见到的单位一致。
"""
import argparse
import glob
import json
import logging
import os
import sys
import tarfile
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import open_clip

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

_TOK = None


def get_tokenizer():
    global _TOK
    if _TOK is None:
        _TOK = open_clip.get_tokenizer('PE-Core-B-16-dinov3')
    return _TOK


def encode_ids(caption):
    """纯 BPE id 列表 (SimpleTokenizer.encode, 不含特殊符)。"""
    return list(get_tokenizer().encode(caption))


def count_tar_tokens(tars):
    """流式遍历 wds tar 的 .txt caption, 累计 BPE token-id 频次。"""
    freq = Counter()
    n_cap = n_tok = 0
    for ti, tar_path in enumerate(tars):
        with tarfile.open(tar_path, 'r') as tar:
            for member in tar:
                if not member.name.endswith('.txt'):
                    continue
                fobj = tar.extractfile(member)
                if fobj is None:
                    continue
                caption = fobj.read().decode('utf-8', errors='ignore')
                ids = encode_ids(caption)
                freq.update(ids)
                n_cap += 1
                n_tok += len(ids)
        log.info(f'[bpe_freq] tar {ti + 1}/{len(tars)} captions={n_cap} tokens={n_tok} vocab={len(freq)}')
    return freq, dict(captions=n_cap, tokens=n_tok, vocab=len(freq))


def load_freq(path):
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    return {int(k): int(v['count'] if isinstance(v, dict) else v) for k, v in raw.items()}


def rare_ids(freq, n):
    return {int(tid) for tid, c in freq.items() if int(c) < int(n)}


def count_rare(caption, rare_set):
    return sum(1 for i in encode_ids(caption) if i in rare_set)


def _save(freq, out_dir):
    tok = get_tokenizer()
    os.makedirs(out_dir, exist_ok=True)
    ordered = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    obj = {str(tid): {'count': c, 'subword': tok.decode([tid])} for tid, c in ordered}
    path = os.path.join(out_dir, 'bpe_freq.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    log.info(f'[bpe_freq] wrote {path}')
    return path


def _plot_dist(freq, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    counts = np.array(sorted(freq.values(), reverse=True), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].loglog(np.arange(1, len(counts) + 1), counts, color='#4C78A8')
    axes[0].set_xlabel('Token rank'); axes[0].set_ylabel('Frequency')
    axes[0].set_title('BPE token frequency (rank-freq, log-log)')
    axes[0].grid(alpha=0.3)
    for thr in (10, 50, 100, 500, 1000):
        n_below = int((counts < thr).sum())
        axes[1].bar(str(thr), n_below, color='#E45756')
        axes[1].text(str(thr), n_below, f' {n_below}', ha='center', va='bottom', fontsize=8)
    axes[1].set_xlabel('Threshold N'); axes[1].set_ylabel('# tokens with freq < N')
    axes[1].set_title('Rare-token count vs threshold')
    fig.tight_layout()
    path = os.path.join(out_dir, 'bpe_freq_dist.png')
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    log.info(f'[bpe_freq] wrote {path}')


def main():
    p = argparse.ArgumentParser(description='CC3M CLIP-BPE token frequency')
    p.add_argument('--tars', required=True, help='tar glob')
    p.add_argument('--out-dir', default='caption_rewrite/outputs')
    args = p.parse_args()
    tars = sorted(glob.glob(args.tars))
    if not tars:
        raise SystemExit(f'no tar matched: {args.tars}')
    freq, stats = count_tar_tokens(tars)
    _save(freq, args.out_dir)
    _plot_dist(freq, args.out_dir)
    with open(os.path.join(args.out_dir, 'bpe_freq_summary.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    log.info(f'[bpe_freq] {stats}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: 运行验证通过**

Run: `pytest caption_rewrite/tests/test_bpe_freq.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: 冒烟 2 个 tar 确认 CLI 与产物**

Run:
```bash
python -m caption_rewrite.bpe_freq \
  --tars '/root/paddlejob/workspace/env_run/penghaotian/datas/cc3m-wds/cc3m-train-000[01].tar' \
  --out-dir /tmp/bpe_smoke
```
Expected: 打印 captions/tokens/vocab，`/tmp/bpe_smoke/` 下有 `bpe_freq.json` + `bpe_freq_dist.png` + summary。之后 `rm -rf /tmp/bpe_smoke`。

- [ ] **Step 6: Commit**

```bash
git add caption_rewrite/bpe_freq.py caption_rewrite/tests/test_bpe_freq.py
git commit -m "feat(caption_rewrite): CLIP-BPE 频次统计与稀有词判定"
```

---

## Task 2: 全量 BPE 频次并定阈值 N

**Files:** 无新代码，运行 Task 1 的 CLI 产出冻结表。

- [ ] **Step 1: 全量统计 (后台, ~7-10 分钟)**

Run:
```bash
mkdir -p caption_rewrite/outputs
nohup python -m caption_rewrite.bpe_freq \
  --tars '/root/paddlejob/workspace/env_run/penghaotian/datas/cc3m-wds/cc3m-train-*.tar' \
  --out-dir caption_rewrite/outputs \
  > caption_rewrite/outputs/bpe_freq.log 2>&1 &
```
Expected: 后台 PID；`tail -f caption_rewrite/outputs/bpe_freq.log` 看进度到 `captions=2905954` 附近。

- [ ] **Step 2: 看分布图定 N**

查看 `caption_rewrite/outputs/bpe_freq_dist.png` 右图 (各阈值下稀有 token 数)，
结合 `bpe_freq_summary.json` 的 vocab 总量，**向用户报告分布并请其确认 N** (默认建议 N=100)。
记入 config：
```bash
python3 -c "import json; json.dump({'rare_threshold_n': 100}, open('caption_rewrite/outputs/config.json','w'), indent=2)"
```

- [ ] **Step 3: Commit (outputs 已 gitignore, 用 -f 记录阈值)**

```bash
git add -f caption_rewrite/outputs/config.json
git commit -m "chore(caption_rewrite): 冻结稀有词阈值 N"
```

---

## Task 3: 采样并筛含稀有 token 的 caption (sample_data.py)

**Files:**
- Create: `caption_rewrite/sample_data.py`

**依赖:** Task 1 的 `bpe_freq.load_freq/rare_ids/count_rare`；Task 2 的 `outputs/bpe_freq.json` + `config.json`。

- [ ] **Step 1: 实现 sample_data.py**

```python
"""从 CC3M wds 采样 caption, 只保留含稀有 BPE token 的句子, 切 train/val jsonl。

只有含稀有 token 的 caption 才需重写, 因此数据集只收这类句子。
"""
import argparse
import glob
import json
import logging
import os
import random
import sys
import tarfile

sys.path.insert(0, os.path.dirname(__file__))
import bpe_freq

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)


def iter_captions(tars, limit_scan):
    seen = 0
    for tar_path in tars:
        with tarfile.open(tar_path, 'r') as tar:
            for member in tar:
                if not member.name.endswith('.txt'):
                    continue
                fobj = tar.extractfile(member)
                if fobj is None:
                    continue
                yield fobj.read().decode('utf-8', errors='ignore').strip()
                seen += 1
                if limit_scan and seen >= limit_scan:
                    return


def main():
    p = argparse.ArgumentParser(description='采样含稀有 token 的 caption')
    p.add_argument('--tars', required=True)
    p.add_argument('--freq', default='caption_rewrite/outputs/bpe_freq.json')
    p.add_argument('--config', default='caption_rewrite/outputs/config.json')
    p.add_argument('--out-dir', default='caption_rewrite/data')
    p.add_argument('--n-train', type=int, default=40)
    p.add_argument('--n-val', type=int, default=20)
    p.add_argument('--limit-scan', type=int, default=50000, help='最多扫描多少条 caption')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    n = json.load(open(args.config))['rare_threshold_n']
    rare = bpe_freq.rare_ids(bpe_freq.load_freq(args.freq), n)
    log.info(f'[sample] N={n}, rare token 数={len(rare)}')

    tars = sorted(glob.glob(args.tars))
    pool = []
    for cap in iter_captions(tars, args.limit_scan):
        if not cap:
            continue
        nr = bpe_freq.count_rare(cap, rare)
        if nr > 0:
            pool.append({'caption': cap, 'n_rare': nr})
    log.info(f'[sample] 含稀有 token 的句子 {len(pool)} 条')

    random.Random(args.seed).shuffle(pool)
    need = args.n_train + args.n_val
    if len(pool) < need:
        raise SystemExit(f'含稀有词句子不足: {len(pool)} < {need}, 调大 --limit-scan')
    train, val = pool[:args.n_train], pool[args.n_train:need]

    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in (('train', train), ('val', val)):
        path = os.path.join(args.out_dir, f'{name}.jsonl')
        with open(path, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        log.info(f'[sample] wrote {path} ({len(rows)})')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 生成小样本数据集**

Run:
```bash
python caption_rewrite/sample_data.py \
  --tars '/root/paddlejob/workspace/env_run/penghaotian/datas/cc3m-wds/cc3m-train-000[0-9].tar' \
  --n-train 40 --n-val 20 --limit-scan 50000
```
Expected: `caption_rewrite/data/train.jsonl` (40 行) + `val.jsonl` (20 行)，每行含 `caption` + `n_rare`>0。

- [ ] **Step 3: Commit**

```bash
git add caption_rewrite/sample_data.py
git commit -m "feat(caption_rewrite): 采样含稀有 token 的 caption 数据集"
```

---

## Task 4: 模型接线 lab_lm.py (student 本地 + teacher Opus)

**Files:**
- Create: `caption_rewrite/lab_lm.py`

**背景:** student 沿用 prompt_lab 本地 vLLM 接线；teacher 读 `~/.claude/settings.json` 的
`ANTHROPIC_BASE_URL`(=`https://oneapi-comate.baidu-int.com`) + `ANTHROPIC_AUTH_TOKEN` +
`ANTHROPIC_CUSTOM_HEADERS`(格式 `"名:JSON"`，拆成 `{名: JSON}` 传 extra_headers)。
model id 用 `ANTHROPIC_DEFAULT_OPUS_MODEL`(=`Opus 4.8`)。

- [ ] **Step 1: 实现 lab_lm.py**

```python
"""dspy 模型接线: student=本地 vLLM(gemma/qwen), teacher=Opus(厂内代理)。

student 坑位同 prompt_lab: OpenAI 兼容端点 + 关思考模式。
teacher 从 ~/.claude/settings.json 读厂内代理配置 (非公网 anthropic)。
"""
import json
import os

import dspy

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
for _k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(_k, None)  # 本地回环 + 厂内代理都不走系统代理

MODELS = {
    "qwen":  ("/dev/shm/models/Qwen3.6-35B-A3B-FP8", [8005, 8006, 8007, 8008]),
    "gemma": ("/dev/shm/models/gemma-4-26B-A4B-it",  [8001, 8002, 8003, 8004]),
}
_SETTINGS = os.path.expanduser("~/.claude/settings.json")


def make_student(which="qwen", *, port=None, think=False, temperature=0.3,
                 max_tokens=1024, **kw):
    """本地 vLLM student。think=False 关思考模式 (dspy 必需)。"""
    model_id, ports = MODELS[which]
    return dspy.LM(
        f"openai/{model_id}",
        api_base=f"http://127.0.0.1:{port or ports[0]}/v1",
        api_key="EMPTY", temperature=temperature, max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": think}},
        **kw,
    )


def _parse_custom_headers(raw):
    """'comate_custom_header:{...json...}' -> {'comate_custom_header': '{...json...}'}"""
    if not raw or ":" not in raw:
        return {}
    name, val = raw.split(":", 1)
    return {name.strip(): val.strip()}


def make_teacher(*, temperature=1.0, max_tokens=4096, **kw):
    """Opus teacher, 走 ~/.claude/settings.json 厂内代理。GEPA reflection + 保真裁判共用。"""
    env = json.load(open(_SETTINGS)).get("env", {})
    base = env["ANTHROPIC_BASE_URL"].strip().rstrip("/")
    token = env["ANTHROPIC_AUTH_TOKEN"].strip()
    model = env.get("ANTHROPIC_DEFAULT_OPUS_MODEL", "Opus 4.8").strip()
    headers = _parse_custom_headers(env.get("ANTHROPIC_CUSTOM_HEADERS", ""))
    return dspy.LM(
        f"openai/{model}",
        api_base=f"{base}/v1", api_key=token,
        temperature=temperature, max_tokens=max_tokens,
        extra_headers=headers or None, **kw,
    )
```

- [ ] **Step 2: 验证 teacher 配置可读 + student 可构造**

Run:
```bash
python3 -c "
import sys; sys.path.insert(0,'caption_rewrite')
import lab_lm
t = lab_lm.make_teacher()
print('teacher model:', t.model)
s = lab_lm.make_student('qwen')
print('student model:', s.model)
"
```
Expected: `teacher model: openai/Opus 4.8`、`student model: openai//dev/shm/models/Qwen3.6-35B-A3B-FP8`。不实际发请求。

- [ ] **Step 3: 验证 teacher 实连 (需外网/厂内网, 单次最小请求)**

Run:
```bash
python3 -c "
import sys; sys.path.insert(0,'caption_rewrite')
import lab_lm, dspy
t = lab_lm.make_teacher(max_tokens=32)
print(t('Reply with the single word: ok'))
"
```
Expected: 返回含 'ok' 的回复。若超时/401 → 报告用户核对 settings，不阻塞后续 (metric 有降级)。

- [ ] **Step 4: Commit**

```bash
git add caption_rewrite/lab_lm.py
git commit -m "feat(caption_rewrite): student(本地) + teacher(Opus代理) 接线"
```

---

## Task 5: 改写程序 rewrite_program.py

**Files:**
- Create: `caption_rewrite/rewrite_program.py`

- [ ] **Step 1: 实现 rewrite_program.py**

```python
"""dspy 改写程序: caption -> rewritten_caption。

策略: 仅把不常用词替换为常用同义词; 严格保原意, 不加不删信息; 无稀有词则原样返回。
GEPA 会在此 instructions 基础上反思进化。
"""
import dspy


class RewriteCaption(dspy.Signature):
    """Rewrite an image caption to use only common, everyday vocabulary.

    Replace rare or unusual words with their most common synonyms.
    Keep the original meaning exactly: do not add, remove, or invent any
    information, objects, attributes, or actions. Change as few words as
    possible. If every word is already common, return the caption unchanged.
    Output lowercase, no quotes, no explanation."""

    caption = dspy.InputField(desc="original image caption")
    rewritten_caption = dspy.OutputField(desc="caption with rare words replaced by common ones, meaning preserved")


class Rewriter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(RewriteCaption)

    def forward(self, caption):
        return self.predict(caption=caption)
```

- [ ] **Step 2: 冒烟 (需 student 端点已起, 见 Task 7 serve)**

Run:
```bash
python3 -c "
import sys; sys.path.insert(0,'caption_rewrite')
import dspy, lab_lm, rewrite_program
dspy.configure(lm=lab_lm.make_student('qwen'))
r = rewrite_program.Rewriter()
print(r(caption='a zebra zooms across the savanna').rewritten_caption)
"
```
Expected: 返回一句改写 (如 'a zebra runs fast across the grassland')。端点未起则跳过此步，Task 8 冒烟统一验证。

- [ ] **Step 3: Commit**

```bash
git add caption_rewrite/rewrite_program.py
git commit -m "feat(caption_rewrite): dspy 改写程序 Signature/Module"
```

---

## Task 6: 逐句评分 metric.py (保真硬闸 + 稀有词降幅 − 编辑距离)

**Files:**
- Create: `caption_rewrite/metric.py`
- Test: `caption_rewrite/tests/test_metric.py`

**评分逻辑:**
```
teacher 判定语义歪曲 → score = unfaithful_score(0.1), feedback 指出哪里歪
否则 → score = max(0, 稀有词降幅率 − λ·归一化编辑距离)
        稀有词降幅率 = (orig_rare − new_rare)/max(orig_rare,1) 夹到 [0,1]; orig==0 记 1.0
        归一化编辑距离 = levenshtein(orig,new)/max(len)
```
保真裁判抽成模块级 `judge_faithful(teacher, orig, new)`，稀有计数抽成 `_count_rare`，
便于测试 monkeypatch。纯本地量不调 teacher。

- [ ] **Step 1: 写失败测试 (本地纯函数 + mock teacher)**

```python
# caption_rewrite/tests/test_metric.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import metric


def test_norm_levenshtein():
    assert metric.norm_levenshtein("abc", "abc") == 0.0
    assert abs(metric.norm_levenshtein("abc", "abd") - 1 / 3) < 1e-9
    assert metric.norm_levenshtein("", "") == 0.0


def test_rare_reduction_rate():
    assert abs(metric.rare_reduction_rate(3, 1) - 2 / 3) < 1e-9
    assert metric.rare_reduction_rate(1, 3) == 0.0
    assert metric.rare_reduction_rate(0, 0) == 1.0


def test_score_faithful_path(monkeypatch):
    monkeypatch.setattr(metric, "judge_faithful", lambda t, o, n: (True, "ok"))
    monkeypatch.setattr(metric, "_count_rare", lambda cap: 2 if cap == "orig" else 0)
    gold = type("G", (), {"caption": "orig"})()
    pred = type("P", (), {"rewritten_caption": "new"})()
    out = metric.make_metric(teacher=None, rare_set=set(), lam=0.3)(gold, pred)
    assert 0.5 < out.score <= 1.0


def test_score_unfaithful_path(monkeypatch):
    monkeypatch.setattr(metric, "judge_faithful", lambda t, o, n: (False, "改变了颜色"))
    monkeypatch.setattr(metric, "_count_rare", lambda cap: 2 if cap == "orig" else 0)
    gold = type("G", (), {"caption": "orig"})()
    pred = type("P", (), {"rewritten_caption": "new"})()
    out = metric.make_metric(teacher=None, rare_set=set(), lam=0.3)(gold, pred)
    assert out.score <= 0.2
    assert "改变了颜色" in out.feedback
```

- [ ] **Step 2: 运行验证失败**

Run: `pytest caption_rewrite/tests/test_metric.py -v`
Expected: FAIL (metric 无这些函数)

- [ ] **Step 3: 实现 metric.py**

```python
"""逐句评分: 保真优先。teacher(Opus) 判语义保真作硬闸,
达标后 score = max(0, 稀有词降幅率 − λ·归一化编辑距离)。
"""
import logging

import dspy

log = logging.getLogger(__name__)

_RARE_SET = set()   # 由 make_metric 注入


def _count_rare(caption):
    import bpe_freq
    return bpe_freq.count_rare(caption, _RARE_SET)


def norm_levenshtein(a, b):
    """归一化编辑距离 ∈ [0,1]。"""
    if not a and not b:
        return 0.0
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, lb + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[lb] / max(la, lb)


def rare_reduction_rate(orig_rare, new_rare):
    """(orig-new)/max(orig,1) 夹到 [0,1]; orig==0 记 1.0 (无需改)。"""
    if orig_rare == 0:
        return 1.0
    return max(0.0, (orig_rare - new_rare) / orig_rare)


class _Faithful(dspy.Signature):
    """Judge whether the rewritten caption preserves the EXACT meaning of the
    original: same objects, attributes, actions, counts, spatial relations.
    Answer faithful=yes only if nothing was added, removed, or distorted."""
    original = dspy.InputField()
    rewritten = dspy.InputField()
    faithful = dspy.OutputField(desc="yes or no")
    reason = dspy.OutputField(desc="short reason, name any distortion")


def judge_faithful(teacher, original, rewritten):
    """返回 (是否保真, 理由)。teacher 失败降级 (True,'judge-failed') 不阻塞。"""
    try:
        with dspy.context(lm=teacher):
            r = dspy.Predict(_Faithful)(original=original, rewritten=rewritten)
        ok = str(r.faithful).strip().lower().startswith("y")
        return ok, str(r.reason)
    except Exception as e:
        log.warning(f"[metric] teacher judge failed: {e}")
        return True, "judge-failed"


def make_metric(teacher, rare_set, lam=0.3, unfaithful_score=0.1):
    """构造 GEPA metric。rare_set 冻结注入; teacher 做保真硬闸。"""
    global _RARE_SET
    _RARE_SET = set(rare_set)

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        orig = gold.caption
        new = getattr(pred, "rewritten_caption", "") or ""
        ok, reason = judge_faithful(teacher, orig, new)
        if not ok:
            return dspy.Prediction(
                score=unfaithful_score,
                feedback=f"语义被改变(硬闸不通过): {reason}. 必须严格保原意, 仅替换稀有词。")
        o_rare, n_rare = _count_rare(orig), _count_rare(new)
        red = rare_reduction_rate(o_rare, n_rare)
        edit = norm_levenshtein(orig, new)
        score = max(0.0, red - lam * edit)
        fb = f"保真通过。稀有token {o_rare}->{n_rare} (降幅率{red:.2f}), 编辑距离{edit:.2f}。"
        if n_rare > 0:
            fb += f" 仍有 {n_rare} 个稀有token未替换, 用更常见的词替代。"
        if edit > 0.5:
            fb += " 改动过大, 尽量少改词。"
        return dspy.Prediction(score=score, feedback=fb)

    return metric
```

- [ ] **Step 4: 运行验证通过**

Run: `pytest caption_rewrite/tests/test_metric.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add caption_rewrite/metric.py caption_rewrite/tests/test_metric.py
git commit -m "feat(caption_rewrite): 逐句评分 (保真硬闸+稀有词降幅-编辑距离)"
```

---

## Task 7: 模型服务脚本 serve_models.sh

**Files:**
- Create: `caption_rewrite/serve_models.sh`

- [ ] **Step 1: 实现 serve_models.sh**

```bash
#!/bin/bash
# tmux 起 student 端点: gemma×4(8001-8004) + qwen×4(8005-8008), 用 sglang。
# 用法: bash caption_rewrite/serve_models.sh [gemma|qwen|all]  (默认 all)
# 停止: tmux kill-session -t crw_serve
set -e
WHICH="${1:-all}"
SESSION=crw_serve
GEMMA=/dev/shm/models/gemma-4-26B-A4B-it
QWEN=/dev/shm/models/Qwen3.6-35B-A3B-FP8
ACT="source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate"

launch() {  # name model port gpu
    local name=$1 model=$2 port=$3 gpu=$4
    tmux new-window -t "$SESSION" -n "$name" \
        "$ACT; CUDA_VISIBLE_DEVICES=$gpu python -m sglang.launch_server \
         --model-path $model --port $port --mem-fraction-static 0.85 \
         2>&1 | tee /tmp/${name}.log"
}

tmux has-session -t "$SESSION" 2>/dev/null && { echo "session $SESSION 已存在"; exit 0; }
tmux new-session -d -s "$SESSION" -n init "sleep 1"

gpu=0
if [ "$WHICH" = "gemma" ] || [ "$WHICH" = "all" ]; then
    for p in 8001 8002 8003 8004; do launch "gemma_$p" "$GEMMA" $p $gpu; gpu=$((gpu+1)); done
fi
if [ "$WHICH" = "qwen" ] || [ "$WHICH" = "all" ]; then
    for p in 8005 8006 8007 8008; do launch "qwen_$p" "$QWEN" $p $gpu; gpu=$((gpu+1)); done
fi
echo "已在 tmux session '$SESSION' 起服务。curl http://127.0.0.1:8001/v1/models 探活。"
echo "停止: tmux kill-session -t $SESSION"
```

- [ ] **Step 2: 语法检查 (不实际起服务)**

Run: `bash -n caption_rewrite/serve_models.sh && echo OK`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add caption_rewrite/serve_models.sh
git commit -m "feat(caption_rewrite): tmux 起 student 端点脚本"
```

---

## Task 8: GEPA 优化主入口 optimize.py

**Files:**
- Create: `caption_rewrite/optimize.py`

**依赖:** Task 1(bpe_freq)、4(lab_lm)、5(rewrite_program)、6(metric)；Task 2 冻结表；Task 3 数据集。

- [ ] **Step 1: 实现 optimize.py**

```python
"""GEPA 优化 caption 改写 prompt。student(gemma/qwen) 改写, teacher(Opus) 反思+保真裁判。

用法:
  # 前置: bash serve_models.sh; python -m caption_rewrite.bpe_freq ...; python sample_data.py ...
  python -m caption_rewrite.optimize --student qwen --max-metric-calls 30
  SMOKE=1 python -m caption_rewrite.optimize --student qwen   # 冒烟(小数据+少调用)
  python -m caption_rewrite.optimize --student qwen --apply   # (预留)全量重写, 本阶段不实现
"""
import argparse
import json
import os
import sys

import dspy

sys.path.insert(0, os.path.dirname(__file__))
import bpe_freq
import lab_lm
import metric as metric_mod
from rewrite_program import Rewriter


def load_examples(path):
    rows = [json.loads(l) for l in open(path, encoding='utf-8')]
    return [dspy.Example(caption=r['caption']).with_inputs('caption') for r in rows]


def avg_report(prog, dset, m):
    """在验证集上跑, 汇总三项指标。"""
    scores, reds, edits, faith = [], [], [], 0
    for ex in dset:
        pred = prog(**ex.inputs())
        out = m(ex, pred)
        scores.append(out.score)
        # 从 feedback 无法稳取分量, 直接用 metric 内部纯函数复算
        o_rare = metric_mod._count_rare(ex.caption)
        n_rare = metric_mod._count_rare(getattr(pred, 'rewritten_caption', '') or '')
        reds.append(metric_mod.rare_reduction_rate(o_rare, n_rare))
        edits.append(metric_mod.norm_levenshtein(ex.caption, getattr(pred, 'rewritten_caption', '') or ''))
        faith += int(not out.feedback.startswith('语义被改变'))
    n = len(dset)
    return dict(score=sum(scores) / n, rare_reduction=sum(reds) / n,
                edit=sum(edits) / n, faithful_rate=faith / n)


def main():
    p = argparse.ArgumentParser(description='GEPA 优化 caption 改写')
    p.add_argument('--student', choices=['qwen', 'gemma'], default='qwen')
    p.add_argument('--data-dir', default='caption_rewrite/data')
    p.add_argument('--freq', default='caption_rewrite/outputs/bpe_freq.json')
    p.add_argument('--config', default='caption_rewrite/outputs/config.json')
    p.add_argument('--out-dir', default='caption_rewrite/outputs')
    p.add_argument('--max-metric-calls', type=int, default=30)
    p.add_argument('--lam', type=float, default=0.3)
    p.add_argument('--num-threads', type=int, default=4)
    p.add_argument('--apply', action='store_true', help='(预留) 全量重写, 本阶段未实现')
    args = p.parse_args()

    if args.apply:
        raise SystemExit('[optimize] --apply 全量重写为后续阶段, 本模块未实现。')

    smoke = os.environ.get('SMOKE') == '1'
    n = json.load(open(args.config))['rare_threshold_n']
    rare = bpe_freq.rare_ids(bpe_freq.load_freq(args.freq), n)

    trainset = load_examples(os.path.join(args.data_dir, 'train.jsonl'))
    valset = load_examples(os.path.join(args.data_dir, 'val.jsonl'))
    if smoke:
        trainset, valset = trainset[:4], valset[:3]
        args.max_metric_calls = min(args.max_metric_calls, 10)

    student = lab_lm.make_student(args.student, cache=False)
    teacher = lab_lm.make_teacher()
    dspy.configure(lm=student)

    m = metric_mod.make_metric(teacher=teacher, rare_set=rare, lam=args.lam)
    program = Rewriter()

    print(f"== 优化前 (student={args.student}) ==")
    print("  ", avg_report(program, valset, m))

    gepa = dspy.GEPA(metric=m, reflection_lm=teacher,
                     max_metric_calls=args.max_metric_calls,
                     num_threads=args.num_threads, track_stats=True)
    optimized = gepa.compile(program, trainset=trainset, valset=valset)

    print("== 优化后 ==")
    rep = avg_report(optimized, valset, m)
    print("  ", rep)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = 'smoke' if smoke else args.student
    prompt_path = os.path.join(args.out_dir, f'optimized_prompt_{tag}.txt')
    with open(prompt_path, 'w', encoding='utf-8') as f:
        f.write(optimized.predict.signature.instructions)
    with open(os.path.join(args.out_dir, f'report_{tag}.json'), 'w') as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"  wrote {prompt_path}")
    print("== 学到的新指令 ==")
    print(optimized.predict.signature.instructions)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 冒烟全链路 (需端点已起 + 冻结表 + 数据集就绪)**

前置确认：`bash caption_rewrite/serve_models.sh qwen` 已起且 `curl 127.0.0.1:8005/v1/models` 通；
`caption_rewrite/outputs/bpe_freq.json` + `config.json` 存在；`caption_rewrite/data/{train,val}.jsonl` 存在。

Run:
```bash
SMOKE=1 python -m caption_rewrite.optimize --student qwen
```
Expected: 打印优化前/后三项指标 (score/rare_reduction/edit/faithful_rate)，
写出 `outputs/optimized_prompt_smoke.txt` + `report_smoke.json`，打印学到的新指令。
若 teacher 超时报 warning 但不崩溃。

- [ ] **Step 3: Commit**

```bash
git add caption_rewrite/optimize.py
git commit -m "feat(caption_rewrite): GEPA 优化主入口 (含冒烟+apply预留)"
```

---

## Task 9: 正式小样本实验 + README

**Files:**
- Create: `caption_rewrite/README.md`

- [ ] **Step 1: 跑 qwen + gemma 两 student 正式优化**

前置：`serve_models.sh all` 起全部端点。
Run:
```bash
python -m caption_rewrite.optimize --student qwen  --max-metric-calls 40
python -m caption_rewrite.optimize --student gemma --max-metric-calls 40
```
Expected: 各产出 `optimized_prompt_{qwen,gemma}.txt` + `report_{qwen,gemma}.json`。
向用户报告两者 val 指标对比 (稀有词降幅 / 保真率 / 编辑距离)。

- [ ] **Step 2: 写 README (记录用法与实验结论)**

````markdown
# caption_rewrite — CC3M caption 改写 (dspy + GEPA)

用 GEPA 优化改写 prompt: 把 CLIP-BPE 稀有 token 换成常用词, 保真优先。

## 环境
```bash
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH=./src:$PYTHONPATH   # 工作目录 open_clip/
```

## 流程
1. 起 student 端点: `bash caption_rewrite/serve_models.sh all`
2. 全量 BPE 频次 + 定阈值 N: `python -m caption_rewrite.bpe_freq --tars '.../cc3m-wds/cc3m-train-*.tar'`
   看 `outputs/bpe_freq_dist.png` 定 N 写入 `outputs/config.json`
3. 采样数据: `python caption_rewrite/sample_data.py --tars '.../cc3m-train-000[0-9].tar'`
4. 优化: `python -m caption_rewrite.optimize --student qwen --max-metric-calls 40`
   冒烟: `SMOKE=1 python -m caption_rewrite.optimize --student qwen`

## 指标 (逐句, 保真优先)
- 保真硬闸: teacher(Opus) 判是否歪曲原意, 不通过直接低分
- 达标后 score = 稀有词降幅率 − λ·归一化编辑距离
- val 报告: score / rare_reduction / edit / faithful_rate

## 模型
- student: 本地 vLLM gemma(8001-4)/qwen(8005-8), 关思考模式
- teacher: Opus 4.8 走 ~/.claude/settings.json 厂内代理, 兼 reflection_lm + 保真裁判

## 后续 (未实现)
- `optimize.py --apply`: 用最优 prompt 全量重写 2.9M caption, 重测语料级 BPE bottom
````

- [ ] **Step 3: Commit**

```bash
git add caption_rewrite/README.md
git commit -m "docs(caption_rewrite): README 用法与实验说明"
```

- [ ] **Step 4: 全部测试回归**

Run: `pytest caption_rewrite/tests/ -v`
Expected: 全 PASS (test_bpe_freq 2 + test_metric 4 = 6 passed)




