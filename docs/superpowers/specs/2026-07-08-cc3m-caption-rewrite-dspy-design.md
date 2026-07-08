# CC3M Caption 改写实验模块 (dspy + GEPA) — 设计

日期: 2026-07-08
状态: 已通过 brainstorming 评审，待写实现计划

## 1. 目标

用 dspy + GEPA 优化一个「caption 改写」prompt，让 student LLM 把 CC3M caption 里
基于 **CLIP 文本塔 BPE** 的稀有 token 替换成常用词，**在极少改动原句、严格保持原意**
的前提下降低语料中的低频 token。

两个指标：
1. **bottom 越少越好**：句子中 CLIP-BPE 稀有 token 数下降。
2. **原句变动越少越好 + 不歪曲原意**：语义保真作为硬闸（teacher 裁判），编辑距离作惩罚。

本模块边界 = **优化 + 小样本验证**（方案 A）。全量 2.9M 重写作为预留接口，标注为后续阶段。

## 2. 关键决策（brainstorming 结论）

- **评分口径**：冻结参考频次表 + 逐句稀有词惩罚（GEPA 唯一可驱动的口径）。
- **参考表数据源**：复用 `open_clip.get_tokenizer('PE-Core-B-16-dinov3')`（BPE, SimpleTokenizer，
  词表 49408）对**全量 CC3M** caption 统计 **token-id 频次**。不用正则 unigram，也不用 5 万条 slot 表。
  已验证 BPE 把稀有词切成子词（`zooms→"zo"+"oms"`, `savanna→"sav"+"anna"`），
  这正是 CLIP 训练真正见到的单位。
- **稀有判定**：频次绝对阈值 N（token 频次 < N 算稀有）。N 先跑全量分布图再定。
- **评分取向**：保真优先。语义保真是**硬闸**——teacher(Opus) 判定歪曲原意则该句直接低分，
  无论稀有词降多少。稀有词下降在保真达标前提下才计分。
- **触发条件**：仅**含稀有 token** 的 caption 才需重生成；不含则原样返回。
- **teacher**：Opus 4.8，走 `~/.claude/settings.json` 厂内代理，兼任 GEPA `reflection_lm` +
  语义保真裁判。
- **student**：gemma（8001-8004）与 qwen（8005-8008）各跑一次实验对比。

## 3. 目录布局

放 open_clip 项目内，顶层 `caption_rewrite/`（与 analysis/ 平级）。dspy 接线借
`prompt_lab/lab_lm.py` 思路但复制精简版进来，不跨项目 import，保持自洽。

```
open_clip/caption_rewrite/
├── README.md
├── lab_lm.py            # 本地 vLLM 接线 + make_teacher()(Opus 走 claude settings 代理)
├── bpe_freq.py          # CLIP BPE 全量 token 频次统计 → 冻结表 + 分布图 + is_rare/count_rare
├── rewrite_program.py   # dspy Signature/Module: caption → rewritten_caption
├── metric.py            # 逐句评分: 保真硬闸(Opus) + 稀有词降幅 − 编辑距离惩罚
├── optimize.py          # GEPA 优化主入口 (student gemma/qwen 切换, teacher Opus)
├── serve_models.sh      # tmux 起 gemma×4 + qwen×4 端点
├── outputs/             # 冻结频次表/分布图/优化后prompt/指标报告 (gitignore)
└── data/                # CC3M 采样 train/val jsonl (gitignore)
```

## 4. 数据流

```
CC3M wds ──bpe_freq.py──> 冻结BPE频次表 bpe_freq.json + 分布图(定N)
                                    │
CC3M caption ──采样──> train/val jsonl ──筛"含稀有token"的句子──> GEPA数据集
                                    │
              ┌─────────────────────┴──────────────────────┐
        rewrite_program (student: gemma/qwen)         metric (逐句评分)
        caption → rewritten_caption                    │
              │                                         │
              └──────────> GEPA优化循环 <───────────────┘
                    teacher(Opus): 反思改prompt + 保真硬闸打分
                                    │
                          优化后prompt + val指标报告
```

## 5. 核心组件接口

### ① `bpe_freq.py`
流式读 wds tar，用 CLIP tokenizer 对每条 caption 编码，统计每个 BPE token-id 频次。
- 产出 `outputs/bpe_freq.json`：`{token_id: {count, subword}}`
- 产出 token 频次分布图（供定 N）
- 提供 `load_freq()`, `count_rare(caption, N) -> int`, `rare_tokens(caption, N) -> list`
  供 metric 与数据筛选复用（同一套 tokenizer + 冻结表，口径一致）

### ② `rewrite_program.py`
```python
dspy.Signature("caption -> rewritten_caption")
# 指令核心: 仅替换不常用词为常用同义词; 严格保原意; 不加不删信息; 无稀有词原样返回
```
只暴露 `forward(caption) -> rewritten_caption`。

### ③ `metric.py`
`metric(gold, pred, trace, pred_name, pred_trace) -> dspy.Prediction(score, feedback)`：
```
teacher 判定语义歪曲  → score = 低分(0.0~0.2), feedback 指出哪里歪了     # 硬闸
否则                 → score = 稀有词降幅率 − λ·编辑距离惩罚
                       feedback 列出未替换的稀有词 + 改动过大提示
```
- 稀有词降幅率 = `(orig稀有数 − new稀有数) / max(orig稀有数,1)`，用冻结 BPE 表本地算。
- 编辑距离惩罚 = 归一化编辑距离 / 长度变化，本地算。
- 保真分 = teacher(Opus) 裁判，0~1，仅此项调 Opus（主成本）。

## 6. 模型服务与 teacher 接线

- `serve_models.sh`：tmux 起 gemma×4(8001-8004) + qwen×4(8005-8008)，沿用 prompt_lab 模型路径。
- `lab_lm.py` 加 `make_teacher()`：读 `~/.claude/settings.json` 的 `ANTHROPIC_BASE_URL` +
  `ANTHROPIC_AUTH_TOKEN` + `ANTHROPIC_CUSTOM_HEADERS`，`dspy.LM("openai/Opus 4.8", ...)`，
  走厂内代理（非公网 anthropic）。teacher 兼任 GEPA reflection_lm + 保真裁判。
- GEPA 用 `num_threads=4` 打满 student 端口并发。

## 7. 成本控制与冒烟

- teacher 保真裁判每条样本调 Opus = 主成本。用小验证集(几十条)压总量；teacher 结果缓存。
- `max_metric_calls` 先 30-50 跑通再拉高。student 关缓存量成本。
- 冒烟：`SMOKE=1` 用 5-10 条 + `max_metric_calls=10` 跑通全链路(含 Opus 连通性)，
  确认无误再上小验证集正式跑。

## 8. 错误处理

- Opus 代理超时/失败 → 该样本保真分记 0 + warning，不中断优化。
- 端点探活失败 → 启动即报错退出。
- 无稀有 token 的 caption → 不进 GEPA 数据集（重写无意义）。

## 9. 产出

- `outputs/bpe_freq.json` + 分布图
- 优化后 prompt（gemma-student / qwen-student 各一份）
- val 指标报告：稀有词降幅 / 保真率 / 平均编辑距离，gemma vs qwen 对比
- 全量重写：`optimize.py` 预留 `--apply` 接口，标注为后续阶段，本次不跑
