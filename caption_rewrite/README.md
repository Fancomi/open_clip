# caption_rewrite — CC3M caption 改写 (dspy + GEPA)

用 GEPA 优化改写 prompt: 把 CLIP-BPE 稀有 token 换成常用词, 保真优先。与主训练解耦。

**闭集口径 (关键)**: 数据集是闭集, 优化目标是数据间互连而非绝对真实度, 故允许"向上抽象"——
把长尾具体词换成图片仍满足的更常见上位类别 (某种鸟名→水生鸟, currant→berry, ramen→noodles),
这正是把稀有词聚到共享常见词、增强互连的核心手段。只禁"改成图里没有的东西"(cat→dog/红→蓝/
二→三) 和"删掉整个物体"。

## 环境
```bash
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH=./src:$PYTHONPATH   # 工作目录 open_clip/
```

## 流程
1. 起 student 端点 (吃 GPU, tmux, gemma 8 端点 8001-8008):
   `bash run_gemma4_sgl.sh -p 8001 -g 0 -n 4` + `-p 8005 -g 4 -n 4`
   (脚本在 llm_infer/sport_ontology/vllm_deploy/); 探活 `curl http://127.0.0.1:8001/v1/models`
2. 全量 BPE 频次 + 定阈值 N:
   `python -m caption_rewrite.bpe_freq --tars '/root/paddlejob/gpfsspace/cc3m-wds/cc3m-train-*.tar'`
   看 `outputs/bpe_freq_dist.png` 定 N 写入 `outputs/config.json` (当前 N=100, freq_mode=word)
3. 采样数据 (只收含稀有 token 的句子):
   `python caption_rewrite/sample_data.py --tars '/root/.../cc3m-train-000[0-9].tar' --n-train 1000 --n-val 200`
4. 优化:
   - 冒烟: `SMOKE=1 python -m caption_rewrite.optimize --student gemma`
   - 正式: `python -m caption_rewrite.optimize --student gemma --max-metric-calls 300`
   - 两 student 对比: 再跑 `--student qwen`
5. 诊断稀有词构成 (无模型): `python -m caption_rewrite.diagnose_rare`
6. 全量改写落盘: `python -m caption_rewrite.apply --student gemma --ports 8001,...,8008 --num-threads 24`
   然后 `python -m caption_rewrite.apply --merge` 合并分片为 `outputs/rewritten/all.jsonl`

## 指标 (逐句, 保真优先)
- 保真硬闸: teacher 判"图文是否仍匹配"(允许向上抽象, 只禁 false/删物体), 不通过 score=0
  (unfaithful_score=0.0, 硬乘子不可交易)
- 达标后 score = max(0, 稀有词降幅率 − λ·归一化编辑距离), λ 默认 0.3
- val 报告: score / rare_reduction / edit / faithful_rate → `outputs/report_{student}.json`
- 学到的新指令 → `outputs/optimized_prompt_{student}.txt`

## 模型
- student: 本地 vLLM/sglang gemma(8001-8)/qwen(8005-8, 与 gemma 端口互斥), 关思考模式
- teacher: `openai/gpt-5.6-sol` 走 ~/.claude/settings.json 厂内代理 (OpenAI chat 协议, base 补 /v1);
  兼 GEPA reflection_lm + 保真裁判。(历史: 早期用 anthropic/Opus 4.8, 见 lab_lm.make_teacher 注释)

## 实验结论 (train1000/val200, 优化口径演进)
口径从"严格保原意"逐步放宽到"闭集允许向上抽象", 保真与降词首次同时到高位:

| 版本 | 口径 | 降词率 | faithful | 说明 |
|---|---|---|---|---|
| v3 | 硬闸但混判泛化=篡改 | 0.52 | 0.49 崩 | 目标自相矛盾 |
| v4 | 只准完全等义替换 | 0.23 | 0.89 | 保真好但降词被焊死 |
| **v5 gemma** | **放行向上抽象** | **0.638** | **1.00** | **最佳, 落盘用它** |
| v5 qwen | 同上 | 0.635 | 1.00 | gemma edit 更小, 选 gemma |

gemma 优胜提示词 `outputs/optimized_prompt_gemma.txt` 自己学到一条我们没写的红线:
`iconic→famous` 不行, 因为 famous 断言了图里看不出的现实名气 (抽象须停在图片可验证范围内)。

## 全量落盘产物 (2026-07-10)
- 全量改写: 2,919,397 caption, 改写 270,265 (9.26%), 用时约 1h45min (8 端点并发)
  → `outputs/rewritten/all.jsonl` (504M, 每行 {key, original, rewritten, changed})
- 双版本训练 TSV (图片单份, filepath 逐字节一致):
  `datas/cc3m-tsv/annotations/clip_train_{orig,rewritten}.tsv` (各 2.9M 行)
  由 `scripts/data/build_cc3m_rewrite_tsv.py` 生成 (VERSIONS 字典可扩展多版本)
- 原始 wds 已迁至 `/root/paddlejob/gpfsspace/cc3m-wds/` (592 tar, 校验后删源)

## 文件
- `bpe_freq.py`   CLIP-BPE token 频次统计 + rare_ids(word/token 模式)/count_rare + 分布图
- `sample_data.py` 采样并筛含稀有 token 的 caption → train/val jsonl
- `lab_lm.py`     student(本地) + teacher(gpt-5.6-sol 代理) 接线
- `rewrite_program.py` dspy Module: caption → rewritten_caption
- `metric.py`     逐句评分 (保真硬闸 + 稀有词降幅 − 编辑距离)
- `diagnose_rare.py` 稀有词构成诊断 (无模型): 专名/数字仅 0.7%, 大头是可抽象的具体词
- `optimize.py`   GEPA 优化主入口 (SMOKE 冒烟)
- `apply.py`      全量改写落盘 (分片 + 断点续跑 + --ports 多卡并发 + --merge)
- `tests/`        bpe_freq + metric 单测 (pytest caption_rewrite/tests/ -v)
