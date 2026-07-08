# caption_rewrite — CC3M caption 改写 (dspy + GEPA)

用 GEPA 优化改写 prompt: 把 CLIP-BPE 稀有 token 换成常用词, 保真优先。与主训练解耦。

## 环境
```bash
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH=./src:$PYTHONPATH   # 工作目录 open_clip/
```

## 流程
1. 起 student 端点 (吃 GPU, tmux): `bash caption_rewrite/serve_models.sh all`
   探活 `curl http://127.0.0.1:8005/v1/models`; 停止 `tmux kill-session -t crw_serve`
2. 全量 BPE 频次 + 定阈值 N:
   `python -m caption_rewrite.bpe_freq --tars '/root/.../cc3m-wds/cc3m-train-*.tar'`
   看 `outputs/bpe_freq_dist.png` 定 N 写入 `outputs/config.json` (当前 N=100)
3. 采样数据 (只收含稀有 token 的句子):
   `python caption_rewrite/sample_data.py --tars '/root/.../cc3m-train-000[0-9].tar'`
4. 优化:
   - 冒烟: `SMOKE=1 python -m caption_rewrite.optimize --student qwen`
   - 正式: `python -m caption_rewrite.optimize --student qwen --max-metric-calls 40`
   - 两 student 对比: 再跑 `--student gemma`

## 指标 (逐句, 保真优先)
- 保真硬闸: teacher(Opus) 判是否歪曲原意, 不通过直接低分 (unfaithful_score=0.1)
- 达标后 score = max(0, 稀有词降幅率 − λ·归一化编辑距离), λ 默认 0.3
- val 报告: score / rare_reduction / edit / faithful_rate → `outputs/report_{student}.json`
- 学到的新指令 → `outputs/optimized_prompt_{student}.txt`

## 模型
- student: 本地 vLLM/sglang gemma(8001-4)/qwen(8005-8), 关思考模式
- teacher: Opus 4.8 走 ~/.claude/settings.json 厂内代理 (anthropic messages 协议),
  兼 GEPA reflection_lm + 保真裁判

## 文件
- `bpe_freq.py`   CLIP-BPE token 频次统计 + is_rare/count_rare + 分布图
- `sample_data.py` 采样并筛含稀有 token 的 caption → train/val jsonl
- `lab_lm.py`     student(本地) + teacher(Opus 代理) 接线
- `rewrite_program.py` dspy Module: caption → rewritten_caption
- `metric.py`     逐句评分 (保真硬闸 + 稀有词降幅 − 编辑距离)
- `optimize.py`   GEPA 优化主入口 (SMOKE 冒烟 + --apply 预留)
- `serve_models.sh` tmux 起 student 端点
- `tests/`        bpe_freq + metric 单测 (pytest caption_rewrite/tests/ -v)

## 后续 (未实现)
- `optimize.py --apply`: 用最优 prompt 全量重写 2.9M caption, 重测语料级 BPE bottom
- 追加更大阈值 N 的版本对比 (当前 N=100)
