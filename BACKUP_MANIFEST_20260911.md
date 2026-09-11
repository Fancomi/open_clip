# 数据与产物备份清单（2026-09-11）

> 生成时间：2026-09-11 14:30 (CST)
> 生成目的：停止全部工作进程后，梳理需要备份的数据与产物（内容 + 大小）
> 工作区根目录：`/root/paddlejob/workspace/env_run/penghaotian`

## 0. 状态快照

- ✅ 所有计算工作进程已停止（sports360 批注任务、gpu_guard/infer.py、serve_review）
- ✅ GPU 已全部释放（8 卡 0% 利用率，显存 1 MiB）
- ✅ 代码仓库已推送（见 §2）

---

## 1. 数据与产物清单（按大小排序）

### 1.1 训练/实验产物 —— 最重要，需优先备份

| 路径 | 大小 | 内容 |
|---|---|---|
| `vision_encoder/open_clip/logs/` | **2.2T** | 121 个 visreg 实验运行；`*.pt` checkpoint 2.1T（949 个）、`*.npz` probe 47G（948 个）、jsonl 结果 113 个 |
| 　├ 最近活跃（近 3 天，3 个运行，各 23G） | 69G | `visreg_gemma_pcmregw0.2p0-roi2mil...0909`、`...CROP00_0909`、`...CROP06_0910` |
| 　├ 单运行最大 | 42G | `visreg_dual_E_0808_1601` |
| 　├ visreg_gemma* 全部 | ~1.7T | Gemma 投影/正则系列实验 |
| 　└ 其余（visreg_cc3m/n/m/x/w/q/sweep/imnzs 等） | ~0.5T | 早期消融与验证 |

> ⚠️ 磁盘已用 93%（3.2T/3.5T），logs 占 2.2T。备份时建议按"最近 3 天 + 已判实验结论"分优先级；`*.pt` 为 2.1T 主体。

### 1.2 数据集（datas/ 606G）

| 路径 | 大小 | 内容 |
|---|---|---|
| `datas/cc3m-tsv` | 292G | CC3M tsv 元数据 |
| `datas/cc3m-wds` | 262G | CC3M webdataset 图像 |
| `datas/coco` | 20G | COCO 数据集 |
| `datas/docci` | 15G | DocCI 数据集 |
| `datas/cc3m_region` | 12G | CC3M 区域标注 |
| `datas/voc2012` | 3.8G | PASCAL VOC 2012 |
| `datas/ade20k` | 1.9G | ADE20K |
| `datas/imagenet-val` | 202M | ImageNet 验证集 |
| `datas/urban1k` | 160M | Urban1k |

### 1.3 模型权重（models/ 189G + sport_project/weights 9.6G）

| 路径 | 大小 | 内容 |
|---|---|---|
| `models/cogvlm` | 59G | CogVLM |
| `models/gemma-4-26B-A4B-it` | 49G | Gemma-4 26B |
| `models/Qwen3.6-35B-A3B-FP8` | 35G | Qwen3.6 35B FP8 |
| `models/timm` | 6.9G | timm 权重 |
| `models/DAM-3B-Video` / `DAM-3B` | 6.7G×2 | DAM 3B |
| `models/sam3` + `sam3.zip` | 6.5G+6.0G | SAM3 |
| `models/DFN5B-CLIP-ViT-H-14-378` | 3.7G | DFN5B CLIP |
| `models/sam-vit-huge` / `sam1` | 2.4G×2 | SAM |
| `models/semantic-sam` / `Florence-2-large` / `sam2` 等 | <2G 各 | 其余权重 |
| `sport_project/weights/stroke_event` | 7.3G | 击球事件模型 |
| `sport_project/weights/player` | 1.1G | 球员检测/姿态 |
| `sport_project/weights/shuttle/court_reference` 等 | ~1.2G | 球/场地模型 |

> 注：多数为公开可再下载权重，如空间紧张可仅备份 `sport_project/weights`（含自定义训练结果）与定制模型。

### 1.4 sports360 项目（161G）

| 路径 | 大小 | 内容 |
|---|---|---|
| `sport_project/sports360/sports360_batminton` | 148G | 原始羽毛球视频（多届赛事） |
| `sport_project/sports360/runs/batch/sports360_1000` | 14G | **批注产物**：1523 段（1512 完成），每段 `observations.jsonl` + `clip.mp4` + `batch_manifest.json` |

> `sports360_batminton` 是同步自远端（rsync `ral@10.109.83.30`）的输入视频，远端有源；真正需要备份的是 `runs/batch/sports360_1000`（14G 批注结果）。

### 1.5 标注产物（cc3m-annotate/out 11G）

| 路径 | 大小 | 内容 |
|---|---|---|
| `vision_encoder/cc3m-annotate/out/ground` | 4.5G | ground 标注（58 个 jsonl） |
| `vision_encoder/cc3m-annotate/out/caption` | 4.4G | caption 标注 |
| `vision_encoder/cc3m-annotate/out/clean` | 2.0G | 清洗后结果 |
| `vision_encoder/cc3m-annotate/out/ab` / `verify_clean.jsonl` | 165M/1.8M | 消融/校验 |
| `vision_encoder/cc3m-annotate/logs` | 38M | 清洗日志 |

### 1.6 其他备份点

| 路径 | 大小 | 内容 |
|---|---|---|
| `vision_encoder/BAK/cc3m_annotate` | 609M | 历史备份 |
| `vision_encoder/open_clip/core.3266397` | 27G | ⚠️ core dump（崩溃转储，**不建议备份**，可删除） |
| `vision_encoder/open_clip/core.3868524` | 897M | ⚠️ core dump（同上） |
| `vision_encoder/open_clip/caption_rewrite` | <1M | caption 重写脚本+少量 outputs |
| `workspace/sglang_v0.5.12` / `DeepGEMM` | 5.4G/273M | 源码（已在 git） |
| `envs/`（11 个 Python 环境） | 71G | ⚠️ 可重建，**不建议备份**（除非需离线复现） |

---

## 2. 代码上传状态（git push）

| 仓库 | 分支 | 状态 |
|---|---|---|
| `vision_encoder/open_clip` | main | ✅ 已同步 origin/main（工作树干净） |
| `vision_encoder/describe-anything` | main | ✅ 已推送（1 个新提交 `24b95c9`） |
| `vision_encoder/FG-CLIP` | main | ✅ 已同步 |
| `vision_encoder/cc3m-annotate` | main | ✅ 已同步 |
| `vision_encoder/novic` / `visreg` | — | ✅ 已同步 |
| `vision_encoder/VL-SAM` | main | ⚠️ 本地已提交（`52d69b9` 诊断代码），**推送超时**——远端为上游 VDIGPKU/VL-SAM（非本人仓库，无写权限），提交已保存在本地 |
| `sport_project/sports360` | badminton_haotian_0910 | ⚠️ 远端 iCode 要求走**评审**流程，无法直接 push；6 个提交在本地（含备份清单刷新 `11d9d5e`） |
| `llm_infer/llm_train`、`sport_ontology`、`label_mocap`、`ultralytics`、`resume-claude-sessions` 等 | — | ✅ 已同步 |

---

## 3. 建议备份优先级

**P0（必须，先备份，约 2.3T）**
1. `open_clip/logs/` 中**最近 3 天活跃运行**（69G，含 CROP00/CROP06 与 pcmregw0.2p0 判结）—— 如需全量则整目录 2.2T
2. `sport_project/sports360/runs/batch/sports360_1000/`（14G 批注产物）
3. `vision_encoder/cc3m-annotate/out/`（11G 标注产物）
4. `sport_project/weights/`（9.6G 训练模型）

**P1（重要）**
5. `models/` 中自定义/稀有权重（Gemma-4、Qwen3.6、CogVLM 等，189G，若远端可下可跳过）
6. `datas/cc3m_region`（12G，可能是自建区域标注）

**P2（按需）**
7. 各公开数据集（datas/ 其余 594G，可从源头重取）
8. `vision_encoder/BAK/`（609M）
9. `sports360_batminton` 原始视频（148G，远端有源，通常跳过）

**不推荐备份**
- `envs/`（71G，可重建）、`core.*` 崩溃转储（28G，可删除）、`workspace/log*`（平台日志）

---

## 4. 已停止的工作

| 进程 | PID | 说明 | 状态 |
|---|---|---|---|
| sports360 批注任务 | 1357656-1358873（8 个 worker+tracker） | `runs/batch/sports360_1000` 批处理 | ✅ 已终止（SIGTERM 干净退出） |
| gpu_guard.sh | 511808 | GPU 守卫 | ✅ 已终止 |
| infer.py 0 / 2 | 1864244 / 2281914 | GPU 占用推理 | ✅ 已终止 |
| serve_review.py | 1531078 | 评审服务 | ✅ 已终止 |
| GPU | 全部 8 卡 | — | ✅ 0% 利用率，显存 1MiB |
