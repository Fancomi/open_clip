# open_clip 数据与产物备份清单（2026-09-11）

> 生成时间：2026-09-11 14:40 (CST)（第二次更新）
> 范围：**仅 open_clip 及相关内容**（vision_encoder/open_clip + 关联的 cc3m-annotate / caption_rewrite / 相关数据与模型）
> 工作区根目录：`/root/paddlejob/workspace/env_run/penghaotian`

## 0. 状态快照

- ✅ 所有计算工作进程已停止（sports360 批注任务、gpu_guard/infer.py、serve_review）
- ✅ GPU 已全部释放（8 卡 0% 利用率，显存 1 MiB）
- ✅ 代码仓库已推送（见 §3）
- ✅ **checkpoint 已全部删除**：`open_clip/logs/` 下 949 个 `*.pt`（2.1T）已删除
- 🟢 磁盘使用率已从 **93%（3.2T/3.5T）降至 33%（1.2T/3.5T）**，释放约 2.1T

---

## 1. 备份范围界定

只备份 **open_clip 及其直接关联内容**，即：

1. `vision_encoder/open_clip/` —— 主仓库（代码 + logs 实验产物 + core 转储）
2. `vision_encoder/cc3m-annotate/` —— open_clip 的 CC3M 标注配套仓库
3. `vision_encoder/open_clip/caption_rewrite/` —— caption 重写模块
4. 关联数据集（open_clip 训练用）：`datas/cc3m-*`（tsv/wds/region）
5. 关联模型权重：`models/` 中 open_clip 实验实际使用的（CLIP/timm/sam/Florence 等）

**不在本次范围**：sport_project/sports360、tools、llm_infer、VL-SAM、novic、describe-anything、FG-CLIP 等其他项目（用户指定只看 open_clip 及相关）。

---

## 2. 备份清单（open_clip 范围内）

### 2.1 open_clip 主仓库 —— 合计 74G

| 路径 | 大小 | 内容 |
|---|---|---|
| `open_clip/logs/` | **47G** | 122 个 visreg 实验运行目录；保留 `*.npz` probe 948 个（47G）+ jsonl 结果 113 个 + 日志。**checkpoint 已删** |
| `open_clip/core.3266397` | 27G | ⚠️ core dump（崩溃转储，**不建议备份**，建议删除） |
| `open_clip/core.3868524` | 815M | ⚠️ core dump（同上） |
| `open_clip/docs/` | 12M | 文档（已在 git） |
| `open_clip/src/` | 4.0M | 源码（已在 git） |
| `open_clip/analysis/` | 1.7M | 分析脚本/输出 |
| `open_clip/scripts/` | 824K | 脚本（已在 git） |
| `open_clip/caption_rewrite/` | 216K | caption 重写模块 |
| 其余（README/pyproject 等） | <1M | 仓库文件（已在 git） |

> **logs/ 明细（删除 checkpoint 后）**
> - 文件类型：`*.npz` 948 个（47G）、`*.log` 124 个、`*.txt` 123 个、`*.jsonl` 113 个
> - 最近活跃运行（近 3 天）：`visreg_gemma_pcmregw0.2p0-roi2mil...0909`、`...CROP00_0909`、`...CROP06_0910`（checkpoint 已删，仅剩 probe/结果）

### 2.2 关联仓库 cc3m-annotate —— 11G

| 路径 | 大小 | 内容 |
|---|---|---|
| `cc3m-annotate/out/ground` | 4.5G | ground 标注（58 个 jsonl） |
| `cc3m-annotate/out/caption` | 4.4G | caption 标注 |
| `cc3m-annotate/out/clean` | 2.0G | 清洗后结果 |
| `cc3m-annotate/out/ab` / `verify_clean.jsonl` | 165M/1.8M | 消融/校验 |
| `cc3m-annotate/logs` | 38M | 清洗日志 |
| 其余（src/scripts/docs） | <1M | 代码（已在 git） |

### 2.3 关联数据集（open_clip 训练用）—— ~566G

| 路径 | 大小 | 内容 |
|---|---|---|
| `datas/cc3m-tsv` | 292G | CC3M tsv 元数据 |
| `datas/cc3m-wds` | 262G | CC3M webdataset 图像 |
| `datas/cc3m_region` | 12G | CC3M 区域标注（自建，重要） |

> 其他数据集（coco/docci/voc2012/ade20k/imagenet-val）为公开可再下载，且与 open_clip 主线关联较弱，按需备份。

### 2.4 关联模型权重 —— 已使用的 ~13G（可选）

| 路径 | 大小 | 内容 |
|---|---|---|
| `models/timm` | 6.9G | timm 权重（CLIP 依赖） |
| `models/DFN5B-CLIP-ViT-H-14-378` | 3.7G | DFN5B CLIP |
| `models/sam3` / `sam2` / `sam-vit-huge` / `Florence-2-large` 等 | ~5G | 标注/分析用模型 |
| `models/hf_cache` | 9.7M | HF 缓存 |

> 均为公开可下载权重，如空间紧张可跳过。

---

## 3. 代码上传状态（git push）

| 仓库 | 分支 | 状态 |
|---|---|---|
| `vision_encoder/open_clip` | main | ✅ 已同步 origin/main（含备份清单 commit `ff37bd7`） |
| `vision_encoder/cc3m-annotate` | main | ✅ 已同步 |
| `vision_encoder/describe-anything` | main | ✅ 已推送（1 个新提交 `24b95c9`） |
| `vision_encoder/FG-CLIP` | main | ✅ 已同步 |
| `vision_encoder/VL-SAM` | main | ⚠️ 本地已提交（`52d69b9`），推送超时（远端为上游 VDIGPKU 无写权限）——不在本次 open_clip 范围，仅供参考 |

---

## 4. 建议备份优先级（open_clip 范围内）

**P0（必须）**
1. `open_clip/logs/`（47G，实验结论与 probe 数据）
2. `cc3m-annotate/out/`（11G，标注产物）
3. `datas/cc3m_region`（12G，自建区域标注）

**P1（重要）**
4. `datas/cc3m-tsv` + `cc3m-wds`（554G，open_clip 训练集，如远端可重取可跳过）
5. 关联模型权重（~13G）

**P2 / 不推荐**
- `open_clip` 的 2 个 core dump（28G）——建议直接删除
- 其他公开数据集（coco/docci 等）——按需
- **checkpoint 已删，无需备份**

---

## 5. 建议后续动作

- **删除 core dump**（28G）：`rm open_clip/core.3266397 open_clip/core.3868524`（如需保留崩溃现场请先归档）
- 如确认不再需要 probe 的 npz（47G），可再释放 47G（用户可随时告知）
- 磁盘当前 33%（2.4T 可用），空间充足
