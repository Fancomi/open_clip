# Orthogonal SigLIP: 相关工作调研

*调研日期: 2026-05-12*

---

## 结论

**`|cos|` 作为负样本 loss 函数推向 cos=0 的做法，在 CLIP/SigLIP 多模态对比学习中是新颖的。** 没有找到直接先例。最接近的是 Lee et al. (2024) 在类增量学习中的相同几何思想，以及 SigLIP 理论分析表明 cos=0 是大 batch 下的最优解。

---

## 1. 高度相关（相同或极接近的思想）

### 1.1 Class Incremental Learning With Large Domain Shift
- **作者:** K. Lee, H. Kim, G. Choi, H. Jeon, N. Kwak
- **发表:** 2024, IEEE Access
- **核心思想:** 显式提出将负样本对推向正交而非对立。动机是类增量学习——当负样本推向 cos=-1 时，新类加入会挤占旧类空间。正交目标留出更多几何空间。
- **与我们的关系:** ★★★★★ 相同的几何直觉，但应用场景完全不同（分类的监督对比学习 vs 我们的多模态 SigLIP sigmoid loss）。差异：(a) 我们作用于跨模态图文对比，(b) 使用 `|cos|` 的对称惩罚实现，(c) 建立与模态鸿沟的联系。

### 1.2 Analysis of Using Sigmoid Loss for Contrastive Learning
- **作者:** C. Lee, J. Chang, J. Sohn
- **发表:** 2024, AISTATS
- **核心思想:** SigLIP sigmoid loss 的理论分析。证明**最优嵌入结构随温度参数变化，从 ETF（cos≈0）到反极配置（cos=-1）**。低温→反极，中温→ETF（cos=-1/(N-1)≈0 for large N）。
- **与我们的关系:** ★★★★ 这是我们方法的直接理论支撑。证明在大 batch 下 cos=0 已经是 SigLIP 的信息论最优。我们的 `|cos|` 可被视为"无论温度如何，直接 enforce ETF 几何"。

### 1.3 Global Minimizers of Sigmoid Contrastive Loss
- **作者:** K. Bangachev, G. Bresler, I. Noman et al.
- **发表:** 2026, NeurIPS
- **核心思想:** 刻画 SigLIP 全局极小值的结构。证明在最优解处，两个模态由超平面分离（正交于特定坐标方向）。
- **与我们的关系:** ★★★★ 证明正交分离在 SigLIP 优化中自然涌现。我们的方法可定位为"通过显式 loss 设计加速收敛到这一自然几何"。

### 1.4 On the Similarities of Embeddings in Contrastive Learning
- **作者:** C. Lee, S. Lim, K. Lee, J. Sohn
- **发表:** 2025, arXiv:2506.09781
- **核心思想:** 统一框架分析对比学习中的余弦相似度分布。提出辅助 loss 降低负样本相似度的方差（集中在 cos≈0 附近）。
- **与我们的关系:** ★★★ 方差约减 loss 与 `|cos|` 哲学相似：都试图让负样本集中在 cos=0 而非分散到极端。

---

## 2. 中度相关（相关几何/理论洞察）

### 2.1 It's Not a Modality Gap: Characterizing and Addressing the Contrastive Gap
- **作者:** A. Fahim, A. Murphy, A. Fyshe
- **发表:** 2024, arXiv:2405.18570
- **核心思想:** 将模态鸿沟归因于低均匀性。通过 alignment+uniformity 项缩小 gap。
- **与我们的关系:** ★★★ 负样本推向 cos=0 应改善均匀性，与该文诊断一致。但他们的解决方案不同（加辅助 uniformity loss，非修改负样本目标）。

### 2.2 Explaining and Mitigating the Modality Gap in Contrastive Multimodal Learning
- **作者:** C. Yaras, S. Chen, P. Wang, Q. Qu
- **发表:** 2024, arXiv:2412.07909
- **核心思想:** 不匹配数据对和可学习温度导致模态鸿沟。两个模态的主成分方向互相正交。
- **与我们的关系:** ★★★ 理论上支持正交结构自然存在；我们的方法是显式 enforce 这一结构。

### 2.3 Is the Modality Gap a Bug or a Feature? A Robustness Perspective
- **作者:** R. Chowers, O. Naparstek, U. Barzelay et al.
- **发表:** 2026, arXiv:2603.29080
- **核心思想:** 证明最优 gap 方向正交于两个模态子空间。gap 与鲁棒性单调相关。
- **与我们的关系:** ★★★ 数学结果支持 cos=0 目标的几何直觉。

### 2.4 Two Effects, One Trigger: On the Modality Gap, Object Bias, and Information Imbalance
- **作者:** S. Schrodi, D.T. Hoffmann, M. Argus et al.
- **发表:** 2025, ICLR
- **核心思想:** gap 方向正交于 image 和 text 嵌入的 span。CLIP loss 的排斥力仅跨模态作用。
- **与我们的关系:** ★★★ 观察到的正交结构与我们的设计方向一致。

### 2.5 OrCo: Towards Better Generalization via Orthogonality and Contrast (FSCIL)
- **作者:** N. Ahmed, A. Kukleva, B. Schiele
- **发表:** 2024, CVPR
- **核心思想:** 正交性 + 对比学习用于 few-shot 类增量学习。正交 loss 使类原型互相正交。
- **与我们的关系:** ★★ 用正交作为类分离原则，但作用于类原型而非 loss 中的负样本目标。

### 2.6 TuneCLIP: Breaking the Limits of Open-Weight CLIP
- **作者:** A. Mehta, X. Wei, X. Chen, T. Yang
- **发表:** 2026, arXiv:2601.09859
- **核心思想:** 新对比 loss 只在相似度差距超过阈值时惩罚正负样本对。
- **与我们的关系:** ★★ 动机相似（减轻负样本过度惩罚），但实现通过 threshold/margin 而非 cos=0 目标。

---

## 3. 低度相关（名称/概念重叠）

| 工作 | 年份 | 关系 |
|------|------|------|
| COrAL (Orthogonalized Multimodal CL) | 2026 | 正交用于分离 shared/unique 信息子空间，非负样本目标 |
| Decipher the Modality Gap (Yi et al.) | 2025 | 理论框架证明维度坍缩是 gap 根因 |
| Double-Ellipsoid Geometry of CLIP (Levi) | 2024 | CLIP 几何结构分析 |
| λ-Orthogonality Regularization | 2026 | 向后兼容表征的正交约束 |

---

## 4. 经典理论基础

| 工作 | 核心贡献 | 与我们的关系 |
|------|----------|-------------|
| Mind the Gap (Liang, NeurIPS 2022) | 模态鸿沟的诊断和表征 | 我们解决他们指出的问题 |
| Neural Collapse (Papyan, 2020) | ETF 几何是多类分类的最优解 | K>>1 时 ETF 类间 cos→0 |
| Alignment & Uniformity (Wang & Isola, ICML 2020) | 高维均匀分布 pairwise cos=0 | 正交 = 高维均匀 |
| Circle Loss (Sun, CVPR 2020) | 可配置负样本目标（含 0） | 最接近的 loss 设计先例 |
| SigLIP (Zhai, ICCV 2023) | Sigmoid pairwise loss | 我们的基础方法 |

---

## 5. 新颖性评估

| 维度 | 已有工作? | 我们的贡献 |
|------|----------|-----------|
| 负样本推向 cos=0 的几何思想 | Lee 2024 (类增量学习) | **首次应用于多模态 VL 对比** |
| `|cos|` 对称惩罚实现 | **无直接先例** | **新颖 loss 设计** |
| 解决模态鸿沟 | Fahim 2024 (uniformity), 多篇诊断 | **从 loss 目标角度根治** |
| SigLIP 最优性理论连接 | Lee 2024 AISTATS (理论分析) | **实践验证 + 简洁实现** |
| 高维球面 cos=0 = 最大熵 | Wang & Isola 2020 (隐含) | **显式利用** |

---

## 6. 定位建议

我们的方法可定位为：

> **首次在多模态对比学习（SigLIP）中显式 enforce 负样本正交的训练目标。** 理论上，这是 SigLIP 在大 batch 下的信息论最优解（Lee 2024 AISTATS）；实践上，通过一行代码改动（`|cos|`）消除了模态鸿沟的结构性来源，实现双向检索指标无 trade-off 提升。

关键引用：Lee 2024 (IEEE Access) 作为最接近先例（区别在场景和实现），Lee 2024 (AISTATS) 作为理论支撑，Liang 2022 作为问题背景。

---

*文档版本: 2026-05-12 v1*
