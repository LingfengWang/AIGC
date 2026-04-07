# Diffusion 模型学习路线 - 关键论文汇总

**整理日期**: 2026-04-06  
**适合人群**: 新手入门 Diffusion 模型  
**预计学习周期**: 2-3 个月

---

## 📚 学习路线图

```
第 1 阶段 (2 周)              第 2 阶段 (3 周)              第 3 阶段 (3 周)
基础理论                     核心突破                     进阶应用
    │                           │                           │
    ▼                           ▼                           ▼
• VAE (2013)                • DDPM (2020)               • LDM/SD (2021)
• 数学基础                   • DDIM (2020)               • ControlNet (2023)
• 概率论基础                 • Score SDE (2020)          • DiT (2022)
                                                        • LoRA (2021)
                                                       
第 4 阶段 (2 周)              第 5 阶段 (2 周)
现代架构                     前沿发展
    │                           │
    ▼                           ▼
• MMDiT (2024)              • Rectified Flow (2022)
• FLUX 架构                   • Flow Matching (2022)
• 流模型                      • EDM/EDM2
```

---

## 📖 阶段一：基础理论 (2 周)

### 1. VAE (2013) - 变分自编码器

**论文**: Auto-Encoding Variational Bayes  
**作者**: Kingma & Welling  
**链接**: https://arxiv.org/abs/1312.6114  
**状态**: ✅ 已下载 (`05_VAE_2013_Kingma.pdf`)

**为什么重要**:
- Diffusion 可以看作"多个 VAE 串联"
- 理解隐空间 (latent space) 概念
- 重参数化技巧是基础

**核心概念**:
- 编码器 - 解码器结构
- 隐变量分布
- ELBO 损失函数
- 重参数化技巧

**学习重点**:
- [ ] 理解 VAE 的生成过程
- [ ] 掌握重参数化技巧
- [ ] 理解 KL 散度的作用

**学习资源**:
- [3Blue1Brown VAE 视频](https://www.youtube.com/watch?v=9zKuYvjFFS8)
- [李宏毅 VAE 教程](https://www.youtube.com/watch?v=uL8OYwvZ5Lw)

**预计用时**: 4-6 小时

---

### 2. 数学基础补充

**需要掌握的数学知识**:

| 主题 | 内容 | 资源 |
|------|------|------|
| 概率论 | 贝叶斯定理、高斯分布 | 3Blue1Brown |
| 信息论 | 熵、KL 散度 | 《深度学习》第 3 章 |
| 微积分 | 梯度、链式法则 | 微积分课程 |
| 线性代数 | 矩阵运算、特征值 | 线性代数课程 |

**预计用时**: 8-10 小时

---

## 📖 阶段二：核心突破 (3 周)

### 3. DDPM (2020) - 扩散模型奠基

**论文**: Denoising Diffusion Probabilistic Models  
**作者**: Jonathan Ho et al. (UC Berkeley)  
**链接**: https://arxiv.org/abs/2006.11239  
**状态**: ✅ 已下载 (`01_DDPM_2020_Ho.pdf`)

**为什么重要**:
- Diffusion 领域的"ImageNet 时刻"
- 证明 Diffusion 可以媲美 GAN
- 训练稳定，不会 mode collapse

**核心创新**:
1. 前向扩散过程 (加噪)
2. 反向去噪过程 (生成)
3. **预测噪声而非原图** (关键洞察!)

**核心公式**:
```
前向过程：q(x_t | x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)
任意时刻：q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t)I)
损失函数：L = E[||ε - ε_θ(x_t, t)||²]
```

**学习重点**:
- [ ] 理解为什么预测噪声
- [ ] 掌握噪声调度 β_t
- [ ] 理解训练和采样过程

**代码实践**:
```python
# 使用 Diffusers 体验 DDPM
from diffusers import DDPMPipeline

pipeline = DDPMPipeline.from_pretrained("google/ddpm-cat-256")
image = pipeline().images[0]
```

**预计用时**: 10-12 小时

---

### 4. DDIM (2020) - 加速采样

**论文**: Denoising Diffusion Implicit Models  
**作者**: Song et al.  
**链接**: https://arxiv.org/abs/2010.02502  
**状态**: ✅ 已下载 (`02_DDIM_2020_Song.pdf`)

**为什么重要**:
- 解除马尔可夫链限制
- 从 1000 步加速到 50-100 步
- 确定性采样 (可复现)

**核心创新**:
- 非马尔可夫扩散过程
- 跳步采样 (skip steps)
- 相同起点 → 相同结果

**学习重点**:
- [ ] 理解 DDIM 与 DDPM 的区别
- [ ] 掌握跳步采样原理
- [ ] 理解确定性 vs 随机性

**预计用时**: 6-8 小时

---

### 5. Score SDE (2020) - 分数匹配

**论文**: Score-Based Generative Modeling through Stochastic Differential Equations  
**作者**: Song Yang et al.  
**链接**: https://arxiv.org/abs/2011.13456  
**状态**: ✅ 已下载 (`16_ScoreSDE_2020_Song.pdf`)

**为什么重要**:
- 统一了扩散模型和分数匹配
- SDE 理论框架
- 后续很多工作的基础

**核心概念**:
- 分数匹配 (Score Matching)
- 随机微分方程 (SDE)
- 概率流 ODE

**学习重点**:
- [ ] 理解分数函数 ∇log p(x)
- [ ] 了解 SDE 基础
- [ ] 理解与 DDPM 的联系

**预计用时**: 8-10 小时

---

### 6. Diffusion Beat GANs (2021) - 性能证明

**论文**: Diffusion Models Beat GANs on Image Synthesis  
**作者**: Prafulla Dhariwal & Alex Nichol (OpenAI)  
**链接**: https://arxiv.org/abs/2105.05233  
**状态**: ✅ 已下载 (`17_DiffusionBeatGANs_2021_Dhariwal.pdf`)

**为什么重要**:
- 正式宣布 Diffusion 超越 GAN
- 详细的对比实验
- 推动行业转向 Diffusion

**核心贡献**:
- 系统性对比 Diffusion vs GAN
- 改进的架构设计
- FID 等指标全面领先

**学习重点**:
- [ ] 理解 FID 等评估指标
- [ ] 了解 Diffusion 的优势
- [ ] 对比 GAN 的优缺点

**预计用时**: 4-6 小时

---

## 📖 阶段三：进阶应用 (3 周)

### 7. LDM / Stable Diffusion (2021) - 商业化起点

**论文**: High-Resolution Image Synthesis with Latent Diffusion Models  
**作者**: Robin Rombach et al. (CompVis)  
**链接**: https://arxiv.org/abs/2112.10752  
**状态**: ✅ 已下载 (`03_LDM_2021_Rombach.pdf`)

**为什么重要**:
- **引入隐空间，计算效率提升 48 倍**
- 催生 Stable Diffusion 系列
- AIGC 商业化起点

**核心创新**:
1. 两阶段训练 (VAE + Diffusion)
2. 隐空间去噪 (而非像素空间)
3. Cross-Attention 条件机制

**架构**:
```
图像 → VAE Encoder → 隐变量 → Diffusion UNet → 隐变量 → VAE Decoder → 图像
              (64×64×4)                          (64×64×4)
```

**学习重点**:
- [ ] 理解隐空间压缩
- [ ] 掌握 Cross-Attention
- [ ] 了解 SD 生态系统

**代码实践**:
```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
image = pipe("a cute cat").images[0]
```

**预计用时**: 10-12 小时

---

### 8. ControlNet (2023) - 条件控制

**论文**: Adding Conditional Control to Text-to-Image Diffusion Models  
**作者**: Lvmin Zhang et al.  
**链接**: https://arxiv.org/abs/2302.05543  
**状态**: ✅ 已下载 (`06_ControlNet_2023_Zhang.pdf`)

**为什么重要**:
- 实现精确控制 (边缘/深度/姿态等)
- 工业级应用关键
- 生态爆发 (数千个 ControlNet 模型)

**核心创新**:
1. 锁定 + 复制策略
2. Zero Convolution (初始化为 0)
3. 通用条件控制框架

**控制类型**:
- Canny (边缘)
- Depth (深度)
- Pose (姿态)
- Scribble (涂鸦)
- ...

**学习重点**:
- [ ] 理解 Zero Convolution
- [ ] 掌握各种控制类型
- [ ] 实践 ControlNet 应用

**预计用时**: 8-10 小时

---

### 9. DiT (2022) - Transformer 架构

**论文**: Scalable Diffusion Models with Transformers  
**作者**: Peebles & Xie (MIT)  
**链接**: https://arxiv.org/abs/2212.09748  
**状态**: ✅ 已下载 (`04_DiT_2022_Peebles.pdf`)

**为什么重要**:
- **Transformer 取代 UNet**
- 可扩展性更好
- SD3/FLUX 都采用此架构

**核心创新**:
- 纯 Transformer 架构
- Patch 化处理图像
- 简单的缩放规律

**学习重点**:
- [ ] 理解 Transformer 在 Diffusion 中的应用
- [ ] 对比 UNet vs DiT
- [ ] 了解注意力机制

**预计用时**: 8-10 小时

---

### 10. LoRA (2021) - 高效微调

**论文**: LoRA: Low-Rank Adaptation of Large Language Models  
**作者**: Hu et al. (Microsoft)  
**链接**: https://arxiv.org/abs/2106.09685  
**状态**: ✅ 已下载 (`07_LoRA_2021_Hu.pdf`)

**为什么重要**:
- 高效微调 (参数量减少 1000 倍)
- 2023 年应用于 SD
- 催生数十万 LoRA 模型

**核心创新**:
- 低秩分解：ΔW = B·A
- 冻结原模型，只训练 LoRA
- 推理时无额外开销

**学习重点**:
- [ ] 理解低秩分解
- [ ] 掌握 LoRA 训练
- [ ] 实践风格微调

**预计用时**: 6-8 小时

---

## 📖 阶段四：现代架构 (2 周)

### 11. MMDiT (2024) - SD3 架构

**论文**: Scaling Rectified Flow Transformers for High-Resolution Image Synthesis  
**作者**: Esser et al. (Stability AI)  
**链接**: https://arxiv.org/abs/2403.03206  
**状态**: ✅ 已下载 (`11_MMDiT_2024_Esser.pdf`)

**为什么重要**:
- SD3 的核心架构
- 多模态 DiT
- 文本显式加入模型

**核心创新**:
- 双塔架构 (图像塔 + 文本塔)
- 联合注意力
- 流匹配 + DiT

**学习重点**:
- [ ] 理解多模态融合
- [ ] 掌握 MMDiT 架构
- [ ] 对比 SDXL vs SD3

**预计用时**: 6-8 小时

---

## 📖 阶段五：前沿发展 (2 周)

### 12. Rectified Flow (2022) - 流模型

**论文**: Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow  
**作者**: Liu et al.  
**链接**: https://arxiv.org/abs/2209.03003  
**状态**: ✅ 已下载 (`08_RectifiedFlow_2022_Liu.pdf`)

**为什么重要**:
- **直线路径，采样更快**
- SD3/FLUX 采用
- 逐渐取代传统 Diffusion

**核心思想**:
- 学习从噪声到数据的直线路径
- ODE 形式，采样简单
- 通常 10-50 步即可

**学习重点**:
- [ ] 理解流模型 vs 扩散模型
- [ ] 掌握 ODE 采样
- [ ] 了解优势与局限

**预计用时**: 6-8 小时

---

### 13. Flow Matching (2022) - 流匹配

**论文**: Flow Matching for Generative Modeling  
**作者**: Lipman et al.  
**链接**: https://arxiv.org/abs/2210.02747  
**状态**: ✅ 已下载 (`09_FlowMatching_2022_Lipman.pdf`)

**为什么重要**:
- 流匹配理论框架
- 更稳定的训练
- 成为新标准

**学习重点**:
- [ ] 理解 Flow Matching 原理
- [ ] 对比 Rectified Flow
- [ ] 了解数学基础

**预计用时**: 8-10 小时

---

### 14. EDM (2022) - 设计空间分析

**论文**: Elucidating the Design Space of Diffusion-Based Generative Models  
**作者**: Karras et al. (NVIDIA)  
**链接**: https://arxiv.org/abs/2206.00364  
**状态**: ✅ 已下载 (`18_EDM_2022_Karras.pdf`)

**为什么重要**:
- 系统性对比各类 Sampler
- 学术参考标准
- 理解采样器设计

**学习重点**:
- [ ] 了解各类 Sampler
- [ ] 理解设计选择
- [ ] 参考实验结果

**预计用时**: 4-6 小时

---

## 📊 完整论文清单

| 序号 | 论文 | 年份 | 状态 | 优先级 |
|------|------|------|------|--------|
| 1 | VAE | 2013 | ✅ | 🔴 必读 |
| 2 | DDPM | 2020 | ✅ | 🔴 必读 |
| 3 | DDIM | 2020 | ✅ | 🔴 必读 |
| 4 | Score SDE | 2020 | ✅ | 🟡 进阶 |
| 5 | Diffusion Beat GANs | 2021 | ✅ | 🟡 进阶 |
| 6 | LDM/Stable Diffusion | 2021 | ✅ | 🔴 必读 |
| 7 | ControlNet | 2023 | ✅ | 🟡 进阶 |
| 8 | DiT | 2022 | ✅ | 🔴 必读 |
| 9 | LoRA | 2021 | ✅ | 🟡 进阶 |
| 10 | MMDiT | 2024 | ✅ | 🟢 前沿 |
| 11 | Rectified Flow | 2022 | ✅ | 🟢 前沿 |
| 12 | Flow Matching | 2022 | ✅ | 🟢 前沿 |
| 13 | EDM | 2022 | ✅ | 🟡 进阶 |

**总计**: 13 篇核心论文，全部已下载 ✅

---

## 🎯 学习建议

### 新手路线 (8 周)

| 周次 | 内容 | 论文 | 实践 |
|------|------|------|------|
| 1-2 | 基础理论 | VAE | VAE 代码实现 |
| 3-4 | 核心突破 | DDPM, DDIM | DDPM 采样 |
| 5-6 | 进阶应用 | LDM, ControlNet | SD 文生图 |
| 7-8 | 现代架构 | DiT, MMDiT | 模型微调 |

### 快速路线 (4 周)

| 周次 | 内容 | 论文 |
|------|------|------|
| 1 | 基础 | VAE, DDPM |
| 2 | 核心 | DDIM, LDM |
| 3 | 应用 | ControlNet, LoRA |
| 4 | 前沿 | DiT, Flow Matching |

---

## 📝 学习检查点

### 阶段一检查点
- [ ] 能解释 VAE 的生成过程
- [ ] 理解重参数化技巧
- [ ] 掌握 KL 散度的含义

### 阶段二检查点
- [ ] 能解释为什么预测噪声
- [ ] 理解 DDIM 如何加速
- [ ] 了解 Score Matching

### 阶段三检查点
- [ ] 理解隐空间压缩
- [ ] 掌握 ControlNet 原理
- [ ] 会训练 LoRA

### 阶段四检查点
- [ ] 理解 Transformer 架构
- [ ] 对比 UNet vs DiT

### 阶段五检查点
- [ ] 理解流模型原理
- [ ] 对比 Diffusion vs Flow

---

## 🔗 学习资源

### 视频教程
- [李宏毅生成式 AI 课程](https://www.youtube.com/playlist?list=PLJV_el3uVTsNxV_IG_5bI7FLQm--xvACh)
- [Diffusion 模型详解](https://www.youtube.com/watch?v=HoKDTa5jHvg)

### 代码资源
- [Diffusers 库](https://github.com/huggingface/diffusers)
- [DDPM 官方实现](https://github.com/hojonathanho/diffusion)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)

### 社区
- [HuggingFace](https://huggingface.co/)
- [Papers With Code](https://paperswithcode.com/)
- [Reddit r/StableDiffusion](https://www.reddit.com/r/StableDiffusion/)

---

## 💡 学习技巧

### ✅ 要做的事
1. **边读边写**: 每篇论文都要做笔记
2. **跑通代码**: 理论 + 实践结合
3. **加入社区**: 提问和分享
4. **定期回顾**: 每周复习所学内容

### ❌ 不要做的事
1. **不要只看不练**: 光读论文学不会
2. **不要追求完美**: 先理解大意
3. **不要孤军奋战**: 遇到问题及时求助
4. **不要贪多**: 一次专注一个主题

---

**祝你学习顺利！有问题随时查阅资料库。** 🚀

---

*最后更新*: 2026-04-06
