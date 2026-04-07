# AIGC 图像生成技术发展脉络

**整理自**: 知乎专栏《"AI 图像生成模型"与"关键技术"迭代历史》  
**原文作者**: 修林  
**整理日期**: 2026-04-03  
**整理人**: [你的名字]

---

## 📜 第零章：重要图像生成技术发展历史总览

### 0.1 GAN 时代

**核心论文**:
- **VQ-VAE** (2017): Neural Discrete Representation Learning
- **VQ-GAN** (2020): Taming Transformers for High-Resolution Image Synthesis

> 这是前 Diffusion 时代非常经典的生图模型，经过了时间检验。

### 0.2 Diffusion 时代与 Flow 时代

#### 关键发展时间线

| 时间 | 论文/技术 | 核心贡献 | 重要性 |
|------|----------|---------|--------|
| **2013.12** | Auto-Encoding Variational Bayes (VAE) | 变分自编码器基础 | 🔴 理论基础 |
| **2015.03** | Deep Unsupervised Learning using Nonequilibrium Thermodynamics | 首次明确提出 Diffusion 概念 | 🟡 早期探索 |
| **2020.06** | **DDPM** - Denoising Diffusion Probabilistic Models | 加噪 - 去噪生成范式，预测噪声而非原图 | 🔴 **里程碑** |
| **2020.10** | **DDIM** - Denoising Diffusion Implicit Models | 解除马尔可夫链限制，支持跳步生成 (1000 步→几十步) | 🔴 推理加速 |
| **2020.11** | Score-Based Generative Modeling through SDE | SDE 系列 Diffusion 基础 | 🟡 理论扩展 |
| **2021.05** | Diffusion Models Beat GANs on Image Synthesis | OpenAI 证明 Diffusion 优于 GAN | 🔴 范式转换 |
| **2021.12** | **LDM** - High-Resolution Image Synthesis with Latent Diffusion Models | 引入隐空间，降低训练量，Stable Diffusion 起源 | 🔴 **商业化起点** |
| **2022.06** | **EDM** - Elucidating the Design Space of Diffusion-Based Generative Models | 系统对比各类 Sampler | 🟡 学术研究 |
| **2022.09** | **Rectified Flow** - Flow Straight and Fast | 流模型诞生 | 🔴 新技术路线 |
| **2022.10** | **Flow Matching** for Generative Modeling | 流匹配正式推出 | 🔴 新技术路线 |
| **2022.12** | **DiT** - Scalable Diffusion Models with Transformers | Transformer 取代 UNet | 🔴 架构革新 |
| **2023.02** | **ControlNet** - Adding Conditional Control | 控制生成范式 | 🔴 工业应用 |
| **2023.01** | **LoRA** - Low-Rank Adaptation (应用于 SD) | 高效微调技术 | 🔴 生态爆发 |
| **2024.03** | **MMDiT** - Scaling Rectified Flow Transformers | SD3 架构，文本显式加入 | 🔴 主流框架 |

#### 前沿探索 (科研方向)

| 方向 | 论文 | 时间 | 核心思想 |
|------|------|------|---------|
| **编码器改进** | VAVAE (CVPR 2025 满分论文) | 2025.01 | 用 DINOv2/MAE 改进隐空间分布 |
| **编码器改进** | RAE - Diffusion Transformers with Representation Autoencoders | 2025.10 | 谢赛宁团队 |
| **无 VAE** | There is No VAE | 2025.10 | 端到端像素空间生成 |
| **像素空间** | PixelDiT | 2025.11 | 直接在像素空间训练 |
| **像素空间** | One-step Latent-free Image Generation with Pixel Mean Flows (何恺民) | 2026.01 | 无需隐空间 |
| **预测目标** | JiT - Back to Basics: Let Denoising Generative Models Denoise (何恺明) | 2025.11 | 反思预测 noise vs 原图 |
| **流模型改进** | Mean Flows for One-step Generative Modeling (何恺民) | 2025.05 | 一步生成 |

### 0.3 自回归生成技术探索

| 论文 | 时间 | 机构 | 贡献 |
|------|------|------|------|
| **VAR** | 2024 | 字节跳动 | NIPS 2024 Best Paper，自回归生成图像 |
| **NextFlow** | 2026.01 | - | VAR 衍生，探索自回归能力 |
| **GLM-Image** | 2026.01 | 智谱 AI | 开源自回归图像模型 |

---

## 📊 第一章：Stable Diffusion 发展史

### SD 版本演进

| 版本 | 时间 | 架构 | 文本编码器 | 参数量 | 分辨率 | 评价 |
|------|------|------|-----------|--------|--------|------|
| **v1.x** | 2022.08 | LDM (UNet) | CLIP ViT | ~0.85B | 512 | 开源奠基 |
| **v1.5** | 2022.10 | LDM | CLIP ViT-L | ~0.85B | 512 | 🔴 **生态之王** |
| **v2.x** | 2022.11 | UNet + OpenCLIP | OpenCLIP ViT-H | ~0.86B | 512/768 | 🟡 过度过滤 |
| **SDXL** | 2023.07 | Bigger UNet/Transformer | 双编码器 | ~2.6B | 1024 | 🔴 开源反击 |
| **SD3 Medium** | 2024.06 | MMDiT + 流模型 | T5/CLIP/OpenCLIP | ~2B | 1024+ | 🟡 翻车 |
| **SD3.5 Large** | 2024.10 | MMDiT | T5/CLIP/OpenCLIP | ~8B | 1M+ | 🟢 修复版 |

### 关键事件

1. **2022.08**: SD v1.4 发布，生成速度从"几十分钟"压缩到"几秒钟"
2. **2022.10**: SD v1.5 发布 (Runway 与 Stability AI 版权纠纷)
3. **2023.07**: SDXL 发布，开源模型首次与闭源模型竞争
4. **2024.06**: SD3 Medium 翻车 (肢体崩坏 + 苛刻许可证)
5. **2024.10**: SD3.5 发布，修复 Bug + 修改协议对标 FLUX

### 核心论文

```
[1] Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models" (CVPR 2022)
    https://arxiv.org/abs/2112.10752
    
[2] Podell et al. "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis" (2023)
    https://arxiv.org/abs/2307.01952
    
[3] Esser et al. "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (2024)
    https://arxiv.org/abs/2403.03206
```

---

## 🌊 第二章：FLUX 发展史

### 背景

- **公司**: Black Forest Labs (原 SD 核心团队 Robin Rombach 等离职创立)
- **融资**: 3100 万美元种子轮 (a16z 领投)
- **定位**: "真正的 SD3"

### FLUX.1 系列 (2024.08)

| 模型 | 参数量 | 架构 | 特点 | 协议 |
|------|--------|------|------|------|
| **FLUX.1 [schnell]** | 12B | Flow Matching + DiT | 4 步生成，速度之王 | Apache 2.0 |
| **FLUX.1 [dev]** | 12B | Flow Matching + DiT | 画质与遵循度，手指/文字完美 | 非商用开源 |

**核心优势**:
- 手指基本完美
- 文字生成准确 (能在图里写对英文单词)
- 提示词理解力极强

### FLUX.2 系列 (2025.11)

| 模型 | 参数量 | 特点 | 显存需求 |
|------|--------|------|---------|
| **FLUX.2 [dev]** | 32B | 原生 4MP 画质，融合 VLM (Mistral-3 24B) | 48H+ (需量化) |
| **FLUX.2 [pro]** | 闭源 | 最先进质量 | API |
| **FLUX.2 [flex]** | 闭源 | 可自由控制参数 | API |
| **FLUX.2 [klein]** | 4B/9B | 轻量化，8GB 显存可运行 | 消费级 |
| **FLUX.2 [MAX]** | 闭源 | 最强，支持 Grounding Search | API |

### 核心技术突破

1. **VLM + Transformer 融合**: 直接缝合 24B Mistral-3，理解能力爆表
2. **多图参考 (Multi-Reference)**: 原生支持 10 张参考图，角色一致性突破
3. **原生 4MP 画质**: 2048x2048+，几乎不需要 Upscaler

### FLUX 相关论文

```
[1] "FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing" (2025.06)
[2] "Demystifying Flux Architecture" (2025.07)
[3] "1.58-bit FLUX" (字节跳动，2024.11)
[4] "InfiniteYou: Flexible Photo Recrafting While Preserving Your Identity" (2025.07)
```

### 核心论文

```
[1] Labs et al. "FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space" (2025)
    https://bfl.ai/blog/flux-1-kontext
```

---

## 🇨🇳 第三章：东方力量 (中国公司)

### 3.1 阿里巴巴

#### (1) Qwen-Image 系列

| 模型 | 时间 | 参数量 | 特点 |
|------|------|--------|------|
| **Qwen-Image** | 2025.08 | 20B | 中英文本渲染最强，9 个基准测试第一 |
| **Qwen-Image-Edit** | 2025.08 | 20B | 图像编辑，文字精准编辑 |
| **Qwen-Image-Layer** | 2025.12 | - | 端到端图层分解，固有可编辑性 |
| **Qwen-Image-2** | 2026.02 | - | 闭源，1k token 指令，2k 分辨率 |

**核心亮点**:
- 中英文本渲染商用水准
- 可做 PPT 图像
- 通用图像风格

#### (2) Wan (通义万相) 系列

| 模型 | 时间 | 能力 | 状态 |
|------|------|------|------|
| **Wan 2.1** | 2025.01 | 文生视频、图生视频 | ✅ 开源 |
| **Wan 2.1-flf2v** | 2025.04 | 首尾帧过渡 | ✅ 开源 |
| **Wan 2.1-VACE** | 2025.05 | 视频编辑 | ✅ 开源 |
| **Wan 2.2** | 2025.07 | MoE 架构，电影级美学 | ✅ 开源 |
| **Wan 2.2-S2V** | 2025.08 | 图片 + 音频→视频 | ✅ 开源 |
| **Wan 2.5 Preview** | 2025.09 | 音画同步 | ❌ 闭源 |
| **Wan 2.6** | 2025.12 | 全面升级 | ❌ 闭源 |

#### (3) Z-Image 系列

| 模型 | 时间 | 特点 |
|------|------|------|
| **Z-Image-Turbo** | 2025.11 | 8 步推理，16GB 显卡亚秒级 |
| **Z-Image-Base** | 2025.11 | 非蒸馏基础版 |
| **Z-Image-Edit** | 2025.11 | 自然语言精准编辑 |

### 3.2 腾讯 (混元 Hunyuan)

| 模型 | 时间 | 参数量 | 特点 |
|------|------|--------|------|
| **Hunyuan-DiT** | 2024.05 | 1.5B | 首个中文原生 DiT |
| **Hunyuan Image 2.0** | 2025.05 | - | 毫秒级实时生图 |
| **HunyuanImage-2.1** | 2025.09 | - | 2K 高分辨率 |
| **HunyuanImage-3.0** | 2025.09 | 80B (MoE) | 全球最大开源，自回归框架 |
| **HunyuanImage-3.0-Instruct** | 2026.01 | - | 图生图能力 |

### 3.3 其他中国公司

| 公司 | 模型 | 时间 | 特点 |
|------|------|------|------|
| **美团** | LongCat | 2025 下半年 | 知名度较低 |
| **智谱 AI** | GLM-Image | 2026.01 | 自回归开源 |
| **深度求索** | - | - | - |

---

## 🎯 第四章：必须掌握的核心论文清单

### 🔴 必读经典 (基础)

1. **VAE** (2013)
   - Kingma & Welling. "Auto-Encoding Variational Bayes"
   - https://arxiv.org/abs/1312.6114

2. **DDPM** (2020)
   - Ho et al. "Denoising Diffusion Probabilistic Models"
   - https://arxiv.org/abs/2006.11239
   - **重要性**: Diffusion 奠基之作

3. **DDIM** (2020)
   - Song et al. "Denoising Diffusion Implicit Models"
   - https://arxiv.org/abs/2010.02502
   - **重要性**: 推理加速关键

4. **LDM/Stable Diffusion** (2021)
   - Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models"
   - https://arxiv.org/abs/2112.10752
   - **重要性**: 商业化起点

5. **DiT** (2022)
   - Peebles & Xie. "Scalable Diffusion Models with Transformers"
   - https://arxiv.org/abs/2212.09748
   - **重要性**: 架构革新

### 🟡 进阶必读 (应用)

6. **ControlNet** (2023)
   - Zhang et al. "Adding Conditional Control to Text-to-Image Diffusion Models"
   - https://arxiv.org/abs/2302.05543
   - **重要性**: 工业应用关键

7. **LoRA** (2021)
   - Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models"
   - https://arxiv.org/abs/2106.09685
   - **重要性**: 高效微调

8. **Rectified Flow** (2022)
   - Liu et al. "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"
   - https://arxiv.org/abs/2209.03003

9. **Flow Matching** (2022)
   - Lipman et al. "Flow Matching for Generative Modeling"
   - https://arxiv.org/abs/2210.02747

10. **SDXL** (2023)
    - Podell et al. "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis"
    - https://arxiv.org/abs/2307.01952

### 🟢 前沿探索 (科研)

11. **MMDiT** (2024)
    - Esser et al. "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"
    - https://arxiv.org/abs/2403.03206

12. **VAR** (2024)
    - "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"
    - NIPS 2024 Best Paper

13. **VAVAE** (2025)
    - "Reconstruction vs. Generation: Taming Optimization Dilemma in Latent Diffusion Models"
    - CVPR 2025 满分论文

14. **RAE** (2025)
    - Xie et al. "Diffusion Transformers with Representation Autoencoders"
    - https://arxiv.org/abs/2510.11690

15. **JiT** (2025)
    - He et al. "Back to Basics: Let Denoising Generative Models Denoise"
    - https://arxiv.org/abs/2511.13720

---

## 📚 第五章：学习路线建议

### 阶段一：基础入门 (1-2 个月)

1. **数学基础**:
   - 概率论 (贝叶斯定理、高斯分布)
   - 线性代数 (矩阵运算、特征分解)
   - 信息论 (熵、KL 散度)

2. **深度学习基础**:
   - CNN、RNN、Transformer
   - 自编码器 (AE、VAE)
   - GAN 基础

3. **核心论文**:
   - VAE (2013)
   - DDPM (2020)
   - DDIM (2020)

### 阶段二：进阶提升 (2-3 个月)

1. **Diffusion 深入**:
   - LDM/Stable Diffusion
   - ControlNet
   - LoRA 微调

2. **架构演进**:
   - UNet → DiT → MMDiT
   - Flow Matching

3. **实践项目**:
   - 基于 SD 的 LoRA 微调
   - ControlNet 应用
   - 自定义 Sampler

### 阶段三：前沿探索 (3-6 个月)

1. **最新架构**:
   - FLUX 系列
   - 自回归模型 (VAR、NextFlow)
   - 像素空间生成

2. **科研方向**:
   - 编码器改进 (VAVAE、RAE)
   - 一步生成 (Mean Flow)
   - 预测目标反思 (JiT)

3. **开源贡献**:
   - 参与开源项目
   - 复现前沿论文
   - 发表自己的研究

---

## 🔗 第六章：资源链接汇总

### 官方代码仓库

```
Stable Diffusion:
- https://github.com/CompVis/stable-diffusion
- https://github.com/Stability-AI/stablediffusion

FLUX:
- https://github.com/black-forest-labs/flux
- https://github.com/black-forest-labs/flux2

Qwen-Image:
- https://github.com/QwenLM/Qwen-Image

Hunyuan:
- https://github.com/Tencent-Hunyuan/HunyuanDiT
```

### 模型权重下载

```
HuggingFace:
- https://huggingface.co/stabilityai
- https://huggingface.co/black-forest-labs
- https://huggingface.co/Qwen
- https://huggingface.co/Wan-AI

ModelScope (阿里):
- https://modelscope.cn/models
```

### 学习资源

```
Papers With Code:
- https://paperswithcode.com/task/text-to-image-generation

Diffusers 库教程:
- https://huggingface.co/docs/diffusers

Civitai (社区模型):
- https://civitai.com
```

---

## 📝 附录：关键术语表

| 术语 | 全称 | 解释 |
|------|------|------|
| **VAE** | Variational Autoencoder | 变分自编码器 |
| **GAN** | Generative Adversarial Network | 生成对抗网络 |
| **DDPM** | Denoising Diffusion Probabilistic Models | 去噪扩散概率模型 |
| **DDIM** | Denoising Diffusion Implicit Models | 去噪扩散隐式模型 |
| **LDM** | Latent Diffusion Models | 隐空间扩散模型 |
| **DiT** | Diffusion Transformer | 扩散 Transformer |
| **MMDiT** | Multi-Modal Diffusion Transformer | 多模态扩散 Transformer |
| **VLM** | Vision-Language Model | 视觉语言模型 |
| **LoRA** | Low-Rank Adaptation | 低秩自适应 |
| **MoE** | Mixture of Experts | 专家混合 |

---

**最后更新**: 2026-04-03  
**持续更新中...**
