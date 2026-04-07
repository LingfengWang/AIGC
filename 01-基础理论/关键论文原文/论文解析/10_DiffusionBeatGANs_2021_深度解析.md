# Diffusion Beat GANs 论文深度解析

**论文**: Diffusion Models Beat GANs on Image Synthesis  
**作者**: Prafulla Dhariwal, Alex Nichol  
**机构**: OpenAI  
**发表**: NeurIPS 2021  
**引用**: 4,000+  
**文件**: `17_DiffusionBeatGANs_2021_Dhariwal.pdf` (38 MB)

---

## 📋 目录

1. [研究背景](#研究背景)
2. [核心问题](#核心问题)
3. [核心贡献](#核心贡献)
4. [数学原理](#数学原理)
5. [模型架构](#模型架构)
6. [训练方法](#训练方法)
7. [实验结果](#实验结果)
8. [代码实现](#代码实现)
9. [深入理解](#深入理解)
10. [与 DDPM 的改进](#与-ddpm-的改进)
11. [后续影响](#后续影响)
12. [学习建议](#学习建议)

---

## 研究背景

### 1.1 GAN 的统治地位 (2014-2020)

```
GAN 的黄金时代:

2014: GAN 提出
  └─ 开创性但难训练

2017: WGAN
  └─ 改进训练稳定性

2018: StyleGAN
  └─ 高质量人脸生成

2019: StyleGAN2
  └─ SOTA 图像生成

2020 现状:
- GAN 是图像生成 SOTA
- FID: 6-10 (CIFAR-10)
- IS: 200+ (ImageNet)
```

### 1.2 DDPM 的崛起 (2020)

```
DDPM (2020):

优点:
- 训练稳定
- 覆盖全面 (无 mode collapse)
- 理论优美

缺点:
- 采样慢 (1000 步)
- 质量不如 GAN
- FID: 9.4 vs GAN 的 6-8

问题:
- Diffusion 能超越 GAN 吗？
- 需要哪些改进？
```

### 1.3 动机

```
OpenAI 的问题:

1. 能否让 Diffusion 超越 GAN？
   - 质量更好
   - 多样性更好
   - 可控性更好

2. 需要哪些改进？
   - 模型架构？
   - 训练方法？
   - 采样策略？

3. 系统性对比
   - 公平对比 GAN vs Diffusion
   - 找出各自优势
```

---

## 核心问题

### 2.1 核心挑战

**如何改进 Diffusion 模型，使其在图像合成质量上超越 GAN？**

### 2.2 关键问题

```
问题 1: 质量差距在哪？
- GAN: FID 6-8
- DDPM: FID 9.4
- 差距原因？

问题 2: 如何改进？
- 更大的模型？
- 更好的架构？
- 改进采样？

问题 3: 如何公平对比？
- 相同数据集
- 相同评估指标
- 相同计算资源
```

### 2.3 解决思路

```
核心思路：系统性改进

1. 模型架构改进
   - 更大的 UNet
   - 更好的注意力
   - 更多参数

2. 训练技巧
   - 分类器引导
   - 改进噪声调度
   - 更多训练数据

3. 采样优化
   - 更少的步数
   - 更好的求解器
```

---

## 核心贡献

### 3.1 首次超越 GAN

**这是论文最核心的贡献！**

```
里程碑结果:

ImageNet 256×256:

| 模型 | FID ↓ | IS ↑ |
|------|-------|------|
| BigGAN | 14.7 | 166.3 |
| DDPM | 15.2 | 79.2 |
| **改进 Diffusion** | **9.6** | **259.7** |

结论:
- Diffusion 首次超越 GAN
- FID 降低 35%
- IS 提高 56%
```

### 3.2 分类器引导 (Classifier Guidance)

```
核心创新：分类器引导

标准 Diffusion:
- 无条件生成
- 或条件通过交叉注意力

分类器引导:
- 训练一个分类器 c(y|x)
- 生成时用分类器梯度引导
- ε_guided = ε + s · ∇_x log c(y|x)

其中 s 是引导强度

效果:
- 更好的条件生成
- 更高的质量
- 更可控
```

### 3.3 架构改进

```
改进的 UNet 架构:

1. 更多参数
   - 从 55M 到 554M
   - 更深更宽

2. 更好的注意力
   - 更多注意力头
   - 更大感受野

3. 改进的残差块
   - 更好的归一化
   - 更稳定的训练
```

### 3.4 系统性对比

```
公平对比实验:

相同条件:
- 相同数据集 (ImageNet)
- 相同分辨率 (256×256)
- 相同评估指标 (FID, IS)
- 相似计算资源

发现:
- Diffusion 多样性更好
- GAN 质量略好 (改进前)
- Diffusion 训练更稳定
```

---

## 数学原理

### 4.1 分类器引导

```
标准条件生成:

p(x|y) = p(x) · p(y|x) / p(y)

取对数:
log p(x|y) = log p(x) + log p(y|x) - log p(y)

梯度:
∇_x log p(x|y) = ∇_x log p(x) + ∇_x log p(y|x)

其中:
- ∇_x log p(x): 无条件分数 (Diffusion 学习)
- ∇_x log p(y|x): 分类器梯度 (额外训练)
```

```
引导采样:

标准反向扩散:
dx = [f(x,t) - g(t)²∇_x log p(x)]dt + g(t)dw̄

引导反向扩散:
dx = [f(x,t) - g(t)²(∇_x log p(x) + s·∇_x log p(y|x))]dt + g(t)dw̄

其中 s 是引导强度:
- s=0: 无条件
- s=1: 标准条件
- s>1: 增强引导
```

### 4.2 改进的噪声调度

```
DDPM 噪声调度:

β_t 从 10⁻⁴ 线性增加到 0.02

改进调度:

余弦调度:
β_t = 1 - (f(t)/f(0))²
f(t) = cos((t/T + s)/(1+s) × π/2)

好处:
- 早期加噪更慢
- 保留更多信息
- 生成质量更好
```

### 4.3 架构缩放

```
模型大小缩放:

| 模型 | 参数 | FID |
|------|------|-----|
| Small | 55M | 12.5 |
| Medium | 200M | 10.8 |
| Large | 554M | 9.6 |

观察:
- 参数增加 → FID 下降
- 没有饱和迹象
- 更大模型应该更好
```

---

## 模型架构

### 5.1 改进的 UNet

```
架构改进:

原始 DDPM UNet:
- 基础通道：128
- 注意力：16×16 分辨率
- 参数：55M

改进 UNet:
- 基础通道：256
- 注意力：32×32, 16×16, 8×8
- 参数：554M

具体:
- 更多残差块
- 更多注意力头
- 更大的嵌入维度
```

### 5.2 注意力机制

```
多尺度注意力:

原始 DDPM:
- 只在 16×16 用注意力

改进:
- 32×32 用注意力
- 16×16 用注意力
- 8×8 用注意力

好处:
- 捕捉多尺度依赖
- 更好的全局结构
- 更清晰的细节
```

### 5.3 残差块改进

```
改进的残差块:

原始:
Conv → GroupNorm → SiLU → Conv → GroupNorm → Add

改进:
Conv → GroupNorm (更小组数) → SiLU → Conv → GroupNorm → Add

变化:
- GroupNorm 组数从 32 减到 1
- 更稳定的训练
- 更好的梯度流
```

---

## 训练方法

### 6.1 训练算法

```
Algorithm 1: 改进 Diffusion 训练

Input: 图像数据集 {x⁽¹⁾, ..., x⁽ᴺ⁾}, 类别 {y⁽¹⁾, ..., y⁽ᴺ⁾}
Initialize: 
  - Diffusion 模型 ε_θ
  - 分类器 c_φ

# 阶段 1: 训练扩散模型
repeat
    for each x⁽ⁱ⁾:
        t ~ Uniform(1, T)
        ε ~ N(0, I)
        x_t = √ᾱ_t x⁽ⁱ⁾ + √(1-ᾱ_t) ε
        ε_θ = UNet(x_t, t)
        L = ||ε - ε_θ||²
    更新 θ
until 收敛

# 阶段 2: 训练分类器
repeat
    for each (x⁽ⁱ⁾, y⁽ⁱ⁾):
        t ~ Uniform(1, T)
        ε ~ N(0, I)
        x_t = √ᾱ_t x⁽ⁱ⁾ + √(1-ᾱ_t) ε
        y_pred = Classifier(x_t, t, y⁽ⁱ⁾)
        L = CrossEntropy(y_pred, y⁽ⁱ⁾)
    更新 φ
until 收敛
```

### 6.2 分类器训练

```
分类器设计:

架构:
- 类似 UNet 的编码器
- 时间条件化
- 输出类别概率

训练:
- 带噪图像 x_t
- 时间 t
- 真实类别 y
- 交叉熵损失

关键:
- 在不同噪声水平都能分类
- 时间条件化很重要
- 不需要很强 (5% 准确率就够)
```

### 6.3 引导采样

```
Algorithm 2: 分类器引导采样

Input: 类别 y, 引导强度 s
Output: 生成的图像 x₀

# 从噪声开始
x_T ~ N(0, I)

for t = T, T-1, ..., 1:
    # 预测噪声 (无条件)
    ε_uncond = UNet(x_t, t)
    
    # 计算分类器梯度
    with grad():
        c = Classifier(x_t, t, y)
        log_c = log(c)
        grad = ∇_{x_t} log_c
    
    # 引导
    ε_guided = ε_uncond + s · g(t)² · grad
    
    # 反向扩散步
    x_{t-1} = DDPM_Step(x_t, ε_guided, t)

return x₀
```

### 6.4 训练技巧

```
1. 大 batch 训练
   - batch size 4096
   - 更稳定的梯度
   - 更好的质量

2. 混合精度
   - FP16 训练
   - 节省显存
   - 加速训练

3. EMA
   - 指数移动平均
   - 改善生成质量
   - 推理时使用

4. 更多数据
   - ImageNet 全量
   - 128 万图像
   - 更好的泛化
```

---

## 实验结果

### 7.1 ImageNet 生成

```
ImageNet 256×256 主要结果:

| 模型 | 类型 | FID ↓ | IS ↑ | Precision | Recall |
|------|------|-------|------|-----------|--------|
| BigGAN | GAN | 14.7 | 166.3 | 0.70 | 0.48 |
| StyleGAN2 | GAN | 15.8 | 135.2 | 0.68 | 0.45 |
| DDPM | Diffusion | 15.2 | 79.2 | 0.71 | 0.59 |
| **Ours (无引导)** | **Diffusion** | **10.4** | **181.2** | **0.75** | **0.63** |
| **Ours (引导 s=4)** | **Diffusion** | **9.6** | **259.7** | **0.80** | **0.57** |

关键发现:
- 无引导已超越 GAN
- 引导后进一步提升
- Precision 和 Recall 都更好
```

### 7.2 CIFAR-10

```
CIFAR-10 结果:

| 模型 | FID ↓ | IS ↑ |
|------|-------|------|
| StyleGAN2 | 8.3 | 9.5 |
| DDPM | 9.4 | 9.0 |
| **Ours** | **6.0** | **9.9** |

结论:
- 小数据集上也超越 GAN
- FID 降低 36%
- IS 提升 40%
```

### 7.3 消融实验

```
架构改进消融:

| 改进 | FID | 提升 |
|------|-----|------|
| 基线 DDPM | 15.2 | - |
| + 更大模型 | 12.5 | 18% |
| + 更多注意力 | 11.2 | 10% |
| + 改进归一化 | 10.8 | 4% |
| + 更多数据 | 10.4 | 4% |
| + 分类器引导 | 9.6 | 8% |

结论:
- 模型大小最重要
- 注意力也很关键
- 引导带来额外提升
```

### 7.4 引导强度

```
引导强度 s 的影响:

| s | FID | IS | Precision | Recall |
|---|-----|-----|-----------|--------|
| 0 (无条件) | 10.4 | 181.2 | 0.75 | 0.63 |
| 1 | 10.1 | 200.5 | 0.77 | 0.61 |
| 2 | 9.8 | 225.3 | 0.79 | 0.59 |
| 4 | 9.6 | 259.7 | 0.80 | 0.57 |
| 8 | 10.2 | 245.1 | 0.82 | 0.52 |

观察:
- s=4 最优 (FID)
- s 越大 Precision 越高
- s 越大 Recall 越低
- 需要权衡
```

---

## 代码实现

### 8.1 分类器模型

```python
class Classifier(nn.Module):
    def __init__(self, num_classes=1000, channels=128):
        super().__init__()
        self.num_classes = num_classes
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, channels * 4),
            nn.SiLU(),
            nn.Linear(channels * 4, channels),
        )
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels, 3, stride=2, padding=1),
            nn.GroupNorm(1, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
            nn.GroupNorm(1, channels * 2),
            nn.SiLU(),
            nn.Conv2d(channels * 2, channels * 4, 3, stride=2, padding=1),
            nn.GroupNorm(1, channels * 4),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        
        # 分类头
        self.classifier = nn.Linear(channels * 4, num_classes)
    
    def forward(self, x, t, y=None):
        # 时间嵌入
        t_emb = self.time_embed(t.float().unsqueeze(-1))
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)
        
        # 编码
        h = self.encoder(x)
        h = h + t_emb  # 添加时间信息
        
        # 分类
        logits = self.classifier(h)
        
        return logits
```

### 8.2 分类器训练

```python
def train_classifier(classifier, diffusion, dataloader, optimizer, device, epochs=10):
    classifier.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # 采样时间
            t = torch.randint(0, diffusion.T, (images.shape[0],)).to(device)
            
            # 采样噪声
            noise = torch.randn_like(images)
            
            # 加噪
            noisy_images = diffusion.add_noise(images, noise, t)
            
            # 分类
            logits = classifier(noisy_images, t, labels)
            
            # 损失
            loss = F.cross_entropy(logits, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}: Loss = {avg_loss:.4f}')
```

### 8.3 引导采样

```python
@torch.no_grad()
def sample_with_guidance(unet, classifier, diffusion, class_label, 
                         guidance_scale=4.0, num_steps=250):
    """分类器引导采样"""
    unet.eval()
    classifier.eval()
    
    # 从噪声开始
    x = torch.randn(1, 3, 256, 256).to(unet.device)
    
    for t in reversed(range(diffusion.T)):
        t_tensor = torch.full((1,), t, device=unet.device)
        
        # 预测噪声 (无条件)
        noise_pred = unet(x, t_tensor)
        
        # 计算分类器梯度
        x.requires_grad_(True)
        logits = classifier(x, t_tensor, class_label)
        log_probs = F.log_softmax(logits, dim=1)
        log_prob = log_probs[0, class_label]
        grad = torch.autograd.grad(log_prob, x)[0]
        x.requires_grad_(False)
        
        # 引导
        noise_pred = noise_pred + guidance_scale * grad
        
        # 反向扩散步
        x = diffusion.step(x, noise_pred, t)
    
    return x
```

### 8.4 改进的 UNet

```python
class ImprovedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=256):
        super().__init__()
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, base_channels * 4),
            nn.SiLU(),
            nn.Linear(base_channels * 4, base_channels * 4),
        )
        
        # 下采样 (更多层)
        self.down = nn.ModuleList([
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            ResBlock(base_channels, base_channels),
            ResBlock(base_channels, base_channels),
            Attention(base_channels),  # 32×32 注意力
            Downsample(base_channels),
            ResBlock(base_channels, base_channels * 2),
            ResBlock(base_channels * 2, base_channels * 2),
            Attention(base_channels * 2),  # 16×16 注意力
            Downsample(base_channels * 2),
            ResBlock(base_channels * 2, base_channels * 4),
            ResBlock(base_channels * 4, base_channels * 4),
            Attention(base_channels * 4),  # 8×8 注意力
        ])
        
        # 中间层
        self.middle = nn.Sequential(
            ResBlock(base_channels * 4, base_channels * 4),
            Attention(base_channels * 4),
            ResBlock(base_channels * 4, base_channels * 4),
        )
        
        # 上采样
        self.up = nn.ModuleList([...])  # 对称结构
        
        # 输出
        self.out_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)
    
    def forward(self, x, t):
        # 时间嵌入
        t_emb = self.time_embed(t.float().unsqueeze(-1))
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)
        
        # UNet 前向
        # ...
        
        return noise_pred
```

---

## 深入理解

### 9.1 为什么分类器引导有效？

```
直觉解释 1: 梯度上升

分类器 c(y|x) 的梯度:
∇_x log c(y|x) 指向:
- 更可能被分类为 y 的区域
- 更符合类别 y 的特征

引导作用:
- 在生成过程中
- 沿着分类器梯度方向
- 生成更符合类别的图像

类比:
- 在山上找特定标记的位置
- 分类器是标记检测器
- 梯度指向标记
```

```
直觉解释 2: 贝叶斯更新

后验分布:
p(x|y) ∝ p(x) · p(y|x)

生成过程:
- p(x): 无条件 Diffusion 学习
- p(y|x): 分类器提供
- 组合得到条件分布

引导强度 s:
- s=0: 只用 p(x)
- s=1: 标准贝叶斯
- s>1: 增强条件
```

### 9.2 为什么需要大模型？

```
模型大小的影响:

小模型 (55M):
- 容量有限
- 学习细节困难
- FID: 12.5

大模型 (554M):
- 容量充足
- 学习细节容易
- FID: 9.6

关键:
- 高质量需要高容量
- 细节需要足够参数
- 没有饱和迹象
```

### 9.3 与 GAN 的对比

```
Diffusion vs GAN:

Diffusion 优势:
- 训练稳定
- 覆盖全面 (高 Recall)
- 多样性好
- 易于训练

GAN 优势:
- 采样快 (1 步 vs 1000 步)
- 推理简单
- 成熟生态

结论:
- Diffusion 质量超越 GAN
- GAN 速度仍有优势
- 各有适用场景
```

### 9.4 引导强度的权衡

```
Precision vs Recall:

引导强度 s 增加:
- Precision ↑ (质量提高)
- Recall ↓ (多样性降低)
- FID 先降后升

最优选择:
- 追求质量：s=4-8
- 追求多样性：s=1-2
- 平衡：s=2-4
```

---

## 与 DDPM 的改进

### 10.1 架构改进

```
DDPM → 改进 Diffusion:

| 组件 | DDPM | 改进 |
|------|------|------|
| **基础通道** | 128 | 256 |
| **参数量** | 55M | 554M |
| **注意力** | 16×16 | 32×16×8 |
| **归一化** | GN(32) | GN(1) |
| **训练数据** | ImageNet 部分 | ImageNet 全量 |
```

### 10.2 训练改进

```
DDPM → 改进 Diffusion:

| 技巧 | DDPM | 改进 |
|------|------|------|
| **Batch Size** | 128 | 4096 |
| **混合精度** | 否 | 是 |
| **EMA** | 是 | 是 |
| **分类器** | 无 | 有 |
| **引导** | 无 | 有 |
```

### 10.3 性能提升

```
FID 提升分解:

DDPM 基线: 15.2
  ↓ +更大模型
12.5 (-18%)
  ↓ +更多注意力
11.2 (-10%)
  ↓ +改进归一化
10.8 (-4%)
  ↓ +更多数据
10.4 (-4%)
  ↓ +分类器引导
9.6 (-8%)

总提升：37%
```

---

## 后续影响

### 11.1 直接后续工作

```
2021-2024 后续工作:

1. Classifier-Free Guidance (2022)
   - 不需要额外分类器
   - 更简单有效

2. GLIDE (2022)
   - 文本引导生成
   - 基于分类器引导

3. Stable Diffusion (2022)
   - 隐空间引导
   - 更高效
```

### 11.2 工业界影响

```
工业界采用:

OpenAI:
- DALL·E 2: 使用引导
- GLIDE: 文本引导

Stability AI:
- SD: 分类器自由引导
- 成为标准配置

Midjourney:
- 高质量生成
- 引导技术
```

### 11.3 研究影响

```
引用趋势:
- 2021: 500+ 引用
- 2022: 1200+ 引用
- 2023: 1500+ 引用
- 2024: 800+ 引用

影响领域:
- 图像生成
- 条件生成
- 引导技术
- 评估方法
```

---

## 学习建议

### 12.1 前置知识

| 知识 | 重要程度 | 推荐资源 |
|------|---------|---------|
| DDPM | ⭐⭐⭐⭐⭐ | DDPM 论文解析 |
| GAN 基础 | ⭐⭐⭐⭐ | GAN 教程 |
| 分类器 | ⭐⭐⭐⭐ | CNN 分类 |
| 评估指标 | ⭐⭐⭐⭐ | FID, IS 计算 |

### 12.2 学习路径

```
第 1 步：掌握 DDPM (6 小时)
  └─ 理解 Diffusion 基础

第 2 步：理解改进 (4 小时)
  └─ 架构改进
  └─ 训练技巧

第 3 步：学习分类器引导 (6 小时)
  └─ 数学原理
  └─ 实现方法

第 4 步：动手实践 (12 小时)
  └─ 训练分类器
  └─ 实现引导采样

第 5 步：深入理解 (6 小时)
  └─ 与 GAN 对比
  └─ 引导强度权衡
```

### 12.3 实践项目

| 项目 | 难度 | 用时 |
|------|------|------|
| 分类器训练 | ⭐⭐⭐ | 4 小时 |
| 引导采样实现 | ⭐⭐⭐⭐ | 8 小时 |
| CIFAR-10 实验 | ⭐⭐⭐⭐ | 12 小时 |
| 引导强度对比 | ⭐⭐⭐ | 6 小时 |

---

## 总结

### 核心要点

1. **首次超越**: Diffusion 首次在图像合成上超越 GAN
2. **分类器引导**: 用分类器梯度引导生成
3. **大模型**: 554M 参数带来质量提升
4. **系统对比**: 公平对比 Diffusion vs GAN

### 历史地位

- **里程碑**: Diffusion 超越 GAN 的转折点
- **影响力**: 4,000+ 引用，工业界标准
- **后续**: 引导技术成为标配

### 学习价值

- **理论基础**: 理解条件生成和引导技术
- **实践价值**: 引导技术广泛使用
- **后续学习**: 为学习 Classifier-Free Guidance 打基础

---

**Diffusion 正式超越 GAN，开启新时代！** 🎉

---

*最后更新*: 2026-04-07
