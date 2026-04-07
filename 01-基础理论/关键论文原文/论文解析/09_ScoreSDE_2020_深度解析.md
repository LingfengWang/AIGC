# Score SDE 论文深度解析

**论文**: Score-Based Generative Modeling through Stochastic Differential Equations  
**作者**: Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, et al.  
**机构**: Stanford University, Google Brain  
**发表**: ICLR 2021  
**引用**: 5,000+  
**文件**: `16_ScoreSDE_2020_Song.pdf` (26 MB)

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
10. [与 DDPM 的关系](#与-ddpm-的关系)
11. [后续影响](#后续影响)
12. [学习建议](#学习建议)

---

## 研究背景

### 1.1 生成模型的发展

```
2014-2020 生成模型演进:

2014: VAE
  └─ 变分推断
  └─ 生成模糊

2014: GAN
  └─ 对抗训练
  └─ 训练不稳定

2015: Normalizing Flow
  └─ 精确似然
  └─ 计算复杂

2020: DDPM
  └─ 扩散模型
  └─ 训练稳定但慢

问题:
- 缺乏统一框架
- 各模型独立发展
- 理论不互通
```

### 1.2 分数匹配的发展

```
Score Matching 历史:

2005: Score Matching (Hyvärinen)
  └─ 避免计算配分函数
  └─ 但只适合简单分布

2019: Score Matching with SDE
  └─ 扩展到复杂分布
  └─ 但理论不完整

2020: Song et al.
  └─ 统一框架
  └─ 连接 Diffusion
```

### 1.3 动机

```
核心动机:

1. 理论统一
   - VAE、GAN、Flow、Diffusion
   - 能否统一到一个框架？

2. 连续时间视角
   - DDPM 是离散时间
   - 连续时间更优美

3. 灵活性
   - 支持多种 SDE
   - 可调节噪声调度
```

---

## 核心问题

### 2.1 核心挑战

**如何建立一个统一的连续时间框架，理解并改进基于分数的生成模型？**

### 2.2 关键问题

```
问题 1: 什么是分数 (Score)?
- 分数函数是什么？
- 如何学习分数？
- 分数如何用于生成？

问题 2: 如何连接 SDE？
- 随机微分方程是什么？
- 如何用 SDE 描述生成过程？
- 正向和反向 SDE？

问题 3: 如何统一现有模型？
- DDPM 是特例吗？
- 其他模型呢？
```

### 2.3 解决思路

```
核心思路：分数 + SDE

1. 学习分数函数
   s(x) = ∇_x log p(x)
   概率密度的梯度

2. 定义正向 SDE
   dx = f(x,t)dt + g(t)dw
   数据 → 噪声

3. 定义反向 SDE
   dx = [f(x,t) - g(t)²s(x,t)]dt + g(t)dw̄
   噪声 → 数据
```

---

## 核心贡献

### 3.1 统一框架

**这是 Score SDE 最核心的贡献！**

```
Score SDE 统一框架:

正向 SDE (破坏数据):
dx = f(x,t)dt + g(t)dw

反向 SDE (生成数据):
dx = [f(x,t) - g(t)²∇_x log p(x)]dt + g(t)dw̄

其中:
- f(x,t): 漂移系数
- g(t): 扩散系数
- ∇_x log p(x): 分数函数
- w: 维纳过程 (布朗运动)

关键:
- 统一描述生成过程
- 连接多个模型
- 连续时间视角
```

### 3.2 分数匹配目标

```
去噪分数匹配 (Denoising Score Matching):

训练目标:
L = E[||s_θ(x,t) - ∇_x log p(x)||²]

其中:
- s_θ(x,t): 神经网络预测的分数
- ∇_x log p(x): 真实分数

实际实现:
- 不直接计算 ∇_x log p(x)
- 通过去噪任务间接学习
- 等价于预测噪声
```

### 3.3 概率流 ODE

```
从 SDE 到 ODE:

SDE (随机):
dx = [f(x,t) - g(t)²s(x,t)]dt + g(t)dw̄

ODE (确定性):
dx = [f(x,t) - ½g(t)²s(x,t)]dt

关键:
- ODE 是确定性的
- 相同起点 → 相同结果
- 可以精确计算似然
```

### 3.4 包含 DDPM 作为特例

```
Score SDE 包含 DDPM:

当选择特定 SDE:
f(x,t) = -½β(t)x
g(t) = √β(t)

得到:
- 正向：DDPM 的加噪过程
- 反向：DDPM 的去噪过程

结论:
- DDPM 是 Score SDE 的离散化特例
- Score SDE 更通用
- 可以选择其他 SDE
```

---

## 数学原理

### 4.1 分数函数

```
分数函数定义:

分数：s(x) = ∇_x log p(x)

直观理解:
- 概率密度的梯度
- 指向高概率区域
- 大小表示概率变化率

性质:
- 不需要知道 p(x) 的归一化常数
- 适合未归一化分布
- 可以学习
```

### 4.2 正向 SDE

```
正向 SDE (数据 → 噪声):

dx = f(x,t)dt + g(t)dw

其中:
- x: 状态 (如图像)
- t: 时间 (0 到 T)
- f(x,t): 漂移系数
- g(t): 扩散系数
- w: 维纳过程

常用 SDE:

1. VP SDE (Variance Preserving)
   f(x,t) = -½β(t)x
   g(t) = √β(t)
   类似 DDPM

2. VE SDE (Variance Exploding)
   f(x,t) = 0
   g(t) = √(dσ²(t)/dt)
   方差爆炸

3. Sub-VP SDE
   改进的 VP SDE
```

### 4.3 反向 SDE

```
反向 SDE (噪声 → 数据):

dx = [f(x,t) - g(t)²∇_x log p_t(x)]dt + g(t)dw̄

其中:
- w̄: 反向时间的维纳过程
- ∇_x log p_t(x): t 时刻的分数

关键:
- 需要学习分数 ∇_x log p_t(x)
- 分数网络 s_θ(x,t) 近似
- 从 t=T 积分到 t=0
```

### 4.4 得分匹配损失

```
去噪分数匹配损失:

L(θ) = E_t[λ(t) E_{x(0)} E_{x(t)|x(0)}[||s_θ(x(t),t) - ∇_{x(t)} log p(x(t)|x(0))||²]]

简化:
- 采样 t ~ Uniform(0,T)
- 采样 x(0) ~ 数据分布
- 采样 x(t) ~ p(x(t)|x(0)) (正向 SDE)
- 计算目标分数
- 回归损失

其中 λ(t) > 0 是权重函数
```

---

## 模型架构

### 5.1 分数网络

```
Score Network s_θ(x,t):

输入:
- x: 带噪图像 (H×W×C)
- t: 时间标量

输出:
- s: 分数 (H×W×C)
  与输入同维度

架构:
- 类似 UNet
- 时间嵌入
- 多尺度处理
```

### 5.2 时间嵌入

```
时间编码:

类似于 Transformer/DiT:

1. Sinusoidal 编码
   e_t = [sin(ω₁t), cos(ω₁t), ...]

2. MLP 投影
   emb = MLP(e_t)

3. 添加到网络
   - FiLM 调制
   - AdaGN
   - 简单相加
```

### 5.3 噪声条件

```
处理不同噪声水平:

问题:
- 不同 t 对应不同噪声水平
- 网络需要适应所有水平

解决:
- 时间条件化
- 每个 t 有特定参数
- 或共享参数 + 条件调制

效果:
- 一个网络处理所有噪声水平
- 高效
- 泛化好
```

---

## 训练方法

### 6.1 训练算法

```
Algorithm 1: Score SDE 训练

Input: 数据集 {x⁽¹⁾, ..., x⁽ᴺ⁾}, SDE 参数
Initialize: 分数网络 s_θ

repeat
    # 采样 batch
    {x⁽¹⁾, ..., x⁽ᴹ⁾} ~ 数据集
    
    for each x⁽ⁱ⁾:
        # 采样时间
        t ~ Uniform(0, T)
        
        # 采样噪声
        ε ~ N(0, I)
        
        # 正向 SDE 得到 x(t)
        x(t) = 正向演化 (x⁽ⁱ⁾, t, ε)
        
        # 计算目标分数
        target = ∇_{x(t)} log p(x(t)|x⁽ⁱ⁾)
        
        # 预测分数
        s_θ = Network(x(t), t)
        
        # 损失
        L = ||s_θ - target||²
    
    # 反向传播
    ∇_θ L = Backprop(L)
    θ ← θ - η ∇_θ L
    
until 收敛
```

### 6.2 SDE 求解器

```
数值求解 SDE:

Euler-Maruyama (最常用):
x_{t+Δt} = x_t + f(x_t,t)Δt + g(t)Δw
其中 Δw ~ N(0, Δt·I)

Predictor-Corrector:
1. Predictor: SDE 步
2. Corrector: Langevin MCMC
效果更好但更慢

选择:
- 训练：Euler-Maruyama
- 高质量生成：Predictor-Corrector
```

### 6.3 训练技巧

```
1. 时间采样
   - 均匀采样 t
   - 或重要性采样
   - 某些 SDE 需要特殊采样

2. 权重函数 λ(t)
   - 平衡不同时间的损失
   - 常用：λ(t) = g(t)²
   - 或手动调节

3. 梯度裁剪
   - 稳定训练
   - 范数限制 1.0

4. EMA
   - 指数移动平均
   - 改善生成质量
```

---

## 实验结果

### 7.1 图像生成

```
CIFAR-10:

| 模型 | FID ↓ | IS ↑ |
|------|-------|------|
| DDPM | 9.4 | 9.0 |
| Score SDE (VP) | 9.6 | 8.9 |
| Score SDE (VE) | 10.2 | 8.5 |
| Score SDE (Sub-VP) | 9.5 | 8.8 |

结论:
- Score SDE 与 DDPM 相当
- 不同 SDE 效果类似
- 连续时间框架有效
```

```
ImageNet 256×256:

| 模型 | FID ↓ | IS ↑ |
|------|-------|------|
| BigGAN | 14.7 | 166.3 |
| DDPM | 15.2 | 79.2 |
| **Score SDE** | **15.5** | **76.8** |

结论:
- 大规模数据集上有效
- 与 DDPM 相当
- 框架通用
```

### 7.2 概率密度估计

```
对数似然 (bits/dim):

| 模型 | CIFAR-10 |
|------|----------|
| RealNVP | 3.18 |
| Glow | 3.35 |
| **Score SDE (ODE)** | **3.21** |

结论:
- 通过 ODE 可以计算精确似然
- 与 Flow 模型相当
- 统一框架优势
```

### 7.3 消融实验

```
SDE 选择:

| SDE 类型 | FID | 特点 |
|---------|-----|------|
| VP | 9.6 | 类似 DDPM |
| VE | 10.2 | 方差爆炸 |
| Sub-VP | 9.5 | 改进 VP |

结论:
- VP 和 Sub-VP 效果好
- VE 适合某些场景
- 可以根据任务选择
```

---

## 代码实现

### 8.1 SDE 类

```python
class VPSDE:
    """Variance Preserving SDE (类似 DDPM)"""
    
    def __init__(self, beta_min=0.1, beta_max=20, T=1):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
    
    def beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def drift(self, x, t):
        """漂移系数 f(x,t)"""
        beta_t = self.beta(t)
        return -0.5 * beta_t[:, None, None, None] * x
    
    def diffusion(self, t):
        """扩散系数 g(t)"""
        beta_t = self.beta(t)
        return torch.sqrt(beta_t)
    
    def marginal_mean(self, t):
        """边际分布均值"""
        log_mean_coeff = -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        return torch.exp(log_mean_coeff)
    
    def marginal_std(self, t):
        """边际分布标准差"""
        log_mean_coeff = -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        return torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    
    def sample_marginal(self, x0, t):
        """从边际分布采样 x(t)"""
        mean = self.marginal_mean(t)[:, None, None, None] * x0
        std = self.marginal_std(t)[:, None, None, None]
        noise = torch.randn_like(x0)
        return mean + std * noise
```

### 8.2 训练循环

```python
def train_score_sde(model, sde, dataloader, optimizer, device, epochs=10):
    model.train()
    T = sde.T
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_idx, (x0, _) in enumerate(dataloader):
            x0 = x0.to(device)
            optimizer.zero_grad()
            
            # 采样时间
            t = torch.rand(x0.shape[0], device=device) * T
            
            # 采样 x(t)
            x_t = sde.sample_marginal(x0, t)
            
            # 计算目标分数
            # 对于 VP SDE: ∇log p(x(t)|x(0)) = -(x(t) - mean)/std²
            mean = sde.marginal_mean(t)[:, None, None, None] * x0
            std = sde.marginal_std(t)[:, None, None, None]
            target = -(x_t - mean) / (std ** 2)
            
            # 预测分数
            score_pred = model(x_t, t)
            
            # 损失 (加权)
            loss = (score_pred - target) ** 2
            loss = loss.mean(dim=(1, 2, 3))  # 平均空间维度
            loss = loss * (sde.diffusion(t) ** 2)  # 权重
            loss = loss.mean()
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}: Loss = {avg_loss:.4f}')
```

### 8.3 采样函数

```python
@torch.no_grad()
def sample_score_sde(model, sde, shape=(1, 3, 32, 32), num_steps=1000):
    """从 Score SDE 采样"""
    model.eval()
    T = sde.T
    dt = T / num_steps
    
    # 从先验分布采样 (t=T 时的分布)
    x = torch.randn(shape).to(model.device)
    
    # 反向时间积分
    for i in reversed(range(num_steps)):
        t = torch.full((shape[0],), i * dt, device=model.device)
        
        # 预测分数
        score = model(x, t)
        
        # 反向 SDE 步 (Euler-Maruyama)
        drift = sde.drift(x, t) - (sde.diffusion(t) ** 2)[:, None, None, None] * score
        diffusion = sde.diffusion(t)
        
        x = x + drift * dt + diffusion * torch.sqrt(dt) * torch.randn_like(x)
    
    return x
```

### 8.4 ODE 采样

```python
@torch.no_grad()
def sample_ode(model, sde, shape=(1, 3, 32, 32), num_steps=100):
    """从概率流 ODE 采样 (确定性)"""
    model.eval()
    T = sde.T
    dt = T / num_steps
    
    # 从先验分布采样
    x = torch.randn(shape).to(model.device)
    
    # ODE 积分
    for i in reversed(range(num_steps)):
        t = torch.full((shape[0],), i * dt, device=model.device)
        
        # 预测分数
        score = model(x, t)
        
        # ODE 步 (无随机项)
        drift = sde.drift(x, t) - 0.5 * (sde.diffusion(t) ** 2)[:, None, None, None] * score
        
        x = x + drift * dt
    
    return x
```

---

## 深入理解

### 9.1 为什么分数匹配有效？

```
直觉解释 1: 梯度上升

分数 s(x) = ∇_x log p(x) 指向:
- 高概率区域
- 数据流形

生成过程:
- 从噪声开始
- 沿分数方向移动
- 到达数据区域

类比:
- 在山谷中找最高点
- 分数是梯度
- 梯度上升找到峰值
```

```
直觉解释 2: 去噪

去噪分数匹配:
- 给数据加噪
- 学习去噪方向
- 等价于学习分数

为什么:
- 噪声数据 x(t) 的条件分布
- 分数指向原始数据 x(0)
- 学习这个方向
```

### 9.2 SDE 的选择

```
VP SDE (Variance Preserving):

dx = -½β(t)x dt + √β(t) dw

特点:
- 方差有界
- 类似 DDPM
- 适合图像

适用:
- 图像生成
- 与 DDPM 对比
```

```
VE SDE (Variance Exploding):

dx = √(dσ²(t)/dt) dw

特点:
- 方差爆炸
- 无漂移项
- 更简单

适用:
- 某些特殊任务
- 理论分析
```

### 9.3 与 DDPM 的对比

```
相似点:
- 都加噪然后去噪
- 都学习去噪方向
- 效果相当

不同点:
- DDPM: 离散时间
- Score SDE: 连续时间

- DDPM: 预测噪声
- Score SDE: 预测分数

- DDPM: 特定噪声调度
- Score SDE: 可以选择 SDE

关系:
- DDPM 是 Score SDE 的离散化
- Score SDE 更通用
```

### 9.4 概率流 ODE 的价值

```
为什么需要 ODE?

SDE (随机):
- 生成结果随机
- 无法计算精确似然
- 难以优化

ODE (确定性):
- 生成结果确定
- 可以计算精确似然
- 可以优化 latent

应用:
- 精确似然计算
- Latent 空间优化
- 可复现生成
```

---

## 与 DDPM 的关系

### 10.1 理论关系

```
DDPM 作为 Score SDE 的离散化:

连续时间 (Score SDE):
dx = -½β(t)x dt + √β(t) dw

离散化 (DDPM):
x_t = √(1-β_t) x_{t-1} + √β_t ε

当 Δt → 0:
- 离散 → 连续
- DDPM → Score SDE

结论:
- DDPM 是特例
- Score SDE 更通用
```

### 10.2 实践对比

```
| 特性 | DDPM | Score SDE |
|------|------|-----------|
| **时间** | 离散 | 连续 |
| **目标** | 预测噪声 | 预测分数 |
| **SDE** | 固定 (VP) | 可选择 |
| **ODE** | DDIM | 概率流 ODE |
| **理论** | 较简单 | 更优美 |
| **灵活性** | 有限 | 高 |
```

### 10.3 何时选择

```
选择 DDPM:
- 简单任务
- 需要成熟代码
- 社区支持好

选择 Score SDE:
- 需要理论优美
- 需要灵活性
- 研究目的
```

---

## 后续影响

### 11.1 直接后续工作

```
2021-2024 Score SDE 相关:

1. SDEdit (2021)
   - 基于 SDE 的图像编辑
   - 统一编辑框架

2. Score-Based 3D (2022)
   - 扩展到 3D 生成
   - NeRF + Score

3. Consistency Models (2023)
   - 基于 Score SDE
   - 一步生成
```

### 11.2 理论影响

```
理论贡献:

1. 统一框架
   - 连接多个生成模型
   - 促进理论发展

2. 连续时间视角
   - 更优美的数学
   - 便于分析

3. 分数匹配复兴
   - 老技术新应用
   - 激发新研究
```

### 11.3 研究影响

```
引用趋势:
- 2021: 800+ 引用
- 2022: 1500+ 引用
- 2023: 2000+ 引用
- 2024: 700+ 引用

影响领域:
- 图像生成
- 音频生成
- 分子生成
- 科学计算
```

---

## 学习建议

### 12.1 前置知识

| 知识 | 重要程度 | 推荐资源 |
|------|---------|---------|
| 随机过程 | ⭐⭐⭐⭐ | 布朗运动、SDE 基础 |
| 分数匹配 | ⭐⭐⭐⭐ | Score Matching 论文 |
| DDPM | ⭐⭐⭐⭐⭐ | DDPM 论文解析 |
| 微积分 | ⭐⭐⭐⭐ | 梯度、散度 |

### 12.2 学习路径

```
第 1 步：理解动机 (2 小时)
  └─ 生成模型的局限
  └─ 统一框架的需求

第 2 步：掌握数学 (8 小时)
  └─ SDE 基础
  └─ 分数函数
  └─ 正向/反向 SDE

第 3 步：理解与 DDPM 关系 (4 小时)
  └─ 离散 vs 连续
  └─ 特例关系

第 4 步：动手实现 (8 小时)
  └─ 实现 SDE 类
  └─ 训练 Score 网络

第 5 步：深入理解 (6 小时)
  └─ 概率流 ODE
  └─ 不同 SDE 对比
```

### 12.3 实践项目

| 项目 | 难度 | 用时 |
|------|------|------|
| Score SDE 实现 | ⭐⭐⭐⭐ | 6 小时 |
| CIFAR-10 训练 | ⭐⭐⭐⭐ | 12 小时 |
| 与 DDPM 对比 | ⭐⭐⭐⭐ | 8 小时 |
| ODE 采样实验 | ⭐⭐⭐⭐⭐ | 10 小时 |

---

## 总结

### 核心要点

1. **统一框架**: 用 SDE 统一描述生成模型
2. **分数匹配**: 学习 ∇log p(x) 指导生成
3. **连续时间**: 更优美的数学描述
4. **包含 DDPM**: DDPM 是离散化特例

### 历史地位

- **理论贡献**: 统一了多个生成模型框架
- **影响力**: 5,000+ 引用，理论基石
- **后续**: Consistency Models 等工作基础

### 学习价值

- **理论基础**: 理解现代生成模型的数学基础
- **研究价值**: 理论研究的起点
- **后续学习**: 为学习 Consistency Models 等打基础

---

**Score SDE 提供了生成模型的统一理论框架！** 📐

---

*最后更新*: 2026-04-07
