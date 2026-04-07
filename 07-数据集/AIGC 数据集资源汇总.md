# AIGC 数据集资源汇总

**最后更新**: 2026-04-04  
**状态**: 持续更新

---

## 📊 数据集概览

```
┌────────────────────────────────────────────────────────────┐
│                    AIGC 数据集分类                           │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  文本数据集              图像数据集                          │
│  ├─ CommonCrawl         ├─ ImageNet                        │
│  ├─ LAION-Text          ├─ COCO                            │
│  └─ FineWeb             └─ LAION-Image                     │
│                                                            │
│  图文对数据集            视频数据集                          │
│  ├─ LAION-400M          ├─ WebVid                         │
│  ├─ LAION-2B            ├─ Kinetics                        │
│  ├─ DataComp            └─ HowTo100M                       │
│  └─ SA-1B               ──────────────────                 │
│                                                            │
│  音频数据集              多模态数据集                        │
│  ├─ LibriSpeech         ├─ LAION-Aesthetics                │
│  ├─ AudioSet            ├─ Multimodal-Instruction          │
│  └─ Common Voice        └─ LLaVA-Data                      │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 📝 文本数据集

### CommonCrawl

| 属性 | 值 |
|------|-----|
| **规模** | 数 PB 网页数据 |
| **语言** | 多语言 |
| **更新频率** | 每月 |
| **许可** | 各网页原有许可 |
| **用途** | LLM 预训练 |

**下载**: https://commoncrawl.org/

**处理工具**:
```python
# 使用 langdetect 过滤语言
# 使用 fasttext 质量过滤
# 使用 dedup 去重
```

---

### FineWeb

| 属性 | 值 |
|------|-----|
| **规模** | 15TB 文本 |
| **语言** | 英文为主 |
| **来源** | CommonCrawl 清洗 |
| **许可** | ODC-By |
| **用途** | LLM 训练 |

**特点**:
- 高质量网页文本
- 严格的质量过滤
- 去重处理

**下载**: https://huggingface.co/datasets/HuggingFaceFW/fineweb

---

## 🖼️ 图像数据集

### ImageNet

| 属性 | 值 |
|------|-----|
| **规模** | 1400 万图像 |
| **类别** | 21,841 类 |
| **分辨率** | 可变 |
| **许可** | 学术使用 |
| **用途** | 分类模型训练 |

**下载**: https://www.image-net.org/

**变体**:
- ImageNet-1K (100 万，最常用)
- ImageNet-21K (全量)

---

### COCO (Common Objects in Context)

| 属性 | 值 |
|------|-----|
| **规模** | 33 万图像 |
| **标注** | 检测、分割、caption |
| **类别** | 80 类 |
| **许可** | CC BY 4.0 |
| **用途** | 多任务训练 |

**下载**: https://cocodataset.org/

**标注类型**:
```
├─ 目标检测 (91 类)
├─ 实例分割
├─ 关键点检测
├─ 图像描述 (5 个 caption/图)
└─ 全景分割
```

---

### LAION 系列

#### LAION-400M

| 属性 | 值 |
|------|-----|
| **规模** | 4 亿图文对 |
| **来源** | CommonCrawl |
| **过滤** | CLIP 相似度 > 0.28 |
| **许可** | 多种 CC 许可 |
| **用途** | 文生图模型训练 |

**下载**: https://laion.ai/blog/laion-400-open-dataset/

---

#### LAION-2B

| 属性 | 值 |
|------|-----|
| **规模** | 20 亿图文对 |
| **来源** | CommonCrawl |
| **过滤** | CLIP 相似度 > 0.3 |
| **许可** | 多种 CC 许可 |
| **用途** | 大规模文生图训练 |

**子集**:
- LAION-2B-en (英文，约 10 亿)
- LAION-2B-multi (多语言，约 10 亿)

**下载**: https://laion.ai/blog/laion-5b/

---

#### LAION-Aesthetics

| 属性 | 值 |
|------|-----|
| **规模** | 6 亿图像 |
| **过滤** | 美学评分 > 5.0 |
| **来源** | LAION-2B |
| **用途** | 高质量图像生成 |

**特点**:
- SD v1/v2 训练数据
- 美学评分过滤
- 水印过滤

---

### DataComp

| 属性 | 值 |
|------|-----|
| **规模** | 14 亿候选图像 |
| **来源** | CommonCrawl |
| **特点** | 基准测试平台 |
| **许可** | 多种 CC 许可 |
| **用途** | 对比学习研究 |

**下载**: https://www.datacomp.ai/

**特点**:
- 统一基准
- 多赛道竞争
- 可复现评估

---

### SA-1B (Segment Anything)

| 属性 | 值 |
|------|-----|
| **规模** | 1100 万图像 |
| **标注** | 10 亿掩码 |
| **来源** | 专业标注 |
| **许可** | Apache 2.0 |
| **用途** | 分割模型训练 |

**下载**: https://segment-anything.com/

**特点**:
- 大规模分割标注
- SAM 模型训练数据
- 高质量掩码

---

## 🎬 视频数据集

### WebVid

| 属性 | 值 |
|------|-----|
| **规模** | 1000 万视频 |
| **来源** | 网页爬取 |
| **标注** | 文本描述 |
| **许可** | 各视频原有许可 |
| **用途** | 文生视频训练 |

**版本**:
- WebVid-2M (200 万)
- WebVid-10M (1000 万)

**下载**: https://max-bain.com/webvid-dataset/

---

### Kinetics

| 属性 | 值 |
|------|-----|
| **规模** | 70 万视频 |
| **类别** | 400/600/700 类 |
| **时长** | 10 秒/段 |
| **许可** | YouTube 许可 |
| **用途** | 视频分类 |

**版本**:
- Kinetics-400
- Kinetics-600
- Kinetics-700

**下载**: https://deepmind.com/research/open-source/kinetics

---

### HowTo100M

| 属性 | 值 |
|------|-----|
| **规模** | 1.36 亿视频片段 |
| **时长** | 200 万小时 |
| **来源** | YouTube 教程 |
| **许可** | YouTube 许可 |
| **用途** | 视频 - 语言学习 |

**下载**: https://www.di.ens.fr/willow/research/howto100m/

---

## 🎵 音频数据集

### LibriSpeech

| 属性 | 值 |
|------|-----|
| **规模** | 1000 小时 |
| **内容** | 有声书朗读 |
| **语言** | 英语 |
| **许可** | Public Domain |
| **用途** | 语音识别 |

**下载**: https://www.openslr.org/12/

---

### AudioSet

| 属性 | 值 |
|------|-----|
| **规模** | 200 万音频片段 |
| **类别** | 527 类 |
| **时长** | 10 秒/段 |
| **来源** | YouTube |
| **用途** | 音频分类 |

**下载**: https://research.google.com/audioset/

---

### Common Voice

| 属性 | 值 |
|------|-----|
| **规模** | 3 万 + 小时 |
| **语言** | 100+ 语言 |
| **来源** | 众包 |
| **许可** | CC0 |
| **用途** | 多语言语音识别 |

**下载**: https://commonvoice.mozilla.org/

---

## 🔀 多模态数据集

### LAION-Aesthetics v2

| 属性 | 值 |
|------|-----|
| **规模** | 6 亿图文对 |
| **过滤** | 美学评分 + 水印 + NSFW |
| **用途** | SD v2 训练 |

---

### LLaVA-Data

| 属性 | 值 |
|------|-----|
| **规模** | 75 万指令数据 |
| **来源** | GPT-4 生成 |
| **用途** | 多模态对话 |

**下载**: https://github.com/haotian-liu/LLaVA

---

## 📦 具身智能数据集

### RoboTwin

| 属性 | 值 |
|------|-----|
| **规模** | 仿真数据 |
| **任务** | 双臂操作 |
| **用途** | VLA 训练 |

---

### AgiBot World

| 属性 | 值 |
|------|-----|
| **规模** | 百万级真机数据 |
| **来源** | 智元机器人 |
| **用途** | 具身智能训练 |

---

### RT-1 / RT-2

| 属性 | 值 |
|------|-----|
| **规模** | 100 万 + 轨迹 |
| **来源** | Google |
| **用途** | 机器人操作 |

**下载**: https://robotics-transformer-x.github.io/

---

## 🔍 数据集搜索工具

### 通用搜索

| 工具 | 链接 | 说明 |
|------|------|------|
| HuggingFace Datasets | https://huggingface.co/datasets | 最大数据集平台 |
| Kaggle Datasets | https://www.kaggle.com/datasets | 社区数据集 |
| Google Dataset Search | https://datasetsearch.research.google.com | 数据集搜索引擎 |
| Papers With Code | https://paperswithcode.com/datasets | 论文关联数据集 |

---

## 📊 数据集选择指南

### 根据任务选择

| 任务 | 推荐数据集 | 规模要求 |
|------|-----------|---------|
| 文生图预训练 | LAION-2B | 10 亿 + |
| 文生图微调 | COCO + 自定义 | 1 万 + |
| 图像分类 | ImageNet-1K | 100 万 |
| 目标检测 | COCO | 10 万 + |
| 图像分割 | SA-1B | 1000 万 |
| 视频生成 | WebVid-10M | 1000 万 |
| 语音识别 | LibriSpeech | 100 小时 + |

### 根据资源选择

| GPU 资源 | 推荐数据集 | 训练时间 |
|---------|-----------|---------|
| 单卡 (24GB) | 小型子集 (10 万) | 1-2 天 |
| 8 卡 (A100) | 中型 (1000 万) | 1-2 周 |
| 多节点 | 大型 (10 亿 +) | 1-2 月 |

---

## ⚖️ 许可说明

### 常见许可类型

| 许可 | 商用 | 修改 | 分发 | 备注 |
|------|------|------|------|------|
| CC0 | ✅ | ✅ | ✅ | 公共领域 |
| CC BY | ✅ | ✅ | ✅ | 需署名 |
| CC BY-SA | ✅ | ✅ | ✅ | 需署名 + 相同许可 |
| CC BY-NC | ❌ | ✅ | ✅ | 非商业 |
| CC BY-ND | ✅ | ❌ | ✅ | 禁止演绎 |
| Apache 2.0 | ✅ | ✅ | ✅ | 专利授权 |
| MIT | ✅ | ✅ | ✅ | 最宽松 |

### 使用注意事项

1. **检查许可**: 使用前务必确认许可条款
2. **署名要求**: CC BY 需要署名
3. **商业限制**: NC 许可不可商用
4. **衍生作品**: SA 许可需相同许可分发
5. **隐私问题**: 人脸等敏感数据需额外注意

---

## 🛠️ 数据处理工具

### 下载工具

```bash
# HuggingFace datasets
pip install datasets
python -c "from datasets import load_dataset; load_dataset('laion/laion2B-en')"

# Kaggle
pip install kaggle
kaggle datasets download -d dataset-name

# AWS S3 (LAION)
aws s3 cp s3://laion-dataset/ ./local/ --recursive
```

### 处理工具

```python
# 图像处理
from PIL import Image
import cv2

# 数据清洗
from datasketch import MinHash  # 去重
from langdetect import detect   # 语言检测

# 质量过滤
import clip  # CLIP 评分
```

### 数据增强

```python
# 图像增强
import albumentations as A

transform = A.Compose([
    A.RandomResizedCrop(512, 512),
    A.HorizontalFlip(),
    A.ColorJitter(),
])

# 文本增强
from nlpaug import augmentor
```

---

## 📈 趋势与建议

### 2026 趋势

1. **高质量 > 大规模**
   - 从追求规模转向质量
   - 精细清洗和标注
   - 合成数据补充

2. **多模态融合**
   - 图文音视频统一
   - 跨模态对齐
   - 统一表示学习

3. **合成数据**
   - AI 生成训练数据
   - 解决长尾问题
   - 隐私保护

4. **合规性加强**
   - 版权审查严格
   - 隐私保护加强
   - 许可规范化

### 建议

1. **优先使用公开数据集**
   - 避免版权问题
   - 可复现性好
   - 社区支持好

2. **建立自己的数据集**
   - 针对特定场景
   - 持续积累
   - 注意标注质量

3. **数据版本管理**
   - 使用 DVC 等工具
   - 记录数据来源
   - 便于复现

---

**持续更新中...**

*最后更新*: 2026-04-04
