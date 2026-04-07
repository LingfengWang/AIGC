# AIGC 工具框架汇总

**最后更新**: 2026-04-04  
**状态**: 持续更新

---

## 🛠️ 工具框架概览

```
┌─────────────────────────────────────────────────────────────┐
│                    AIGC 工具链                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  训练框架                 推理部署                            │
│  ├─ PyTorch              ├─ vLLM                            │
│  ├─ DeepSpeed            ├─ TensorRT                        │
│  ├─ Megatron-LM          ├─ ONNX Runtime                    │
│  └─ FSDP                 └─ OpenVINO                        │
│                                                             │
│  开发工具                 评测工具                            │
│  ├─ Diffusers            ├─ CLIP Score                      │
│  ├─ Transformers         ├─ FID                             │
│  ├─ LangChain            ├─ IS                              │
│  └─ LlamaIndex           └─ Human Preference                │
│                                                             │
│  WebUI                    工作流                              │
│  ├─ SD WebUI             ├─ ComfyUI                         │
│  ├─ Fooocus              ├─ InvokeAI                        │
│  └─ Gradio               └─ Node-RED                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🏋️ 训练框架

### PyTorch

| 属性 | 值 |
|------|-----|
| **发布** | 2016 年 (Facebook) |
| **语言** | Python/C++ |
| **特点** | 动态图、易用 |
| **生态** | 最丰富 |
| **官网** | https://pytorch.org/ |

**安装**:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**适用场景**:
- ✅ 研究和开发
- ✅ 原型验证
- ✅ 大多数 AIGC 项目

---

### DeepSpeed

| 属性 | 值 |
|------|-----|
| **发布** | 2020 年 (Microsoft) |
| **特点** | ZeRO 优化、3D 并行 |
| **规模** | 万亿参数模型 |
| **官网** | https://www.deepspeed.ai/ |

**核心特性**:
```
ZeRO (Zero Redundancy Optimizer)
├─ ZeRO-1: 优化器状态分片
├─ ZeRO-2: 梯度分片
├─ ZeRO-3: 参数分片
└─ ZeRO-Offload: CPU 卸载
```

**安装**:
```bash
pip install deepspeed
```

**适用场景**:
- ✅ 大模型训练
- ✅ 多卡/多节点
- ✅ 显存受限场景

---

### Megatron-LM

| 属性 | 值 |
|------|-----|
| **发布** | 2019 年 (NVIDIA) |
| **特点** | 张量并行、流水线并行 |
| **规模** | 超大规模 |
| **官网** | https://github.com/NVIDIA/Megatron-LM |

**核心特性**:
- 张量并行 (Tensor Parallel)
- 流水线并行 (Pipeline Parallel)
- 数据并行 (Data Parallel)
- 3D 并行组合

**适用场景**:
- ✅ 超大规模模型
- ✅ GPU 集群
- ✅ 生产环境

---

### FSDP (Fully Sharded Data Parallel)

| 属性 | 值 |
|------|-----|
| **发布** | 2021 年 (Facebook) |
| **集成** | PyTorch 原生 |
| **特点** | 参数分片、易用 |
| **文档** | https://pytorch.org/docs/stable/fsdp.html |

**使用示例**:
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(model, device_id=0)
```

**适用场景**:
- ✅ PyTorch 原生项目
- ✅ 中等规模模型
- ✅ 简化分布式训练

---

## 🚀 推理部署

### vLLM

| 属性 | 值 |
|------|-----|
| **发布** | 2023 年 (UC Berkeley) |
| **特点** | PagedAttention、高吞吐 |
| **性能** | 2-24x 提升 |
| **官网** | https://vllm.ai/ |

**安装**:
```bash
pip install vllm
```

**使用**:
```python
from vllm import LLM

llm = LLM(model="facebook/opt-125m")
output = llm.generate("Hello, my name is")
```

**适用场景**:
- ✅ 高并发服务
- ✅ LLM 推理
- ✅ 生产部署

---

### TensorRT

| 属性 | 值 |
|------|-----|
| **发布** | NVIDIA |
| **特点** | 图优化、量化、多流 |
| **性能** | 2-10x 提升 |
| **官网** | https://developer.nvidia.com/tensorrt |

**安装**:
```bash
pip install tensorrt
```

**适用场景**:
- ✅ NVIDIA GPU
- ✅ 生产部署
- ✅ 延迟敏感场景

---

### ONNX Runtime

| 属性 | 值 |
|------|-----|
| **发布** | Microsoft |
| **特点** | 跨平台、多后端 |
| **支持** | CPU/GPU/NPU |
| **官网** | https://onnxruntime.ai/ |

**安装**:
```bash
pip install onnxruntime-gpu  # GPU 版本
pip install onnxruntime      # CPU 版本
```

**适用场景**:
- ✅ 跨平台部署
- ✅ 模型转换
- ✅ 边缘设备

---

### OpenVINO

| 属性 | 值 |
|------|-----|
| **发布** | Intel |
| **特点** | Intel 硬件优化 |
| **支持** | CPU/GPU/VPU |
| **官网** | https://docs.openvino.ai/ |

**适用场景**:
- ✅ Intel 硬件
- ✅ 边缘部署
- ✅ 低功耗场景

---

## 💻 开发工具

### Diffusers

| 属性 | 值 |
|------|-----|
| **发布** | 2022 年 (Hugging Face) |
| **特点** | 统一 API、模型丰富 |
| **支持** | SD、DDPM、Score SDE |
| **官网** | https://huggingface.co/docs/diffusers |

**安装**:
```bash
pip install diffusers transformers accelerate
```

**使用示例**:
```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
image = pipe("a cute cat").images[0]
```

**适用场景**:
- ✅ 扩散模型开发
- ✅ 快速原型
- ✅ 模型微调

---

### Transformers

| 属性 | 值 |
|------|-----|
| **发布** | 2018 年 (Hugging Face) |
| **模型** | 10 万 + |
| **支持** | NLP/CV/Audio |
| **官网** | https://huggingface.co/docs/transformers |

**安装**:
```bash
pip install transformers
```

**适用场景**:
- ✅ 预训练模型使用
- ✅ 微调
- ✅ 多模态项目

---

### LangChain

| 属性 | 值 |
|------|-----|
| **发布** | 2022 年 |
| **特点** | LLM 应用开发框架 |
| **功能** | Chain、Agent、Memory |
| **官网** | https://www.langchain.com/ |

**安装**:
```bash
pip install langchain langchain-community
```

**适用场景**:
- ✅ LLM 应用开发
- ✅ RAG 系统
- ✅ Agent 系统

---

### LlamaIndex

| 属性 | 值 |
|------|-----|
| **发布** | 2022 年 |
| **特点** | 数据索引、RAG |
| **功能** | 文档加载、索引、查询 |
| **官网** | https://www.llamaindex.ai/ |

**适用场景**:
- ✅ 知识库问答
- ✅ 文档检索
- ✅ RAG 系统

---

## 🎨 WebUI 工具

### Stable Diffusion WebUI (AUTOMATIC1111)

| 属性 | 值 |
|------|-----|
| **GitHub** | https://github.com/AUTOMATIC1111/stable-diffusion-webui |
| **Stars** | 120k+ |
| **特点** | 功能最全、插件丰富 |
| **难度** | 中等 |

**安装**:
```bash
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui
cd stable-diffusion-webui
./webui.sh  # Linux/Mac
webui.bat   # Windows
```

**核心功能**:
- 文生图/图生图
- Inpaint/Outpaint
- ControlNet
- LoRA 训练
- 批量处理
- 插件系统

**适用场景**:
- ✅ 高级用户
- ✅ 需要完整功能
- ✅ 本地部署

---

### ComfyUI

| 属性 | 值 |
|------|-----|
| **GitHub** | https://github.com/comfyanonymous/ComfyUI |
| **Stars** | 50k+ |
| **特点** | 节点式工作流、高效 |
| **难度** | 较高 |

**安装**:
```bash
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
python main.py
```

**核心功能**:
- 节点式工作流
- 自定义节点
- 高效内存管理
- API 支持

**适用场景**:
- ✅ 专业用户
- ✅ 复杂工作流
- ✅ 批量生产

---

### Fooocus

| 属性 | 值 |
|------|-----|
| **GitHub** | https://github.com/lllyasviel/Fooocus |
| **Stars** | 40k+ |
| **特点** | 简单易用、开箱即用 |
| **难度** | 低 |

**安装**:
```bash
git clone https://github.com/lllyasviel/Fooocus
cd Fooocus
python entry_with_update.py
```

**适用场景**:
- ✅ 新手用户
- ✅ 快速上手
- ✅ 简洁界面

---

### InvokeAI

| 属性 | 值 |
|------|-----|
| **GitHub** | https://github.com/invoke-ai/InvokeAI |
| **Stars** | 20k+ |
| **特点** | 专业编辑功能 |
| **难度** | 中等 |

**适用场景**:
- ✅ 设计师
- ✅ 专业编辑
- ✅ 商业使用

---

## 📊 评测工具

### CLIP Score

**用途**: 评估图文匹配度

**计算**:
```python
import clip
import torch

model, preprocess = clip.load("ViT-B/32")
image = preprocess(Image.open("output.png")).unsqueeze(0)
text = clip.tokenize(["a cute cat"])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    similarity = (image_features @ text_features.T).softmax(dim=-1)
```

---

### FID (Fréchet Inception Distance)

**用途**: 评估生成图像质量

**安装**:
```bash
pip install pytorch-fid
```

**使用**:
```bash
python -m pytorch_fid path/to/real path/to/generated
```

**说明**: 越低越好，通常 < 10 为优秀

---

### IS (Inception Score)

**用途**: 评估生成图像多样性和质量

**安装**:
```bash
pip install torch-fidelity
```

**使用**:
```bash
fidelity --gpu 0 --is path/to/generated
```

**说明**: 越高越好

---

### Human Preference

**用途**: 人工评估

**方法**:
- A/B 测试
-  Likert 量表 (1-5 分)
-  pairwise comparison

**平台**:
- [Scale AI](https://scale.com/)
- [Amazon Mechanical Turk](https://www.mturk.com/)

---

## 🔧 辅助工具

### 模型管理

| 工具 | 用途 | 链接 |
|------|------|------|
| HuggingFace Hub | 模型托管 | https://huggingface.co/ |
| Weights & Biases | 实验跟踪 | https://wandb.ai/ |
| MLflow | 实验管理 | https://mlflow.org/ |

---

### 数据处理

| 工具 | 用途 | 链接 |
|------|------|------|
| Label Studio | 数据标注 | https://labelstud.io/ |
| CVAT | 图像标注 | https://www.cvat.ai/ |
| DVC | 数据版本 | https://dvc.org/ |

---

### 性能分析

| 工具 | 用途 | 链接 |
|------|------|------|
| PyTorch Profiler | 性能分析 | PyTorch 内置 |
| NVIDIA Nsight | GPU 分析 | NVIDIA |
| TensorBoard | 可视化 | TensorFlow/PyTorch |

---

## 📦 部署方案

### 本地部署

```bash
# 使用 Docker
docker run --gpus all -p 7860:7860 ghcr.io/automatic1111/stable-diffusion-webui:latest

# 使用 Conda
conda create -n aigc python=3.10
conda activate aigc
pip install torch torchvision diffusers
```

### 云端部署

| 平台 | 价格 | 特点 |
|------|------|------|
| RunPod | $0.40/小时 | 便宜、灵活 |
| Vast.ai | $0.20/小时 | 最便宜 |
| Lambda Labs | $0.50/小时 | 稳定 |
| AWS | $3+/小时 | 企业级 |

### API 服务

| 平台 | 价格 | 特点 |
|------|------|------|
| Replicate | $0.002/秒 | 按使用付费 |
| HuggingFace Inference | 免费/$9/月 | 简单 |
| Stability AI API | $0.002/图 | 官方 |

---

## 🔮 趋势与建议

### 2026 趋势

1. **推理优化**
   - 量化普及 (INT8/INT4)
   - 蒸馏技术成熟
   - 一步生成实用化

2. **端侧部署**
   - 手机端运行
   - 边缘计算
   - 隐私保护

3. **工具整合**
   - 一体化平台
   - 工作流自动化
   - 低代码开发

4. **标准化**
   - 模型格式统一
   - API 标准化
   - 评测基准规范

### 建议

1. **选择合适的工具**
   - 研究：PyTorch + Diffusers
   - 原型：WebUI
   - 生产：TensorRT + vLLM

2. **关注性能优化**
   - 量化
   - 蒸馏
   - 批处理

3. **建立工作流**
   - 版本控制
   - 自动化测试
   - 持续集成

---

**持续更新中...**

*最后更新*: 2026-04-04
