#!/bin/bash
# AIGC 核心论文下载脚本
# 使用方法：bash download_papers.sh

cd /Users/wanglingfeng/Documents/AIGC/01-基础理论/关键论文原文

echo "开始下载 AIGC 核心论文..."

# 基础理论 (🔴 必读)
echo "下载 DDPM (2020)..."
curl -L -o "01_DDPM_2020_Ho.pdf" "https://arxiv.org/pdf/2006.11239.pdf"

echo "下载 DDIM (2020)..."
curl -L -o "02_DDIM_2020_Song.pdf" "https://arxiv.org/pdf/2010.02502.pdf"

echo "下载 LDM/Stable Diffusion (2021)..."
curl -L -o "03_LDM_2021_Rombach.pdf" "https://arxiv.org/pdf/2112.10752.pdf"

echo "下载 DiT (2022)..."
curl -L -o "04_DiT_2022_Peebles.pdf" "https://arxiv.org/pdf/2212.09748.pdf"

echo "下载 VAE (2013)..."
curl -L -o "05_VAE_2013_Kingma.pdf" "https://arxiv.org/pdf/1312.6114.pdf"

# 进阶应用 (🟡 必读)
echo "下载 ControlNet (2023)..."
curl -L -o "06_ControlNet_2023_Zhang.pdf" "https://arxiv.org/pdf/2302.05543.pdf"

echo "下载 LoRA (2021)..."
curl -L -o "07_LoRA_2021_Hu.pdf" "https://arxiv.org/pdf/2106.09685.pdf"

echo "下载 Rectified Flow (2022)..."
curl -L -o "08_RectifiedFlow_2022_Liu.pdf" "https://arxiv.org/pdf/2209.03003.pdf"

echo "下载 Flow Matching (2022)..."
curl -L -o "09_FlowMatching_2022_Lipman.pdf" "https://arxiv.org/pdf/2210.02747.pdf"

echo "下载 SDXL (2023)..."
curl -L -o "10_SDXL_2023_Podell.pdf" "https://arxiv.org/pdf/2307.01952.pdf"

# 前沿探索 (🟢 科研)
echo "下载 MMDiT/SD3 (2024)..."
curl -L -o "11_MMDiT_2024_Esser.pdf" "https://arxiv.org/pdf/2403.03206.pdf"

echo "下载 VAVAE (2025)..."
curl -L -o "12_VAVAE_2025.pdf" "https://arxiv.org/pdf/2501.01423.pdf"

echo "下载 RAE (2025)..."
curl -L -o "13_RAE_2025_Xie.pdf" "https://arxiv.org/pdf/2510.11690.pdf"

echo "下载 JiT (2025)..."
curl -L -o "14_JiT_2025_He.pdf" "https://arxiv.org/pdf/2511.13720.pdf"

echo "下载 PixelDiT (2025)..."
curl -L -o "15_PixelDiT_2025.pdf" "https://arxiv.org/pdf/2511.20645.pdf"

echo ""
echo "✅ 论文下载完成！"
echo ""
echo "已下载论文列表:"
ls -lh *.pdf
