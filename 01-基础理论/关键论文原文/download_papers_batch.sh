#!/bin/bash
# AIGC 核心论文下载脚本 - 分批下载版本
# 使用方法：bash download_papers_batch.sh [批次号]
# 批次号：1, 2, 3, 4

BATCH=${1:-1}
cd /Users/wanglingfeng/Documents/AIGC/01-基础理论/关键论文原文

echo "开始下载 AIGC 核心论文 - 批次 $BATCH"
echo "=========================================="

case $BATCH in
    1)
        echo "批次 1: 基础理论 (3 篇)"
        echo "下载 VAE (2013)..."
        curl -L -o "05_VAE_2013_Kingma.pdf" "https://arxiv.org/pdf/1312.6114.pdf"
        
        echo "下载 DiT (2022)..."
        curl -L -o "04_DiT_2022_Peebles.pdf" "https://arxiv.org/pdf/2212.09748.pdf"
        
        echo "下载 Rectified Flow (2022)..."
        curl -L -o "08_RectifiedFlow_2022_Liu.pdf" "https://arxiv.org/pdf/2209.03003.pdf"
        ;;
        
    2)
        echo "批次 2: 进阶应用 (3 篇)"
        echo "下载 ControlNet (2023)..."
        curl -L -o "06_ControlNet_2023_Zhang.pdf" "https://arxiv.org/pdf/2302.05543.pdf"
        
        echo "下载 LoRA (2021)..."
        curl -L -o "07_LoRA_2021_Hu.pdf" "https://arxiv.org/pdf/2106.09685.pdf"
        
        echo "下载 Flow Matching (2022)..."
        curl -L -o "09_FlowMatching_2022_Lipman.pdf" "https://arxiv.org/pdf/2210.02747.pdf"
        ;;
        
    3)
        echo "批次 3: SOTA 模型 (3 篇)"
        echo "下载 SDXL (2023)..."
        curl -L -o "10_SDXL_2023_Podell.pdf" "https://arxiv.org/pdf/2307.01952.pdf"
        
        echo "下载 MMDiT/SD3 (2024)..."
        curl -L -o "11_MMDiT_2024_Esser.pdf" "https://arxiv.org/pdf/2403.03206.pdf"
        
        echo "下载 VAVAE (2025)..."
        curl -L -o "12_VAVAE_2025.pdf" "https://arxiv.org/pdf/2501.01423.pdf"
        ;;
        
    4)
        echo "批次 4: 前沿探索 (3 篇)"
        echo "下载 RAE (2025)..."
        curl -L -o "13_RAE_2025_Xie.pdf" "https://arxiv.org/pdf/2510.11690.pdf"
        
        echo "下载 JiT (2025)..."
        curl -L -o "14_JiT_2025_He.pdf" "https://arxiv.org/pdf/2511.13720.pdf"
        
        echo "下载 PixelDiT (2025)..."
        curl -L -o "15_PixelDiT_2025.pdf" "https://arxiv.org/pdf/2511.20645.pdf"
        ;;
        
    *)
        echo "错误：批次号必须是 1, 2, 3, 或 4"
        echo "用法：bash download_papers_batch.sh [1|2|3|4]"
        exit 1
        ;;
esac

echo ""
echo "✅ 批次 $BATCH 下载完成！"
echo ""
echo "已下载论文列表:"
ls -lh *.pdf 2>/dev/null | tail -10
