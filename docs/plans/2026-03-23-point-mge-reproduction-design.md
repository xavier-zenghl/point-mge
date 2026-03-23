# Point-MGE 论文复现设计文档

> 日期：2026-03-23
> 论文：Point-MGE: A Joint Framework of Point Cloud Representation Learning and 3D Shape Generation

## 1. 项目概述

复现 Point-MGE 论文的完整方法，包括 VQVAE 标记器训练、滑动遮蔽预训练、所有下游任务评估（分类、少样本学习、部件分割）以及 3D 形状生成。

**技术栈**：PyTorch, DDP 多卡训练, YAML 配置文件, 模块化代码组织

## 2. 项目结构

```
point-mge/
├── configs/                    # YAML配置文件
│   ├── pretrain.yaml           # 预训练配置
│   ├── vqvae.yaml              # VQVAE训练配置
│   ├── nerf.yaml               # NeRF训练配置
│   ├── cls_modelnet40.yaml     # ModelNet40分类
│   ├── cls_scanobjectnn.yaml   # ScanObjectNN分类
│   ├── fewshot.yaml            # 少样本学习
│   ├── partseg.yaml            # 部件分割
│   └── generation.yaml         # 3D生成
├── data/                       # 数据集目录
│   ├── ShapeNet55/
│   ├── ModelNet40/
│   ├── ScanObjectNN/
│   └── ShapeNetPart/
├── datasets/                   # 数据加载器
│   ├── __init__.py
│   ├── shapenet.py
│   ├── modelnet40.py
│   ├── scanobjectnn.py
│   ├── shapenetpart.py
│   └── data_utils.py           # FPS、KNN等点云工具
├── models/                     # 模型定义
│   ├── __init__.py
│   ├── point_patch_embed.py    # 点云分块嵌入（FPS+KNN+miniPointNet）
│   ├── vqvae.py                # VQVAE标记器（编码器+码本+解码器）
│   ├── nerf.py                 # NeRF模型（三平面表示）
│   ├── extractor.py            # Extractor（多尺度ViT编码器）
│   ├── generator.py            # Generator（ViT解码器）
│   ├── gpt_generator.py        # GPT自回归生成模型
│   └── heads/                  # 下游任务头
│       ├── cls_head.py
│       ├── partseg_head.py
│       └── generation_head.py
├── tools/                      # 训练/评估入口脚本
│   ├── train_nerf.py           # 阶段1：训练NeRF
│   ├── train_vqvae.py          # 阶段2：训练VQVAE
│   ├── pretrain.py             # 阶段3：预训练Extractor-Generator
│   ├── finetune_cls.py         # 分类微调
│   ├── fewshot.py              # 少样本学习评估
│   ├── finetune_partseg.py     # 部件分割微调
│   ├── train_generation.py     # 3D生成训练
│   └── eval_generation.py      # 生成评估
├── utils/                      # 工具函数
│   ├── __init__.py
│   ├── config.py               # 配置解析
│   ├── logger.py               # 日志
│   ├── distributed.py          # DDP工具
│   ├── metrics.py              # 评估指标
│   ├── scheduler.py            # 学习率调度
│   └── checkpoint.py           # 检查点管理
├── scripts/                    # 数据下载/处理脚本
│   ├── download_shapenet.sh
│   ├── download_modelnet40.sh
│   ├── download_scanobjectnn.sh
│   └── prepare_nerf_data.py
├── paper/
│   └── point-mge.pdf
├── requirements.txt
└── README.md
```

## 3. 核心模型架构

### 3.1 Point Patch Embedding

- 输入：N=2048 个点的点云 (B, 2048, 3)
- FPS 采样 G=64 个中心点
- KNN 分组：每个中心点找 K=32 个邻近点
- mini-PointNet：对每个 patch 内32个点做特征提取 → (B, 64, C), C=384

### 3.2 VQVAE 标记器

1. ViT 编码器：patch embeddings (B, 64, C) → latent (B, 64, D)
2. 向量量化：码本大小 K=8192，EMA 更新
3. ViT 解码器：量化特征 → 重建 NeRF 三平面特征
4. 损失：重建损失 + commitment loss (β=0.25) + codebook loss

### 3.3 NeRF 三平面表示

- 每个 ShapeNet 物体训练一个三平面 NeRF
- 提取三平面特征 (3, H, W, F) 作为 VQVAE 重建目标
- 渲染损失（RGB + 密度）+ 特征对齐损失

### 3.4 Extractor（预训练编码器）

- 多尺度 ViT 架构
- 处理被遮蔽后的可见 token
- 滑动遮蔽比率：r = 1 - β^(epoch/total_epoch * u)，β=0.5, u=2

### 3.5 Generator（预训练解码器）

- 轻量级 ViT 解码器
- 预测被遮蔽位置的 VQVAE token indices
- 交叉熵损失

### 3.6 GPT 生成器

- 自回归 Transformer
- 支持无条件/类别条件生成
- top-k/top-p 采样

## 4. 训练流水线

### 阶段1：NeRF 训练

- 数据：ShapeNet 多视角渲染图 + 相机参数
- 方法：每个物体训练一个三平面 NeRF
- 输出：保存三平面特征到磁盘

### 阶段2：VQVAE 训练

- 输入：点云 → Patch Embedding
- 目标：NeRF 三平面特征
- 优化器：AdamW, lr=1e-3, weight_decay=0.05
- 训练：~300 epochs on ShapeNet

### 阶段3：预训练 Extractor-Generator

- 滑动遮蔽 + 离散 token 预测
- 优化器：AdamW, lr=1e-3, cosine schedule
- 训练：300 epochs on ShapeNet

### 阶段4：下游微调与生成

| 任务 | 数据集 | 关键超参数 |
|------|--------|-----------|
| 形状分类 | ModelNet40 | lr=5e-4, 300 epochs |
| 形状分类 | ScanObjectNN (3变体) | lr=5e-4, 300 epochs |
| 少样本学习 | ModelNet40 | k-way n-shot, 10轮平均 |
| 部件分割 | ShapeNetPart | lr=2e-4, 300 epochs |
| 3D 生成 | ShapeNet | GPT + top-k sampling |

## 5. 评估指标

- **分类**：Overall Accuracy (OA)，投票策略
- **少样本**：k-way n-shot 准确率 ± 标准差
- **部件分割**：class mIoU / instance mIoU
- **生成**：COV, MMD, 1-NNA（CD/EMD距离）
- **NeRF**：PSNR / SSIM

## 6. 多卡训练

- `torchrun` 启动 DDP
- SyncBatchNorm
- 梯度同步 + all_reduce 指标聚合
- 所有阶段统一支持

## 7. 依赖

```
torch >= 2.0
torchvision
timm
einops
pointnet2_ops
pytorch3d
open3d
numpy, scipy
tensorboard / wandb
easydict, pyyaml
tqdm
```

## 8. 错误处理

- CUDA OOM：梯度累积支持
- 断点续训：检查点保存/恢复
- 数据加载失败：友好提示
- 分布式异常：graceful shutdown
