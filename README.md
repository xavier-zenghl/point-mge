# Point-MGE: A Joint Framework of Point Cloud Representation Learning and 3D Shape Generation

[![arXiv](https://img.shields.io/badge/arXiv-Point--MGE-b31b1b.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-3776AB.svg)](https://www.python.org/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-EE4C2C.svg)](https://pytorch.org/)

## Abstract

This repository is an unofficial reproduction of **Point-MGE**, a joint framework that unifies point cloud representation learning and 3D shape generation. Point-MGE introduces a VQVAE tokenizer targeting NeRF triplane features, a sliding-mask pretraining strategy with an Extractor-Generator architecture, and an autoregressive GPT generator for 3D shape synthesis. The pretrained Extractor transfers to downstream tasks including shape classification, few-shot learning, and part segmentation.

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0
- CUDA 11.8+

```bash
# Clone repository
git clone https://github.com/your-username/point-mge.git
cd point-mge

# Create conda environment
conda create -n point-mge python=3.10
conda activate point-mge

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Install pointnet2_ops (requires CUDA toolkit)
pip install git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#subdirectory=pointnet2_ops_lib

# Install package
pip install -e .
```

## Data Preparation

### 1. ShapeNet55 (Pretraining)

Download ShapeNet55 point clouds and place them under `data/ShapeNet55/`:

```text
data/ShapeNet55/
├── shapenet_pc/           # .npy files, each (8192, 3)
├── train.txt              # taxonomy_id-model_id per line
└── test.txt
```

### 2. ModelNet40 (Classification)

```text
data/ModelNet40/
├── modelnet40_pc/         # .npy files, each (10000, 3)
├── train.txt              # filename<TAB>label per line
└── test.txt
```

### 3. ScanObjectNN (Classification)

Download the official h5 files and place them under `data/ScanObjectNN/`:

```text
data/ScanObjectNN/
├── training_objectdataset_augmentedrot_scale75.h5  # hardest variant
├── test_objectdataset_augmentedrot_scale75.h5
├── training_objectdataset.h5                       # obj_bg variant
├── test_objectdataset.h5
├── training_objectdataset_augmented25rot.h5         # obj_only variant
└── test_objectdataset_augmented25rot.h5
```

### 4. ShapeNetPart (Part Segmentation)

```text
data/ShapeNetPart/
├── synsetoffset2category.txt
├── train_test_split/
│   ├── shuffled_train_file_list.json
│   ├── shuffled_val_file_list.json
│   └── shuffled_test_file_list.json
└── <synset_id>/           # per-category point cloud .txt files
```

## Training

### Full Pipeline (4 Stages)

```bash
# Stage 1: Train per-object NeRF and extract triplane features
python tools/train_nerf.py --config configs/nerf.yaml

# Stage 2: Train VQVAE tokenizer (targets: NeRF triplane features)
python tools/train_vqvae.py --config configs/vqvae.yaml
# Multi-GPU:
torchrun --nproc_per_node=4 tools/train_vqvae.py --config configs/vqvae.yaml

# Stage 3: Pretrain Extractor-Generator with sliding masking
python tools/pretrain.py --config configs/pretrain.yaml \
    --vqvae_ckpt experiments/vqvae/final.pth
# Multi-GPU:
torchrun --nproc_per_node=4 tools/pretrain.py --config configs/pretrain.yaml \
    --vqvae_ckpt experiments/vqvae/final.pth

# Stage 4: Downstream tasks (see below)
```

### Downstream Tasks

```bash
# Classification on ModelNet40
python tools/finetune_cls.py --config configs/cls_modelnet40.yaml

# Classification on ScanObjectNN (hardest)
python tools/finetune_cls.py --config configs/cls_scanobjectnn.yaml

# Few-shot learning evaluation
python tools/fewshot.py --config configs/fewshot.yaml \
    --ckpt experiments/pretrain/final.pth

# Part segmentation on ShapeNetPart
python tools/finetune_partseg.py --config configs/partseg.yaml

# 3D shape generation (GPT)
python tools/train_generation.py --config configs/generation.yaml \
    --vqvae_ckpt experiments/vqvae/final.pth
```

## Evaluation

```bash
# Classification evaluation (voting)
python tools/finetune_cls.py --config configs/cls_modelnet40.yaml \
    --resume experiments/cls_modelnet40/best.pth

# Generation quality (COV, MMD, 1-NNA)
python tools/eval_generation.py --config configs/generation.yaml \
    --gpt_ckpt experiments/generation/final.pth \
    --vqvae_ckpt experiments/vqvae/final.pth \
    --num_generate 200
```

## Project Structure

```text
point-mge/
├── models/                    # Model definitions
│   ├── point_patch_embed.py   # FPS + KNN + mini-PointNet + Morton sort
│   ├── vqvae.py               # VQVAE tokenizer (VQ layer + encoder/decoder)
│   ├── nerf.py                # Triplane NeRF representation
│   ├── extractor.py           # Multi-scale ViT encoder (pretraining)
│   ├── generator.py           # ViT decoder for masked token prediction
│   ├── gpt_generator.py       # Autoregressive GPT for 3D generation
│   ├── masking.py             # Sliding masking strategy
│   └── heads/                 # Downstream task heads
│       ├── cls_head.py        # Classification head
│       └── partseg_head.py    # Part segmentation head
├── datasets/                  # Data loading
│   ├── data_utils.py          # FPS, KNN, Morton sort, augmentation
│   ├── shapenet.py            # ShapeNet55 dataset
│   ├── modelnet40.py          # ModelNet40 dataset
│   ├── scanobjectnn.py        # ScanObjectNN dataset (3 variants)
│   └── shapenetpart.py        # ShapeNetPart dataset
├── tools/                     # Training & evaluation scripts
│   ├── train_nerf.py          # Stage 1: NeRF training
│   ├── train_vqvae.py         # Stage 2: VQVAE training
│   ├── pretrain.py            # Stage 3: Extractor-Generator pretraining
│   ├── finetune_cls.py        # Classification finetuning
│   ├── fewshot.py             # Few-shot evaluation
│   ├── finetune_partseg.py    # Part segmentation finetuning
│   ├── train_generation.py    # GPT generation training
│   └── eval_generation.py     # Generation evaluation
├── utils/                     # Utilities
│   ├── config.py              # YAML config with CLI overrides
│   ├── logger.py              # Logging
│   ├── checkpoint.py          # Checkpoint save/load
│   ├── scheduler.py           # Cosine warmup LR scheduler
│   ├── distributed.py         # DDP utilities
│   └── metrics.py             # Accuracy, IoU, CD, COV, MMD, 1-NNA
├── configs/                   # YAML configuration files
├── tests/                     # Unit & integration tests (82 tests)
└── paper/                     # Original paper PDF
```

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| N (input points) | 2048 | Input point cloud size |
| G (num groups) | 64 | Number of patches (FPS centers) |
| K (group size) | 32 | Points per patch (KNN neighbors) |
| D (embed dim) | 384 | Transformer embedding dimension |
| Codebook K | 8192 | VQVAE codebook size |
| Mask beta | 0.5 | Sliding mask base ratio |
| Mask u | 2.0 | Sliding mask exponent |
| Extractor depth | 12 | ViT encoder layers |
| Generator depth | 4 | ViT decoder layers |

## Acknowledgements

This project builds upon several excellent open-source projects:
- [Point-BERT](https://github.com/lulutang0608/Point-BERT) for the VQVAE tokenization idea
- [Point-M2AE](https://github.com/ZrrSkywalker/Point-M2AE) for the multi-scale architecture
- [timm](https://github.com/huggingface/pytorch-image-models) for Vision Transformer blocks
- [PyTorch](https://pytorch.org/) as the deep learning framework

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
