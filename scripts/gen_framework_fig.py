"""Generate Point-MGE framework diagram."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(20, 11))
ax.set_xlim(0, 20)
ax.set_ylim(0, 11)
ax.axis("off")
fig.patch.set_facecolor("white")

# Color palette
C_INPUT = "#E8F5E9"
C_EMBED = "#E3F2FD"
C_VQVAE = "#FFF3E0"
C_PRETRAIN = "#F3E5F5"
C_DOWNSTREAM = "#FFEBEE"
C_GEN = "#E0F7FA"
C_NERF = "#FFF9C4"
C_BORDER_INPUT = "#4CAF50"
C_BORDER_EMBED = "#1976D2"
C_BORDER_VQVAE = "#E65100"
C_BORDER_PRETRAIN = "#7B1FA2"
C_BORDER_DOWN = "#C62828"
C_BORDER_GEN = "#00838F"
C_BORDER_NERF = "#F9A825"
C_ARROW = "#455A64"
C_TITLE = "#263238"
C_PHASE = "#78909C"


def draw_box(x, y, w, h, label, fc, ec, fontsize=9, bold=False, sublabel=None, radius=0.15):
    box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.08,rounding_size={radius}",
                         facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=3)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    if sublabel:
        ax.text(x + w / 2, y + h / 2 + 0.15, label, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, zorder=4, color="#212121")
        ax.text(x + w / 2, y + h / 2 - 0.2, sublabel, ha="center", va="center",
                fontsize=7, zorder=4, color="#616161", style="italic")
    else:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, zorder=4, color="#212121")


def draw_arrow(x1, y1, x2, y2, color=C_ARROW, style="-|>", lw=1.5, ls="-"):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=style, color=color,
                            linewidth=lw, linestyle=ls,
                            mutation_scale=14, zorder=2,
                            connectionstyle="arc3,rad=0")
    ax.add_patch(arrow)


def draw_curved_arrow(x1, y1, x2, y2, color=C_ARROW, rad=0.3, lw=1.5):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle="-|>", color=color,
                            linewidth=lw, mutation_scale=14, zorder=2,
                            connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(arrow)


# === Title ===
ax.text(10, 10.5, "Point-MGE Framework", ha="center", va="center",
        fontsize=18, fontweight="bold", color=C_TITLE, zorder=5)

# === Phase labels ===
ax.text(1.0, 9.6, "Stage 1-2", fontsize=8, color=C_PHASE, fontstyle="italic")
ax.text(7.0, 9.6, "Stage 3: Pretraining", fontsize=8, color=C_PHASE, fontstyle="italic")
ax.text(14.5, 9.6, "Stage 4: Downstream", fontsize=8, color=C_PHASE, fontstyle="italic")

# ===================== LEFT: Input & Tokenization =====================

# Input point cloud
draw_box(0.3, 7.5, 2.0, 0.9, "Point Cloud", C_INPUT, C_BORDER_INPUT, fontsize=10, bold=True,
         sublabel="(B, 2048, 3)")

# Point Patch Embedding
draw_box(0.3, 5.5, 2.0, 1.3, "Patch Embed", C_EMBED, C_BORDER_EMBED, fontsize=9, bold=True,
         sublabel="FPS + KNN + PointNet")
draw_arrow(1.3, 7.5, 1.3, 6.85)

# Patch tokens output
draw_box(0.3, 3.8, 2.0, 0.9, "Patch Tokens", C_EMBED, C_BORDER_EMBED, fontsize=9,
         sublabel="(B, 64, 384)")
draw_arrow(1.3, 5.5, 1.3, 4.75)

# ===================== VQVAE Branch =====================

# VQVAE Encoder
draw_box(3.2, 5.5, 1.8, 1.3, "VQ Encoder", C_VQVAE, C_BORDER_VQVAE, fontsize=9, bold=True,
         sublabel="ViT Blocks")
draw_arrow(2.3, 4.25, 3.2, 5.9)  # from patch tokens to VQ encoder

# VQ Codebook
draw_box(3.2, 3.8, 1.8, 0.9, "Codebook", C_VQVAE, C_BORDER_VQVAE, fontsize=9, bold=True,
         sublabel="K = 8192")
draw_arrow(4.1, 5.5, 4.1, 4.75)

# VQ Decoder
draw_box(3.2, 2.1, 1.8, 0.9, "VQ Decoder", C_VQVAE, C_BORDER_VQVAE, fontsize=9,
         sublabel="ViT Blocks")
draw_arrow(4.1, 3.8, 4.1, 3.05)

# NeRF Triplane target
draw_box(0.3, 2.1, 2.0, 0.9, "NeRF Triplane", C_NERF, C_BORDER_NERF, fontsize=9, bold=True,
         sublabel="3 x (C, H, W)")
draw_arrow(3.2, 2.55, 2.3, 2.55)  # decoder -> nerf (reconstruction target)
ax.text(2.75, 2.8, "target", fontsize=7, color="#E65100", ha="center", fontstyle="italic")

# Discrete tokens output (going right to pretraining)
draw_box(3.2, 0.8, 1.8, 0.7, "Discrete Tokens", C_VQVAE, C_BORDER_VQVAE, fontsize=8,
         sublabel="(B, 64)")
draw_arrow(4.1, 2.1, 4.1, 1.55)

# ===================== CENTER: Pretraining =====================

# Sliding masking
draw_box(6.5, 7.5, 2.2, 0.9, "Sliding Mask", C_PRETRAIN, C_BORDER_PRETRAIN, fontsize=9, bold=True,
         sublabel="m_r = 1 - β^(γ/Γ·u)")
draw_arrow(2.3, 7.95, 6.5, 7.95)  # from point cloud to masking

# Visible tokens
draw_box(6.0, 5.5, 1.5, 0.9, "Visible\nTokens", C_PRETRAIN, C_BORDER_PRETRAIN, fontsize=8)
draw_arrow(7.1, 7.5, 6.75, 6.45)

# Mask tokens
draw_box(8.2, 5.5, 1.5, 0.9, "Mask\nTokens", C_PRETRAIN, C_BORDER_PRETRAIN, fontsize=8)
draw_arrow(8.1, 7.5, 8.95, 6.45)

# Extractor
draw_box(6.0, 3.8, 1.5, 0.9, "Extractor", C_PRETRAIN, C_BORDER_PRETRAIN, fontsize=10, bold=True,
         sublabel="ViT-12L")
draw_arrow(6.75, 5.5, 6.75, 4.75)

# Generator
draw_box(7.0, 2.1, 2.7, 0.9, "Generator", C_PRETRAIN, C_BORDER_PRETRAIN, fontsize=10, bold=True,
         sublabel="ViT-4L → predict token indices")
draw_arrow(6.75, 3.8, 7.5, 3.05)  # extractor -> generator
draw_arrow(8.95, 5.5, 8.7, 3.05)  # mask tokens -> generator

# CE Loss
draw_box(7.4, 0.8, 1.8, 0.7, "CE Loss", C_PRETRAIN, C_BORDER_PRETRAIN, fontsize=9, bold=True)
draw_arrow(8.35, 2.1, 8.35, 1.55)
draw_arrow(5.0, 1.15, 7.4, 1.15)  # discrete tokens -> CE loss
ax.text(6.2, 1.35, "supervision", fontsize=7, color="#7B1FA2", ha="center", fontstyle="italic")

# ===================== RIGHT: Downstream =====================

# Arrow from extractor to downstream
draw_arrow(7.5, 4.25, 11.5, 4.25, color="#C62828", lw=2.0)
ax.text(9.5, 4.5, "pretrained weights", fontsize=8, color="#C62828", ha="center",
        fontstyle="italic", fontweight="bold")

# Classification
draw_box(11.5, 7.5, 2.5, 0.9, "Classification", C_DOWNSTREAM, C_BORDER_DOWN, fontsize=9, bold=True,
         sublabel="ModelNet40 / ScanObjectNN")

# Few-shot
draw_box(11.5, 5.9, 2.5, 0.9, "Few-Shot Learning", C_DOWNSTREAM, C_BORDER_DOWN, fontsize=9, bold=True,
         sublabel="5-way 10-shot / 20-shot")

# Part Segmentation
draw_box(11.5, 4.3, 2.5, 0.9, "Part Segmentation", C_DOWNSTREAM, C_BORDER_DOWN, fontsize=9, bold=True,
         sublabel="ShapeNetPart mIoU")

# Extractor block (downstream)
draw_box(11.5, 2.7, 2.5, 0.9, "Extractor", C_PRETRAIN, C_BORDER_PRETRAIN, fontsize=9, bold=True,
         sublabel="+ Task Head")

draw_arrow(12.75, 3.6, 12.75, 4.3)
draw_arrow(12.75, 5.25, 12.75, 5.9)
draw_arrow(12.75, 6.85, 12.75, 7.5)

# ===================== FAR RIGHT: Generation =====================

# GPT Generator
draw_box(15.5, 5.9, 2.5, 0.9, "GPT Generator", C_GEN, C_BORDER_GEN, fontsize=9, bold=True,
         sublabel="Autoregressive Transformer")

# VQVAE Decode
draw_box(15.5, 4.3, 2.5, 0.9, "VQVAE Decode", C_GEN, C_BORDER_GEN, fontsize=9,
         sublabel="Tokens → Point Cloud")
draw_arrow(16.75, 5.9, 16.75, 5.25)

# Generated shapes
draw_box(15.5, 2.7, 2.5, 0.9, "3D Shapes", C_GEN, C_BORDER_GEN, fontsize=9, bold=True,
         sublabel="COV / MMD / 1-NNA")
draw_arrow(16.75, 4.3, 16.75, 3.65)

# Token supervision from VQVAE to GPT
draw_curved_arrow(5.0, 0.95, 15.5, 6.1, color=C_BORDER_GEN, rad=-0.25, lw=1.5)
ax.text(10.5, 0.5, "token sequence supervision", fontsize=7, color="#00838F",
        ha="center", fontstyle="italic")

# ===================== Dashed region boxes =====================
# Tokenization region
rect1 = mpatches.FancyBboxPatch((0.0, 0.4), 5.3, 8.8,
                                 boxstyle="round,pad=0.1,rounding_size=0.3",
                                 facecolor="none", edgecolor="#90A4AE",
                                 linewidth=1.2, linestyle="--", zorder=1)
ax.add_patch(rect1)
ax.text(2.65, 9.35, "Tokenization (Stage 1-2)", fontsize=9, ha="center",
        color="#546E7A", fontweight="bold")

# Pretraining region
rect2 = mpatches.FancyBboxPatch((5.6, 0.4), 4.8, 8.8,
                                 boxstyle="round,pad=0.1,rounding_size=0.3",
                                 facecolor="none", edgecolor="#90A4AE",
                                 linewidth=1.2, linestyle="--", zorder=1)
ax.add_patch(rect2)
ax.text(8.0, 9.35, "Pretraining (Stage 3)", fontsize=9, ha="center",
        color="#546E7A", fontweight="bold")

# Downstream region
rect3 = mpatches.FancyBboxPatch((11.1, 2.3), 3.3, 6.9,
                                 boxstyle="round,pad=0.1,rounding_size=0.3",
                                 facecolor="none", edgecolor="#90A4AE",
                                 linewidth=1.2, linestyle="--", zorder=1)
ax.add_patch(rect3)
ax.text(12.75, 9.35, "Downstream (Stage 4)", fontsize=9, ha="center",
        color="#546E7A", fontweight="bold")

# Generation region
rect4 = mpatches.FancyBboxPatch((15.1, 2.3), 3.3, 5.3,
                                 boxstyle="round,pad=0.1,rounding_size=0.3",
                                 facecolor="none", edgecolor="#90A4AE",
                                 linewidth=1.2, linestyle="--", zorder=1)
ax.add_patch(rect4)
ax.text(16.75, 7.75, "Generation (Stage 4)", fontsize=9, ha="center",
        color="#546E7A", fontweight="bold")

plt.tight_layout()
plt.savefig("/home/xavierzeng/workspace/code/point-mge/docs/images/framework.png",
            dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
plt.close()
print("Done")
