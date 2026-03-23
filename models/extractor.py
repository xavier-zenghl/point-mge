import torch
import torch.nn as nn
from timm.models.vision_transformer import Block as ViTBlock


class Extractor(nn.Module):
    def __init__(self, embed_dim: int = 384, depth: int = 12, num_heads: int = 6, mlp_ratio: float = 4.0, num_groups: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_groups = num_groups
        self.center_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, embed_dim))
        self.blocks = nn.ModuleList([ViTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        pos = self.center_embed(centers)
        x = tokens + pos
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x
