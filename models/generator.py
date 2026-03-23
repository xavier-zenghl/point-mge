import torch
import torch.nn as nn
from timm.models.vision_transformer import Block as ViTBlock


class Generator(nn.Module):
    def __init__(self, embed_dim: int = 384, depth: int = 4, num_heads: int = 6, mlp_ratio: float = 4.0, num_groups: int = 64, codebook_size: int = 8192):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_groups = num_groups
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.center_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, embed_dim))
        self.blocks = nn.ModuleList([ViTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.token_head = nn.Linear(embed_dim, codebook_size)
        self.center_head = nn.Sequential(nn.Linear(embed_dim, 128), nn.GELU(), nn.Linear(128, 3))

    def forward(self, visible_features: torch.Tensor, visible_centers: torch.Tensor, mask_centers: torch.Tensor, visible_bool: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, V, D = visible_features.shape
        M = mask_centers.shape[1]
        mask_tokens = self.mask_token.expand(B, M, -1)
        all_centers = torch.cat([visible_centers, mask_centers], dim=1)
        pos = self.center_embed(all_centers)
        full_tokens = torch.cat([visible_features, mask_tokens], dim=1)
        full_tokens = full_tokens + pos
        for block in self.blocks:
            full_tokens = block(full_tokens)
        full_tokens = self.norm(full_tokens)
        masked_features = full_tokens[:, V:]
        logits = self.token_head(masked_features)
        center_pred = self.center_head(masked_features)
        return logits, center_pred
