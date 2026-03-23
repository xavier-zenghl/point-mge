import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block as ViTBlock
from models.point_patch_embed import PointPatchEmbed


class VectorQuantize(nn.Module):
    def __init__(self, dim: int, codebook_size: int = 8192, commitment_weight: float = 0.25, ema_decay: float = 0.99):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.ema_decay = ema_decay
        self.codebook = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_embed_sum", self.codebook.weight.clone())

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, D = z.shape
        flat_z = z.reshape(-1, D)
        dist = (flat_z.pow(2).sum(dim=-1, keepdim=True) + self.codebook.weight.pow(2).sum(dim=-1, keepdim=True).t() - 2 * flat_z @ self.codebook.weight.t())
        indices = dist.argmin(dim=-1)
        quantized = self.codebook(indices).reshape(B, L, D)
        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(indices, self.codebook_size).float()
                cluster_size = one_hot.sum(dim=0)
                embed_sum = one_hot.t() @ flat_z
                self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)
                self.ema_embed_sum.mul_(self.ema_decay).add_(embed_sum, alpha=1 - self.ema_decay)
                n = self.ema_cluster_size.sum()
                cluster_size_smoothed = (self.ema_cluster_size + 1e-5) / (n + self.codebook_size * 1e-5) * n
                self.codebook.weight.data.copy_(self.ema_embed_sum / cluster_size_smoothed.unsqueeze(1))
        commitment_loss = F.mse_loss(z, quantized.detach())
        codebook_loss = F.mse_loss(quantized, z.detach())
        loss = codebook_loss + self.commitment_weight * commitment_loss
        quantized = z + (quantized - z).detach()
        indices = indices.reshape(B, L)
        return quantized, indices, loss


class VQVAE(nn.Module):
    def __init__(self, embed_dim: int = 384, num_groups: int = 64, group_size: int = 32, encoder_depth: int = 6, decoder_depth: int = 6, num_heads: int = 6, codebook_size: int = 8192, commitment_weight: float = 0.25):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PointPatchEmbed(in_channels=3, embed_dim=embed_dim, num_groups=num_groups, group_size=group_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_groups, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.encoder = nn.Sequential(*[ViTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True) for _ in range(encoder_depth)])
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.vq = VectorQuantize(dim=embed_dim, codebook_size=codebook_size, commitment_weight=commitment_weight)
        self.decoder = nn.Sequential(*[ViTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True) for _ in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(embed_dim)

    def encode(self, pc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, centers = self.patch_embed(pc)
        tokens = tokens + self.pos_embed
        tokens = self.encoder_norm(self.encoder(tokens))
        _, indices, _ = self.vq(tokens)
        return indices, centers

    def decode(self, indices: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        quantized = self.vq.codebook(indices)
        quantized = quantized + self.pos_embed
        decoded = self.decoder_norm(self.decoder(quantized))
        return decoded

    def forward(self, pc: torch.Tensor) -> dict:
        tokens, centers = self.patch_embed(pc)
        tokens = tokens + self.pos_embed
        encoded = self.encoder_norm(self.encoder(tokens))
        quantized, indices, vq_loss = self.vq(encoded)
        quantized_with_pos = quantized + self.pos_embed
        decoded = self.decoder_norm(self.decoder(quantized_with_pos))
        return {"quantized": decoded, "indices": indices, "vq_loss": vq_loss, "centers": centers, "encoded": encoded}
