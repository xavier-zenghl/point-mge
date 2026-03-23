import torch
import torch.nn as nn
from datasets.data_utils import farthest_point_sample, knn_query, morton_sort


class MiniPointNet(nn.Module):
    def __init__(self, in_channels: int = 3, embed_dim: int = 384):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Conv1d(256, embed_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, G, K, C = x.shape
        x = x.reshape(B * G, K, C).transpose(1, 2)
        x = self.mlp(x)
        x = x.max(dim=-1).values
        return x.reshape(B, G, -1)


class PointPatchEmbed(nn.Module):
    def __init__(self, in_channels: int = 3, embed_dim: int = 384, num_groups: int = 64, group_size: int = 32, use_morton_sort: bool = False):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.use_morton_sort = use_morton_sort
        self.mini_pointnet = MiniPointNet(in_channels, embed_dim)

    def forward(self, pc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, C = pc.shape
        center_idx = farthest_point_sample(pc, self.num_groups)
        batch_idx = torch.arange(B, device=pc.device).unsqueeze(1).expand(-1, self.num_groups)
        centers = pc[batch_idx, center_idx]
        if self.use_morton_sort:
            sort_idx = morton_sort(centers)
            batch_idx2 = torch.arange(B, device=pc.device).unsqueeze(1).expand(-1, self.num_groups)
            centers = centers[batch_idx2, sort_idx]
            center_idx = center_idx[batch_idx2, sort_idx]
        knn_idx = knn_query(pc, centers, k=self.group_size)
        batch_idx3 = torch.arange(B, device=pc.device).reshape(B, 1, 1).expand(-1, self.num_groups, self.group_size)
        grouped = pc[batch_idx3, knn_idx]
        grouped = grouped - centers.unsqueeze(2)
        tokens = self.mini_pointnet(grouped)
        return tokens, centers
