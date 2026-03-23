# models/heads/partseg_head.py
import torch
import torch.nn as nn


class PartSegHead(nn.Module):
    def __init__(self, embed_dim: int = 384, num_groups: int = 64, num_parts: int = 50, num_categories: int = 16):
        super().__init__()
        self.num_groups = num_groups
        self.cat_embed = nn.Embedding(num_categories, 64)
        self.head = nn.Sequential(
            nn.Conv1d(embed_dim + 64 + 3, 256, 1), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Conv1d(256, 256, 1), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Conv1d(256, num_parts, 1),
        )

    def forward(self, features, centers, pc, category):
        B, N, _ = pc.shape
        dist = torch.cdist(pc, centers)
        knn_dist, knn_idx = dist.topk(3, dim=-1, largest=False)
        weights = 1.0 / (knn_dist + 1e-8)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        batch_idx = torch.arange(B, device=pc.device).reshape(B, 1, 1).expand(-1, N, 3)
        knn_features = features[batch_idx, knn_idx]
        point_features = (knn_features * weights.unsqueeze(-1)).sum(dim=2)
        cat_feat = self.cat_embed(category).unsqueeze(1).expand(-1, N, -1)
        combined = torch.cat([point_features, cat_feat, pc], dim=-1)
        combined = combined.transpose(1, 2)
        seg_logits = self.head(combined)
        return seg_logits.transpose(1, 2)
