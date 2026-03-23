# models/heads/cls_head.py
import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, embed_dim: int = 384, num_classes: int = 40, dropout: float = 0.5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        max_pool = features.max(dim=1).values
        avg_pool = features.mean(dim=1)
        pooled = torch.cat([max_pool, avg_pool], dim=-1)
        return self.head(pooled)
