import torch
import torch.nn.functional as F


def farthest_point_sample(xyz: torch.Tensor, n_points: int) -> torch.Tensor:
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, n_points, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), device=device)
    batch_indices = torch.arange(B, device=device)
    for i in range(n_points):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.min(distance, dist)
        farthest = distance.argmax(dim=-1)
    return centroids


def knn_query(xyz: torch.Tensor, centers: torch.Tensor, k: int) -> torch.Tensor:
    dist = torch.cdist(centers, xyz, p=2.0)
    _, idx = dist.topk(k, dim=-1, largest=False)
    return idx


def morton_sort(centers: torch.Tensor) -> torch.Tensor:
    B, G, _ = centers.shape
    mins = centers.min(dim=1, keepdim=True).values
    maxs = centers.max(dim=1, keepdim=True).values
    span = (maxs - mins).clamp(min=1e-6)
    normalized = ((centers - mins) / span * 1023).long().clamp(0, 1023)
    codes = torch.zeros(B, G, dtype=torch.long, device=centers.device)
    for bit in range(10):
        codes |= ((normalized[..., 0] >> bit) & 1) << (3 * bit)
        codes |= ((normalized[..., 1] >> bit) & 1) << (3 * bit + 1)
        codes |= ((normalized[..., 2] >> bit) & 1) << (3 * bit + 2)
    sorted_indices = codes.argsort(dim=-1)
    return sorted_indices


def random_point_dropout(pc: torch.Tensor, max_dropout_ratio: float = 0.875) -> torch.Tensor:
    B, N, C = pc.shape
    result = pc.clone()
    for b in range(B):
        dropout_ratio = torch.rand(1).item() * max_dropout_ratio
        drop_idx = torch.where(torch.rand(N) < dropout_ratio)[0]
        if len(drop_idx) > 0:
            result[b, drop_idx] = result[b, 0].clone()
    return result


def random_scale_shift(pc: torch.Tensor, scale_low: float = 0.8, scale_high: float = 1.25, shift_range: float = 0.1) -> torch.Tensor:
    B, N, C = pc.shape
    scales = torch.empty(B, 1, 1).uniform_(scale_low, scale_high).to(pc.device)
    shifts = torch.empty(B, 1, C).uniform_(-shift_range, shift_range).to(pc.device)
    return pc * scales + shifts
