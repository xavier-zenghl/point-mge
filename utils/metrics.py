# utils/metrics.py
import torch
import numpy as np


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    correct = (pred == target).sum().item()
    total = target.numel()
    return 100.0 * correct / total


def compute_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> list[float]:
    ious = []
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(intersection / union)
    return ious


def chamfer_distance_batch(pc1: torch.Tensor, pc2: torch.Tensor) -> torch.Tensor:
    dist = torch.cdist(pc1, pc2, p=2.0)
    cd1 = dist.min(dim=2).values.mean(dim=1)
    cd2 = dist.min(dim=1).values.mean(dim=1)
    return cd1 + cd2


def _pairwise_cd(set1: torch.Tensor, set2: torch.Tensor) -> torch.Tensor:
    M, N = set1.shape[0], set2.shape[0]
    dists = torch.zeros(M, N)
    for i in range(M):
        cd = chamfer_distance_batch(set1[i:i+1].expand(N, -1, -1), set2)
        dists[i] = cd
    return dists


def compute_cov(gen: torch.Tensor, ref: torch.Tensor) -> float:
    dists = _pairwise_cd(gen, ref)
    nn_idx = dists.argmin(dim=0)
    coverage = nn_idx.unique().shape[0] / ref.shape[0] * 100
    return coverage


def compute_mmd(gen: torch.Tensor, ref: torch.Tensor) -> float:
    dists = _pairwise_cd(gen, ref)
    min_dists = dists.min(dim=0).values
    return min_dists.mean().item()


def compute_1nna(gen: torch.Tensor, ref: torch.Tensor) -> float:
    all_pc = torch.cat([gen, ref], dim=0)
    labels = torch.cat([torch.zeros(gen.shape[0]), torch.ones(ref.shape[0])])
    N = all_pc.shape[0]
    dists = _pairwise_cd(all_pc, all_pc)
    dists.fill_diagonal_(float("inf"))
    nn_idx = dists.argmin(dim=1)
    nn_labels = labels[nn_idx]
    correct = (nn_labels == labels).float().mean().item()
    return correct * 100
