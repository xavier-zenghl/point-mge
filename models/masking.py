import torch


def compute_mask_ratio(epoch: int, total_epochs: int, beta: float = 0.5, u: float = 2.0) -> float:
    progress = epoch / max(total_epochs - 1, 1)
    mask_ratio = 1.0 - beta ** (progress * u)
    return max(mask_ratio, beta)


def sliding_mask(batch_size: int, num_groups: int, mask_ratio: float, device: torch.device = None) -> torch.Tensor:
    num_visible = max(1, int(num_groups * (1 - mask_ratio)))
    noise = torch.rand(batch_size, num_groups, device=device)
    ids_sorted = noise.argsort(dim=-1)
    visible_mask = torch.zeros(batch_size, num_groups, dtype=torch.bool, device=device)
    visible_ids = ids_sorted[:, :num_visible]
    visible_mask.scatter_(1, visible_ids, True)
    return visible_mask
