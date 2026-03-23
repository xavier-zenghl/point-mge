import torch
from models.masking import sliding_mask, compute_mask_ratio

def test_mask_ratio_at_start():
    ratio = compute_mask_ratio(epoch=0, total_epochs=300, beta=0.5, u=2.0)
    assert abs(ratio - 0.5) < 0.01

def test_mask_ratio_at_end():
    ratio = compute_mask_ratio(epoch=299, total_epochs=300, beta=0.5, u=2.0)
    assert ratio >= 0.75

def test_mask_ratio_monotonic():
    ratios = [compute_mask_ratio(e, 300, 0.5, 2.0) for e in range(300)]
    for i in range(1, len(ratios)):
        assert ratios[i] >= ratios[i - 1] - 1e-6

def test_sliding_mask_output_shape():
    num_groups = 64
    mask_ratio = 0.75
    mask = sliding_mask(batch_size=4, num_groups=num_groups, mask_ratio=mask_ratio)
    assert mask.shape == (4, num_groups)
    assert mask.dtype == torch.bool
    visible_count = mask.sum(dim=1).float().mean().item()
    assert abs(visible_count - 16) < 2

def test_sliding_mask_different_ratios():
    for ratio in [0.25, 0.5, 0.75, 0.9]:
        mask = sliding_mask(batch_size=8, num_groups=64, mask_ratio=ratio)
        expected_visible = int((1 - ratio) * 64)
        actual_visible = mask.sum(dim=1)[0].item()
        assert abs(actual_visible - expected_visible) <= 1
