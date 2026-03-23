import torch
from models.point_patch_embed import PointPatchEmbed

def test_point_patch_embed_output_shape():
    model = PointPatchEmbed(in_channels=3, embed_dim=384, num_groups=64, group_size=32)
    pc = torch.randn(2, 2048, 3)
    tokens, centers = model(pc)
    assert tokens.shape == (2, 64, 384)
    assert centers.shape == (2, 64, 3)

def test_point_patch_embed_gradient_flow():
    model = PointPatchEmbed(in_channels=3, embed_dim=384, num_groups=64, group_size=32)
    pc = torch.randn(2, 2048, 3, requires_grad=True)
    tokens, centers = model(pc)
    loss = tokens.sum()
    loss.backward()
    assert pc.grad is not None
    assert pc.grad.shape == pc.shape

def test_point_patch_embed_with_morton_sort():
    model = PointPatchEmbed(in_channels=3, embed_dim=384, num_groups=64, group_size=32, use_morton_sort=True)
    pc = torch.randn(2, 2048, 3)
    tokens, centers = model(pc)
    assert tokens.shape == (2, 64, 384)

def test_point_patch_embed_different_sizes():
    model = PointPatchEmbed(in_channels=3, embed_dim=256, num_groups=32, group_size=16)
    pc = torch.randn(4, 1024, 3)
    tokens, centers = model(pc)
    assert tokens.shape == (4, 32, 256)
    assert centers.shape == (4, 32, 3)
