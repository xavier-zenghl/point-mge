import torch
from models.generator import Generator

def test_generator_forward_shape():
    model = Generator(embed_dim=384, depth=4, num_heads=6, num_groups=64, codebook_size=512)
    visible_features = torch.randn(2, 32, 384)
    visible_centers = torch.randn(2, 32, 3)
    mask_centers = torch.randn(2, 32, 3)
    visible_bool = torch.cat([torch.ones(2, 32), torch.zeros(2, 32)], dim=1).bool()
    logits, center_pred = model(visible_features, visible_centers, mask_centers, visible_bool)
    assert logits.shape == (2, 32, 512)
    assert center_pred.shape == (2, 32, 3)

def test_generator_gradient_flow():
    model = Generator(embed_dim=128, depth=2, num_heads=4, num_groups=16, codebook_size=64)
    vis_feat = torch.randn(1, 8, 128, requires_grad=True)
    vis_centers = torch.randn(1, 8, 3)
    mask_centers = torch.randn(1, 8, 3)
    vis_bool = torch.cat([torch.ones(1, 8), torch.zeros(1, 8)], dim=1).bool()
    logits, center_pred = model(vis_feat, vis_centers, mask_centers, vis_bool)
    loss = logits.sum() + center_pred.sum()
    loss.backward()
    assert vis_feat.grad is not None
