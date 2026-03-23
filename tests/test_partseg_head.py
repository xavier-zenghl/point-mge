import torch
from models.heads.partseg_head import PartSegHead

def test_partseg_head_output_shape():
    head = PartSegHead(embed_dim=384, num_groups=64, num_parts=50, num_categories=16)
    features = torch.randn(2, 64, 384)
    centers = torch.randn(2, 64, 3)
    pc = torch.randn(2, 2048, 3)
    category = torch.tensor([0, 3])
    seg_logits = head(features, centers, pc, category)
    assert seg_logits.shape == (2, 2048, 50)

def test_partseg_head_gradient():
    head = PartSegHead(embed_dim=128, num_groups=16, num_parts=10, num_categories=4)
    features = torch.randn(1, 16, 128, requires_grad=True)
    centers = torch.randn(1, 16, 3)
    pc = torch.randn(1, 256, 3)
    category = torch.tensor([0])
    seg_logits = head(features, centers, pc, category)
    seg_logits.sum().backward()
    assert features.grad is not None
