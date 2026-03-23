import torch
from models.heads.cls_head import ClassificationHead

def test_cls_head_output_shape():
    head = ClassificationHead(embed_dim=384, num_classes=40)
    features = torch.randn(2, 64, 384)
    logits = head(features)
    assert logits.shape == (2, 40)

def test_cls_head_gradient():
    head = ClassificationHead(embed_dim=128, num_classes=10)
    features = torch.randn(2, 16, 128, requires_grad=True)
    logits = head(features)
    logits.sum().backward()
    assert features.grad is not None
