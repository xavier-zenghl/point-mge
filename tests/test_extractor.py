import torch
from models.extractor import Extractor

def test_extractor_forward_shape():
    model = Extractor(embed_dim=384, depth=12, num_heads=6, num_groups=64)
    tokens = torch.randn(2, 32, 384)
    centers = torch.randn(2, 32, 3)
    out = model(tokens, centers)
    assert out.shape == (2, 32, 384)

def test_extractor_with_different_token_counts():
    model = Extractor(embed_dim=384, depth=12, num_heads=6, num_groups=64)
    for n_visible in [16, 32, 48, 64]:
        tokens = torch.randn(2, n_visible, 384)
        centers = torch.randn(2, n_visible, 3)
        out = model(tokens, centers)
        assert out.shape == (2, n_visible, 384)

def test_extractor_gradient_flow():
    model = Extractor(embed_dim=128, depth=2, num_heads=4, num_groups=64)
    tokens = torch.randn(1, 32, 128, requires_grad=True)
    centers = torch.randn(1, 32, 3)
    out = model(tokens, centers)
    out.sum().backward()
    assert tokens.grad is not None
