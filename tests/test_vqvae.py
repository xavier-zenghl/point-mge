import torch
from models.vqvae import VQVAE

def test_vqvae_forward_shape():
    model = VQVAE(embed_dim=384, num_groups=64, group_size=32, encoder_depth=4, decoder_depth=4, num_heads=6, codebook_size=512)
    pc = torch.randn(2, 2048, 3)
    result = model(pc)
    assert result["quantized"].shape == (2, 64, 384)
    assert result["indices"].shape == (2, 64)
    assert result["vq_loss"].shape == ()
    assert result["centers"].shape == (2, 64, 3)

def test_vqvae_encode():
    model = VQVAE(embed_dim=384, num_groups=64, group_size=32, encoder_depth=4, decoder_depth=4, num_heads=6, codebook_size=512)
    pc = torch.randn(2, 2048, 3)
    indices, centers = model.encode(pc)
    assert indices.shape == (2, 64)
    assert centers.shape == (2, 64, 3)

def test_vqvae_decode_from_indices():
    model = VQVAE(embed_dim=384, num_groups=64, group_size=32, encoder_depth=4, decoder_depth=4, num_heads=6, codebook_size=512)
    indices = torch.randint(0, 512, (2, 64))
    centers = torch.randn(2, 64, 3)
    decoded = model.decode(indices, centers)
    assert decoded.shape == (2, 64, 384)

def test_vqvae_gradient_flow():
    model = VQVAE(embed_dim=128, num_groups=16, group_size=8, encoder_depth=2, decoder_depth=2, num_heads=4, codebook_size=64)
    pc = torch.randn(1, 256, 3, requires_grad=True)
    result = model(pc)
    loss = result["vq_loss"] + result["quantized"].sum()
    loss.backward()
    assert pc.grad is not None
