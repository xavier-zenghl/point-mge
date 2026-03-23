import torch
from models.vqvae import VectorQuantize

def test_vq_output_shape():
    vq = VectorQuantize(dim=384, codebook_size=8192, commitment_weight=0.25)
    z = torch.randn(2, 64, 384)
    quantized, indices, loss = vq(z)
    assert quantized.shape == (2, 64, 384)
    assert indices.shape == (2, 64)
    assert loss.shape == ()

def test_vq_indices_valid_range():
    vq = VectorQuantize(dim=384, codebook_size=512)
    z = torch.randn(2, 64, 384)
    _, indices, _ = vq(z)
    assert indices.min() >= 0
    assert indices.max() < 512

def test_vq_gradient_through_straight_through():
    vq = VectorQuantize(dim=384, codebook_size=512)
    z = torch.randn(2, 64, 384, requires_grad=True)
    quantized, _, loss = vq(z)
    total_loss = quantized.sum() + loss
    total_loss.backward()
    assert z.grad is not None

def test_vq_codebook_lookup():
    vq = VectorQuantize(dim=64, codebook_size=128)
    z = torch.randn(1, 10, 64)
    _, indices, _ = vq(z)
    looked_up = vq.codebook(indices)
    assert looked_up.shape == (1, 10, 64)
