# tests/test_eval_generation.py
import torch
from models.gpt_generator import GPTGenerator
from models.vqvae import VQVAE

def test_generation_pipeline():
    codebook_size = 64
    D, G = 128, 16
    gpt = GPTGenerator(codebook_size=codebook_size, embed_dim=128, depth=2, num_heads=4, seq_len=G, num_classes=10)
    vqvae = VQVAE(embed_dim=D, num_groups=G, group_size=8, encoder_depth=2, decoder_depth=2, num_heads=4, codebook_size=codebook_size)
    gpt.eval()
    vqvae.eval()
    class_label = torch.tensor([3, 7])
    generated_indices = gpt.generate(class_label, temperature=1.0, top_k=10)
    assert generated_indices.shape == (2, G)
    centers = torch.randn(2, G, 3)
    decoded = vqvae.decode(generated_indices, centers)
    assert decoded.shape == (2, G, D)
