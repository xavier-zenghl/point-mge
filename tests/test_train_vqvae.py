# tests/test_train_vqvae.py
import torch
from models.vqvae import VQVAE

def test_vqvae_training_step():
    model = VQVAE(embed_dim=128, num_groups=16, group_size=8, encoder_depth=2, decoder_depth=2, num_heads=4, codebook_size=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    pc = torch.randn(2, 256, 3)
    target = torch.randn(2, 16, 128)
    result = model(pc)
    recon_loss = ((result["quantized"] - target) ** 2).mean()
    loss1 = recon_loss + result["vq_loss"]
    optimizer.zero_grad()
    loss1.backward()
    optimizer.step()
    result2 = model(pc)
    recon_loss2 = ((result2["quantized"] - target) ** 2).mean()
    loss2 = recon_loss2 + result2["vq_loss"]
    assert loss2.item() < loss1.item() * 1.5

def test_vqvae_codebook_utilization():
    model = VQVAE(embed_dim=64, num_groups=16, group_size=8, encoder_depth=2, decoder_depth=2, num_heads=4, codebook_size=32)
    pc = torch.randn(8, 256, 3)
    result = model(pc)
    unique_codes = result["indices"].unique()
    assert len(unique_codes) > 1
