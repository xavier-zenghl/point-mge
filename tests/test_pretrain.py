# tests/test_pretrain.py
import torch
from models.point_patch_embed import PointPatchEmbed
from models.extractor import Extractor
from models.generator import Generator
from models.masking import sliding_mask, compute_mask_ratio
from models.vqvae import VQVAE

def test_pretrain_forward_pass():
    B, N, G, D = 2, 512, 16, 128
    codebook_size = 64
    patch_embed = PointPatchEmbed(in_channels=3, embed_dim=D, num_groups=G, group_size=8)
    extractor = Extractor(embed_dim=D, depth=2, num_heads=4, num_groups=G)
    generator = Generator(embed_dim=D, depth=2, num_heads=4, num_groups=G, codebook_size=codebook_size)
    pc = torch.randn(B, N, 3)
    tokens, centers = patch_embed(pc)
    assert tokens.shape == (B, G, D)
    mask_ratio = compute_mask_ratio(epoch=150, total_epochs=300)
    visible_mask = sliding_mask(B, G, mask_ratio)
    vis_tokens_list, vis_centers_list, mask_centers_list = [], [], []
    for b in range(B):
        v_idx = visible_mask[b].nonzero(as_tuple=True)[0]
        m_idx = (~visible_mask[b]).nonzero(as_tuple=True)[0]
        vis_tokens_list.append(tokens[b, v_idx])
        vis_centers_list.append(centers[b, v_idx])
        mask_centers_list.append(centers[b, m_idx])
    vis_tokens = torch.stack(vis_tokens_list)
    vis_centers = torch.stack(vis_centers_list)
    mask_centers = torch.stack(mask_centers_list)
    extracted = extractor(vis_tokens, vis_centers)
    assert extracted.shape == vis_tokens.shape
    logits, center_pred = generator(extracted, vis_centers, mask_centers, visible_mask)
    assert logits.shape[0] == B
    assert logits.shape[2] == codebook_size
    assert center_pred.shape == mask_centers.shape

def test_pretrain_loss_computation():
    B, M, K = 2, 8, 64
    logits = torch.randn(B, M, K)
    target_indices = torch.randint(0, K, (B, M))
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, K), target_indices.reshape(-1))
    assert loss.shape == ()
    assert not torch.isnan(loss)
