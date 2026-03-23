# tests/test_integration.py
import torch

def test_full_pretrain_pipeline():
    from models.point_patch_embed import PointPatchEmbed
    from models.extractor import Extractor
    from models.generator import Generator
    from models.vqvae import VQVAE
    from models.masking import sliding_mask, compute_mask_ratio
    B, N, G, D = 2, 512, 16, 128
    codebook_size = 64
    patch_embed = PointPatchEmbed(in_channels=3, embed_dim=D, num_groups=G, group_size=8)
    vqvae = VQVAE(embed_dim=D, num_groups=G, group_size=8, encoder_depth=2, decoder_depth=2, num_heads=4, codebook_size=codebook_size)
    extractor = Extractor(embed_dim=D, depth=2, num_heads=4, num_groups=G)
    generator = Generator(embed_dim=D, depth=2, num_heads=4, num_groups=G, codebook_size=codebook_size)
    pc = torch.randn(B, N, 3)
    with torch.no_grad():
        target_indices, _ = vqvae.encode(pc)
    tokens, centers = patch_embed(pc)
    mask_ratio = compute_mask_ratio(100, 300)
    visible_mask = sliding_mask(B, G, mask_ratio)
    vis_list, vis_c_list, mask_c_list, tgt_list = [], [], [], []
    for b in range(B):
        v = visible_mask[b].nonzero(as_tuple=True)[0]
        m = (~visible_mask[b]).nonzero(as_tuple=True)[0]
        vis_list.append(tokens[b, v])
        vis_c_list.append(centers[b, v])
        mask_c_list.append(centers[b, m])
        tgt_list.append(target_indices[b, m])
    vis_tokens = torch.stack(vis_list)
    vis_centers = torch.stack(vis_c_list)
    mask_centers = torch.stack(mask_c_list)
    targets = torch.stack(tgt_list)
    features = extractor(vis_tokens, vis_centers)
    logits, center_pred = generator(features, vis_centers, mask_centers, visible_mask)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, codebook_size), targets.reshape(-1))
    loss.backward()
    assert not torch.isnan(loss)
    assert loss.item() > 0

def test_full_classification_pipeline():
    from models.point_patch_embed import PointPatchEmbed
    from models.extractor import Extractor
    from models.heads.cls_head import ClassificationHead
    D, G = 128, 16
    patch_embed = PointPatchEmbed(in_channels=3, embed_dim=D, num_groups=G, group_size=8)
    extractor = Extractor(embed_dim=D, depth=2, num_heads=4, num_groups=G)
    cls_head = ClassificationHead(embed_dim=D, num_classes=40)
    pc = torch.randn(4, 256, 3)
    tokens, centers = patch_embed(pc)
    features = extractor(tokens, centers)
    logits = cls_head(features)
    labels = torch.randint(0, 40, (4,))
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    assert logits.shape == (4, 40)
    assert not torch.isnan(loss)

def test_full_generation_pipeline():
    from models.vqvae import VQVAE
    from models.gpt_generator import GPTGenerator
    codebook_size = 64
    D, G = 128, 16
    vqvae = VQVAE(embed_dim=D, num_groups=G, group_size=8, encoder_depth=2, decoder_depth=2, num_heads=4, codebook_size=codebook_size)
    gpt = GPTGenerator(codebook_size=codebook_size, embed_dim=128, depth=2, num_heads=4, seq_len=G, num_classes=10)
    vqvae.eval()
    gpt.eval()
    with torch.no_grad():
        indices = gpt.generate(torch.tensor([5, 3]), temperature=1.0, top_k=10)
        centers = torch.randn(2, G, 3)
        decoded = vqvae.decode(indices, centers)
    assert indices.shape == (2, G)
    assert decoded.shape == (2, G, D)
