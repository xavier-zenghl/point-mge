import torch
from models.point_patch_embed import PointPatchEmbed
from models.extractor import Extractor
from models.heads.cls_head import ClassificationHead

def test_cls_pipeline_forward():
    D, G = 128, 16
    patch_embed = PointPatchEmbed(in_channels=3, embed_dim=D, num_groups=G, group_size=8)
    extractor = Extractor(embed_dim=D, depth=2, num_heads=4, num_groups=G)
    cls_head = ClassificationHead(embed_dim=D, num_classes=40)
    pc = torch.randn(2, 256, 3)
    tokens, centers = patch_embed(pc)
    features = extractor(tokens, centers)
    logits = cls_head(features)
    assert logits.shape == (2, 40)
    labels = torch.randint(0, 40, (2,))
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    assert not torch.isnan(loss)
