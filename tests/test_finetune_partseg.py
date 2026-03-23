import torch
from models.point_patch_embed import PointPatchEmbed
from models.extractor import Extractor
from models.heads.partseg_head import PartSegHead
from utils.metrics import compute_iou

def test_partseg_pipeline_forward():
    D, G = 128, 16
    patch_embed = PointPatchEmbed(in_channels=3, embed_dim=D, num_groups=G, group_size=8)
    extractor = Extractor(embed_dim=D, depth=2, num_heads=4, num_groups=G)
    seg_head = PartSegHead(embed_dim=D, num_groups=G, num_parts=10, num_categories=4)
    pc = torch.randn(2, 256, 3)
    category = torch.tensor([0, 2])
    tokens, centers = patch_embed(pc)
    features = extractor(tokens, centers)
    logits = seg_head(features, centers, pc, category)
    assert logits.shape == (2, 256, 10)
    labels = torch.randint(0, 10, (2, 256))
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 10), labels.reshape(-1))
    loss.backward()
    assert not torch.isnan(loss)

def test_iou_computation():
    pred = torch.tensor([0, 0, 1, 1, 2, 2])
    target = torch.tensor([0, 0, 1, 2, 2, 2])
    iou = compute_iou(pred, target, num_classes=3)
    assert all(0 <= v <= 1 for v in iou if v == v)
