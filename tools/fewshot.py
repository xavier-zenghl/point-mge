# tools/fewshot.py
"""Stage 4b: Few-shot learning evaluation.
Usage: python tools/fewshot.py --config configs/fewshot.yaml --ckpt experiments/pretrain/best.pth
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config, merge_config
from utils.logger import get_logger
from utils.checkpoint import load_checkpoint
from datasets import build_dataset
from models.point_patch_embed import PointPatchEmbed
from models.extractor import Extractor


def run_fewshot_episode(support_features, support_labels, query_features, query_labels):
    classes = support_labels.unique()
    centroids = []
    for c in classes:
        mask = support_labels == c
        centroids.append(support_features[mask].mean(dim=0))
    centroids = torch.stack(centroids)
    dists = torch.cdist(query_features, centroids)
    preds = classes[dists.argmin(dim=-1)]
    correct = (preds == query_labels).sum().item()
    return 100.0 * correct / query_labels.shape[0]


class FeatureExtractor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D = cfg.model.embed_dim
        G = cfg.model.num_groups
        self.patch_embed = PointPatchEmbed(in_channels=3, embed_dim=D, num_groups=G, group_size=cfg.model.group_size)
        self.extractor = Extractor(embed_dim=D, depth=cfg.model.extractor_depth, num_heads=cfg.model.num_heads, num_groups=G)

    def forward(self, pc):
        tokens, centers = self.patch_embed(pc)
        features = self.extractor(tokens, centers)
        return features.max(dim=1).values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.opts:
        cfg = merge_config(cfg, args.opts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger("fewshot")
    model = FeatureExtractor(cfg).to(device)
    load_checkpoint(args.ckpt, model)
    model.eval()
    dataset = build_dataset(cfg.data.dataset, data_root=cfg.data.data_root, split="test", n_points=cfg.data.n_points)
    all_features, all_labels = [], []
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    with torch.no_grad():
        for batch in loader:
            pc = batch["points"].to(device)
            feat = model(pc)
            all_features.append(feat.cpu())
            all_labels.append(torch.tensor(batch["label"]) if not isinstance(batch["label"], torch.Tensor) else batch["label"])
    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)
    k_way = cfg.get("k_way", 5)
    n_shot = cfg.get("n_shot", 10)
    n_query = cfg.get("n_query", 20)
    n_episodes = cfg.get("n_episodes", 10)
    accs = []
    for ep in range(n_episodes):
        classes = all_labels.unique()
        selected = classes[torch.randperm(len(classes))[:k_way]]
        support_feats, support_labs, query_feats, query_labs = [], [], [], []
        for c in selected:
            idx = (all_labels == c).nonzero(as_tuple=True)[0]
            perm = idx[torch.randperm(len(idx))]
            support_feats.append(all_features[perm[:n_shot]])
            support_labs.extend([c] * n_shot)
            query_feats.append(all_features[perm[n_shot:n_shot + n_query]])
            query_labs.extend([c] * min(n_query, len(perm) - n_shot))
        support_feats = torch.cat(support_feats)
        support_labs = torch.tensor(support_labs)
        query_feats = torch.cat(query_feats)
        query_labs = torch.tensor(query_labs)
        acc = run_fewshot_episode(support_feats, support_labs, query_feats, query_labs)
        accs.append(acc)
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    logger.info(f"{k_way}-way {n_shot}-shot: {mean_acc:.2f} +/- {std_acc:.2f}%")

if __name__ == "__main__":
    main()
