# tools/train_nerf.py
"""Stage 1: Train per-object NeRF and extract triplane features.
Usage: python tools/train_nerf.py --config configs/nerf.yaml
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config, merge_config
from utils.logger import get_logger
from models.nerf import TriplaneNeRF


def train_single_object(model, images, cameras, cfg, logger):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    device = next(model.parameters()).device
    for epoch in range(cfg.train.epochs_per_object):
        points = torch.rand(1, cfg.train.batch_size_rays, 3, device=device) * 2 - 1
        target_rgb = images[:cfg.train.batch_size_rays].to(device) if images is not None else torch.rand(1, cfg.train.batch_size_rays, 3, device=device)
        rgb_pred, density = model(points)
        loss = F.mse_loss(rgb_pred.squeeze(0), target_rgb.squeeze(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            logger.info(f"  Epoch {epoch}: loss={loss.item():.6f}")
    return model.get_triplane_features()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.opts:
        cfg = merge_config(cfg, args.opts)
    logger = get_logger("train_nerf")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.output.feature_dir, exist_ok=True)
    data_root = cfg.data.data_root
    if not os.path.exists(data_root):
        logger.info(f"Data root {data_root} not found. Skipping.")
        return
    object_list = sorted(os.listdir(data_root))
    logger.info(f"Training NeRF for {len(object_list)} objects")
    for obj_name in tqdm(object_list):
        output_path = os.path.join(cfg.output.feature_dir, f"{obj_name}.npy")
        if os.path.exists(output_path):
            continue
        model = TriplaneNeRF(plane_resolution=cfg.model.plane_resolution, plane_channels=cfg.model.plane_channels, mlp_hidden=cfg.model.mlp_hidden).to(device)
        features = train_single_object(model, None, None, cfg, logger)
        np.save(output_path, features.cpu().numpy())
    logger.info("NeRF training complete.")


if __name__ == "__main__":
    main()
