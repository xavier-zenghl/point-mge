# tools/pretrain.py
"""Stage 3: Pretrain Extractor-Generator with sliding masking.
Usage: python tools/pretrain.py --config configs/pretrain.yaml
       torchrun --nproc_per_node=4 tools/pretrain.py --config configs/pretrain.yaml
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config, merge_config
from utils.logger import get_logger
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.scheduler import build_scheduler
from utils.distributed import setup_distributed, cleanup_distributed, is_main_process, get_rank
from utils.metrics import AverageMeter
from datasets import build_dataset
from models.point_patch_embed import PointPatchEmbed
from models.extractor import Extractor
from models.generator import Generator
from models.vqvae import VQVAE
from models.masking import sliding_mask, compute_mask_ratio


class PretrainModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D = cfg.model.embed_dim
        G = cfg.model.num_groups
        self.patch_embed = PointPatchEmbed(in_channels=3, embed_dim=D, num_groups=G, group_size=cfg.model.group_size)
        self.extractor = Extractor(embed_dim=D, depth=cfg.model.extractor_depth, num_heads=cfg.model.num_heads, num_groups=G)
        self.generator = Generator(embed_dim=D, depth=cfg.model.generator_depth, num_heads=cfg.model.num_heads, num_groups=G, codebook_size=cfg.model.codebook_size)
        self.num_groups = G

    def forward(self, pc, visible_mask, target_indices):
        B, G = visible_mask.shape
        tokens, centers = self.patch_embed(pc)
        vis_tokens_list, vis_centers_list, mask_centers_list, target_list = [], [], [], []
        for b in range(B):
            v_idx = visible_mask[b].nonzero(as_tuple=True)[0]
            m_idx = (~visible_mask[b]).nonzero(as_tuple=True)[0]
            vis_tokens_list.append(tokens[b, v_idx])
            vis_centers_list.append(centers[b, v_idx])
            mask_centers_list.append(centers[b, m_idx])
            target_list.append(target_indices[b, m_idx])
        vis_tokens = torch.stack(vis_tokens_list)
        vis_centers = torch.stack(vis_centers_list)
        mask_centers = torch.stack(mask_centers_list)
        targets = torch.stack(target_list)
        extracted = self.extractor(vis_tokens, vis_centers)
        logits, center_pred = self.generator(extracted, vis_centers, mask_centers, visible_mask)
        token_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        center_loss = F.mse_loss(center_pred, mask_centers)
        return {"token_loss": token_loss, "center_loss": center_loss, "loss": token_loss + 0.1 * center_loss}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--vqvae_ckpt", type=str, default=None)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.opts:
        cfg = merge_config(cfg, args.opts)
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        setup_distributed()
    device = torch.device(f"cuda:{get_rank()}" if torch.cuda.is_available() else "cpu")
    logger = get_logger("pretrain", log_file=os.path.join(cfg.output.exp_dir, "train.log") if is_main_process() else None)
    os.makedirs(cfg.output.exp_dir, exist_ok=True)
    vqvae = VQVAE(embed_dim=cfg.model.embed_dim, num_groups=cfg.model.num_groups, group_size=cfg.model.group_size, encoder_depth=6, decoder_depth=6, num_heads=cfg.model.num_heads, codebook_size=cfg.model.codebook_size).to(device)
    if args.vqvae_ckpt:
        load_checkpoint(args.vqvae_ckpt, vqvae)
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False
    dataset = build_dataset(cfg.data.dataset, data_root=cfg.data.data_root, split=cfg.data.train_split, n_points=cfg.data.n_points)
    sampler = DistributedSampler(dataset) if distributed else None
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, sampler=sampler, shuffle=(sampler is None), num_workers=4, pin_memory=True, drop_last=True)
    model = PretrainModel(cfg).to(device)
    if distributed:
        model = DDP(model, device_ids=[get_rank()])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = build_scheduler(optimizer, epochs=cfg.train.epochs, warmup_epochs=cfg.train.warmup_epochs, min_lr=cfg.train.min_lr)
    start_epoch = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, model.module if distributed else model, optimizer)
        start_epoch = ckpt["epoch"] + 1
    for epoch in range(start_epoch, cfg.train.epochs):
        if distributed:
            sampler.set_epoch(epoch)
        model.train()
        loss_meter = AverageMeter()
        mask_ratio = compute_mask_ratio(epoch, cfg.train.epochs, cfg.model.mask_beta, cfg.model.mask_u)
        for batch_idx, batch in enumerate(loader):
            pc = batch["points"].to(device)
            B = pc.shape[0]
            with torch.no_grad():
                target_indices, _ = vqvae.encode(pc)
            visible_mask = sliding_mask(B, cfg.model.num_groups, mask_ratio, device=device)
            result = model(pc, visible_mask, target_indices)
            loss = result["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), B)
        scheduler.step()
        if is_main_process():
            logger.info(f"Epoch [{epoch}] Loss: {loss_meter.avg:.4f} MaskRatio: {mask_ratio:.3f}")
            if (epoch + 1) % cfg.output.save_freq == 0:
                save_checkpoint(os.path.join(cfg.output.exp_dir, f"epoch_{epoch}.pth"), model.module if distributed else model, optimizer, epoch=epoch)
    if is_main_process():
        save_checkpoint(os.path.join(cfg.output.exp_dir, "final.pth"), model.module if distributed else model, optimizer, epoch=cfg.train.epochs - 1)
    if distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
