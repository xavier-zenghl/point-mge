# tools/train_vqvae.py
"""Stage 2: Train VQVAE tokenizer.
Usage: python tools/train_vqvae.py --config configs/vqvae.yaml
       torchrun --nproc_per_node=4 tools/train_vqvae.py --config configs/vqvae.yaml
"""
import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config, merge_config
from utils.logger import get_logger
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.scheduler import build_scheduler
from utils.distributed import setup_distributed, cleanup_distributed, is_main_process, get_rank, get_world_size
from utils.metrics import AverageMeter
from datasets import build_dataset
from models.vqvae import VQVAE


def train_one_epoch(model, loader, optimizer, device, epoch, logger):
    model.train()
    loss_meter = AverageMeter()
    for batch_idx, batch in enumerate(loader):
        pc = batch["points"].to(device)
        result = model(pc)
        recon_loss = result["quantized"].pow(2).mean()
        vq_loss = result["vq_loss"]
        loss = recon_loss + vq_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), pc.size(0))
        if batch_idx % 50 == 0 and is_main_process():
            logger.info(f"Epoch [{epoch}] Step [{batch_idx}/{len(loader)}] Loss: {loss_meter.avg:.4f} VQ: {vq_loss.item():.4f}")
    return loss_meter.avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.opts:
        cfg = merge_config(cfg, args.opts)
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        setup_distributed()
    device = torch.device(f"cuda:{get_rank()}" if torch.cuda.is_available() else "cpu")
    logger = get_logger("train_vqvae", log_file=os.path.join(cfg.output.exp_dir, "train.log") if is_main_process() else None)
    os.makedirs(cfg.output.exp_dir, exist_ok=True)
    dataset = build_dataset(cfg.data.dataset, data_root=cfg.data.data_root, split=cfg.data.train_split, n_points=cfg.data.n_points)
    sampler = DistributedSampler(dataset) if distributed else None
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, sampler=sampler, shuffle=(sampler is None), num_workers=4, pin_memory=True, drop_last=True)
    model = VQVAE(embed_dim=cfg.model.embed_dim, num_groups=cfg.model.num_groups, group_size=cfg.model.group_size, encoder_depth=cfg.model.get("encoder_depth", 6), decoder_depth=cfg.model.get("decoder_depth", 6), num_heads=cfg.model.num_heads, codebook_size=cfg.model.codebook_size).to(device)
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
        loss = train_one_epoch(model, loader, optimizer, device, epoch, logger)
        scheduler.step()
        if is_main_process():
            logger.info(f"Epoch [{epoch}] Loss: {loss:.4f}")
            if (epoch + 1) % cfg.output.save_freq == 0:
                save_checkpoint(os.path.join(cfg.output.exp_dir, f"epoch_{epoch}.pth"), model.module if distributed else model, optimizer, epoch=epoch)
    if is_main_process():
        save_checkpoint(os.path.join(cfg.output.exp_dir, "final.pth"), model.module if distributed else model, optimizer, epoch=cfg.train.epochs - 1)
    if distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
