# tools/train_generation.py
"""Stage 4d: Train GPT generator for 3D shape generation."""
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
from utils.distributed import setup_distributed, cleanup_distributed, is_main_process, get_rank
from utils.metrics import AverageMeter
from datasets import build_dataset
from models.vqvae import VQVAE
from models.gpt_generator import GPTGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--vqvae_ckpt", type=str, required=True)
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
    logger = get_logger("train_gen", log_file=os.path.join(cfg.output.exp_dir, "train.log") if is_main_process() else None)
    os.makedirs(cfg.output.exp_dir, exist_ok=True)
    vqvae = VQVAE(embed_dim=cfg.model.vqvae_embed_dim, num_groups=cfg.model.num_groups, group_size=cfg.model.group_size, encoder_depth=6, decoder_depth=6, num_heads=cfg.model.num_heads, codebook_size=cfg.model.codebook_size).to(device)
    load_checkpoint(args.vqvae_ckpt, vqvae)
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False
    model = GPTGenerator(codebook_size=cfg.model.codebook_size, embed_dim=cfg.model.gpt_embed_dim, depth=cfg.model.gpt_depth, num_heads=cfg.model.gpt_heads, seq_len=cfg.model.num_groups, num_classes=cfg.model.get("num_classes", 55)).to(device)
    if distributed:
        model = DDP(model, device_ids=[get_rank()])
    dataset = build_dataset(cfg.data.dataset, data_root=cfg.data.data_root, split=cfg.data.train_split, n_points=cfg.data.n_points)
    sampler = DistributedSampler(dataset) if distributed else None
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, sampler=sampler, shuffle=(sampler is None), num_workers=4, pin_memory=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = build_scheduler(optimizer, epochs=cfg.train.epochs, warmup_epochs=cfg.train.warmup_epochs, min_lr=cfg.train.min_lr)
    for epoch in range(cfg.train.epochs):
        if distributed:
            sampler.set_epoch(epoch)
        model.train()
        loss_meter = AverageMeter()
        for batch in loader:
            pc = batch["points"].to(device)
            with torch.no_grad():
                indices, _ = vqvae.encode(pc)
            input_ids = indices[:, :-1]
            target_ids = indices[:, 1:]
            logits = model(input_ids)[:, :target_ids.shape[1]]
            loss = F.cross_entropy(logits.reshape(-1, cfg.model.codebook_size), target_ids.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), pc.shape[0])
        scheduler.step()
        if is_main_process():
            logger.info(f"Epoch [{epoch}] Loss: {loss_meter.avg:.4f}")
            if (epoch + 1) % cfg.output.save_freq == 0:
                save_checkpoint(os.path.join(cfg.output.exp_dir, f"epoch_{epoch}.pth"), model.module if distributed else model, optimizer, epoch=epoch)
    if is_main_process():
        save_checkpoint(os.path.join(cfg.output.exp_dir, "final.pth"), model.module if distributed else model, optimizer, epoch=cfg.train.epochs - 1)
    if distributed:
        cleanup_distributed()

if __name__ == "__main__":
    main()
