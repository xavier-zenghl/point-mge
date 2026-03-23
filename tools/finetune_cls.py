# tools/finetune_cls.py
"""Stage 4a: Finetune for classification.
Usage: python tools/finetune_cls.py --config configs/cls_modelnet40.yaml
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
from utils.metrics import accuracy, AverageMeter
from datasets import build_dataset
from models.point_patch_embed import PointPatchEmbed
from models.extractor import Extractor
from models.heads.cls_head import ClassificationHead


class ClassificationModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D = cfg.model.embed_dim
        G = cfg.model.num_groups
        self.patch_embed = PointPatchEmbed(in_channels=3, embed_dim=D, num_groups=G, group_size=cfg.model.group_size)
        self.extractor = Extractor(embed_dim=D, depth=cfg.model.extractor_depth, num_heads=cfg.model.num_heads, num_groups=G)
        self.cls_head = ClassificationHead(embed_dim=D, num_classes=cfg.data.num_classes)

    def forward(self, pc):
        tokens, centers = self.patch_embed(pc)
        features = self.extractor(tokens, centers)
        return self.cls_head(features)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        pc = batch["points"].to(device)
        labels = batch["label"]
        labels = labels.to(device) if isinstance(labels, torch.Tensor) else torch.tensor(labels, device=device)
        logits = model(pc)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.shape[0]
    return 100.0 * correct / total if total > 0 else 0.0


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
    logger = get_logger("finetune_cls", log_file=os.path.join(cfg.output.exp_dir, "train.log") if is_main_process() else None)
    os.makedirs(cfg.output.exp_dir, exist_ok=True)
    train_ds = build_dataset(cfg.data.dataset, data_root=cfg.data.data_root, split=cfg.data.train_split, n_points=cfg.data.n_points)
    val_ds = build_dataset(cfg.data.dataset, data_root=cfg.data.data_root, split=cfg.data.val_split, n_points=cfg.data.n_points)
    train_sampler = DistributedSampler(train_ds) if distributed else None
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    model = ClassificationModel(cfg).to(device)
    if cfg.model.get("pretrained"):
        ckpt = torch.load(cfg.model.pretrained, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
    if distributed:
        model = DDP(model, device_ids=[get_rank()])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = build_scheduler(optimizer, epochs=cfg.train.epochs, warmup_epochs=cfg.train.warmup_epochs, min_lr=cfg.train.min_lr)
    best_acc = 0.0
    for epoch in range(cfg.train.epochs):
        if distributed:
            train_sampler.set_epoch(epoch)
        model.train()
        loss_meter = AverageMeter()
        for batch in train_loader:
            pc = batch["points"].to(device)
            labels = batch["label"]
            labels = labels.to(device) if isinstance(labels, torch.Tensor) else torch.tensor(labels, device=device)
            logits = model(pc)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), pc.shape[0])
        scheduler.step()
        if is_main_process():
            acc = evaluate(model.module if distributed else model, val_loader, device)
            logger.info(f"Epoch [{epoch}] Loss: {loss_meter.avg:.4f} Acc: {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                save_checkpoint(os.path.join(cfg.output.exp_dir, "best.pth"), model.module if distributed else model, optimizer, epoch=epoch, best_metric=best_acc)
    if is_main_process():
        logger.info(f"Best accuracy: {best_acc:.2f}%")
    if distributed:
        cleanup_distributed()

if __name__ == "__main__":
    main()
