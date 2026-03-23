import math
import torch


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, min_lr: float = 1e-5):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            alpha = epoch / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine for base_lr in self.base_lrs]


def build_scheduler(optimizer, name="cosine", epochs=300, warmup_epochs=10, min_lr=1e-5):
    if name == "cosine":
        return CosineWarmupScheduler(optimizer, warmup_epochs, epochs, min_lr)
    else:
        raise ValueError(f"Unknown scheduler: {name}")
