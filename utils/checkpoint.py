import torch
import torch.nn as nn


def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer | None = None, epoch: int = 0, best_metric: float = 0.0, **kwargs):
    state = {"epoch": epoch, "best_metric": best_metric, "model_state_dict": model.state_dict()}
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    state.update(kwargs)
    torch.save(state, path)


def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer | None = None) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt
