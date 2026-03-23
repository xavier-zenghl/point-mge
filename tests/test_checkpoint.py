import os
import tempfile
import torch
import torch.nn as nn
from utils.checkpoint import save_checkpoint, load_checkpoint


def test_save_and_load_checkpoint():
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.pth")
        save_checkpoint(path, model, optimizer, epoch=10, best_metric=0.95)
        model2 = nn.Linear(10, 5)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        ckpt = load_checkpoint(path, model2, optimizer2)
        assert ckpt["epoch"] == 10
        assert ckpt["best_metric"] == 0.95
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)


def test_load_checkpoint_model_only():
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.pth")
        save_checkpoint(path, model, optimizer, epoch=5)
        model2 = nn.Linear(10, 5)
        ckpt = load_checkpoint(path, model2)
        assert ckpt["epoch"] == 5
