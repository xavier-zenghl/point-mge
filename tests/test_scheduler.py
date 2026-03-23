import torch
import torch.nn as nn
from utils.scheduler import build_scheduler


def test_cosine_scheduler():
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = build_scheduler(optimizer, name="cosine", epochs=100, warmup_epochs=10, min_lr=1e-5)
    lrs = []
    for _ in range(100):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    assert lrs[0] < lrs[9]
    assert lrs[10] > lrs[-1]
    assert abs(lrs[-1] - 1e-5) < 1e-6
