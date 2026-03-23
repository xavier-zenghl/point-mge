# tests/test_train_nerf.py
import torch
from models.nerf import TriplaneNeRF

def test_nerf_training_step():
    model = TriplaneNeRF(plane_resolution=16, plane_channels=8, mlp_hidden=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    points = torch.rand(1, 64, 3) * 2 - 1
    target_rgb = torch.rand(1, 64, 3)
    target_density = torch.rand(1, 64, 1)
    rgb1, density1 = model(points)
    loss1 = ((rgb1 - target_rgb) ** 2).mean() + ((density1 - target_density) ** 2).mean()
    optimizer.zero_grad()
    loss1.backward()
    optimizer.step()
    rgb2, density2 = model(points)
    loss2 = ((rgb2 - target_rgb) ** 2).mean() + ((density2 - target_density) ** 2).mean()
    assert loss2.item() < loss1.item() * 1.1
