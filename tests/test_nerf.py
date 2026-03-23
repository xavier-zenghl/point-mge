import torch
from models.nerf import TriplaneNeRF

def test_triplane_nerf_output():
    model = TriplaneNeRF(plane_resolution=64, plane_channels=32, mlp_hidden=128)
    points = torch.randn(2, 100, 3)
    rgb, density = model(points)
    assert rgb.shape == (2, 100, 3)
    assert density.shape == (2, 100, 1)

def test_triplane_feature_extractor():
    model = TriplaneNeRF(plane_resolution=64, plane_channels=32, mlp_hidden=128)
    features = model.get_triplane_features()
    assert features.shape == (3, 32, 64, 64)

def test_triplane_nerf_gradient():
    model = TriplaneNeRF(plane_resolution=32, plane_channels=16, mlp_hidden=64)
    points = torch.randn(1, 50, 3)
    rgb, density = model(points)
    loss = rgb.sum() + density.sum()
    loss.backward()
    assert model.planes.grad is not None
