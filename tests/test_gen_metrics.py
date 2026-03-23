# tests/test_gen_metrics.py
import torch
from utils.metrics import chamfer_distance_batch, compute_cov, compute_mmd, compute_1nna

def test_chamfer_distance_self():
    pc = torch.randn(5, 256, 3)
    cd = chamfer_distance_batch(pc, pc)
    assert cd.shape == (5,)
    assert torch.allclose(cd, torch.zeros(5), atol=1e-3)

def test_chamfer_distance_positive():
    pc1 = torch.randn(3, 128, 3)
    pc2 = torch.randn(3, 128, 3) + 5.0
    cd = chamfer_distance_batch(pc1, pc2)
    assert (cd > 0).all()

def test_cov_score():
    ref = torch.randn(10, 64, 3)
    gen = ref.clone()
    cov = compute_cov(gen, ref)
    assert cov == 100.0

def test_mmd_perfect():
    ref = torch.randn(10, 64, 3)
    mmd = compute_mmd(ref, ref)
    assert mmd < 0.01

def test_1nna_random():
    ref = torch.randn(50, 32, 3)
    gen = torch.randn(50, 32, 3)
    nna = compute_1nna(gen, ref)
    assert 20 < nna < 80
