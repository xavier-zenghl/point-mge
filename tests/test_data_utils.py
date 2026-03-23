import torch
import pytest
from datasets.data_utils import farthest_point_sample, knn_query, morton_sort, random_point_dropout, random_scale_shift

def test_fps_output_shape():
    pc = torch.randn(2, 1024, 3)
    idx = farthest_point_sample(pc, 64)
    assert idx.shape == (2, 64)
    assert idx.max() < 1024
    assert idx.min() >= 0

def test_fps_no_duplicate_indices():
    pc = torch.randn(1, 256, 3)
    idx = farthest_point_sample(pc, 64)
    unique = torch.unique(idx[0])
    assert len(unique) == 64

def test_knn_output_shape():
    pc = torch.randn(2, 1024, 3)
    centers = torch.randn(2, 64, 3)
    idx = knn_query(pc, centers, k=32)
    assert idx.shape == (2, 64, 32)
    assert idx.max() < 1024

def test_morton_sort():
    centers = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]])
    sorted_indices = morton_sort(centers)
    assert sorted_indices.shape == (1, 4)
    assert sorted_indices[0, 0].item() == 3

def test_random_point_dropout():
    pc = torch.randn(2, 1024, 3)
    dropped = random_point_dropout(pc, max_dropout_ratio=0.5)
    assert dropped.shape == pc.shape

def test_random_scale_shift():
    pc = torch.randn(2, 1024, 3)
    transformed = random_scale_shift(pc, scale_low=0.8, scale_high=1.2, shift_range=0.1)
    assert transformed.shape == pc.shape
    assert not torch.allclose(transformed, pc)
