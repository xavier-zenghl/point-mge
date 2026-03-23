import torch
import pytest
import tempfile
import os
import numpy as np
from datasets.shapenet import ShapeNet55Dataset

@pytest.fixture
def mock_shapenet_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        split_file = os.path.join(tmpdir, "train.txt")
        with open(split_file, "w") as f:
            for i in range(10):
                f.write(f"02691156-model_{i:04d}\n")
        pc_dir = os.path.join(tmpdir, "shapenet_pc")
        os.makedirs(pc_dir, exist_ok=True)
        for i in range(10):
            pc = np.random.randn(8192, 3).astype(np.float32)
            np.save(os.path.join(pc_dir, f"02691156-model_{i:04d}.npy"), pc)
        yield tmpdir

def test_shapenet_dataset_length(mock_shapenet_dir):
    ds = ShapeNet55Dataset(data_root=mock_shapenet_dir, pc_dir="shapenet_pc", split="train", n_points=2048)
    assert len(ds) == 10

def test_shapenet_dataset_item_shape(mock_shapenet_dir):
    ds = ShapeNet55Dataset(data_root=mock_shapenet_dir, pc_dir="shapenet_pc", split="train", n_points=2048)
    item = ds[0]
    assert item["points"].shape == (2048, 3)
    assert isinstance(item["taxonomy_id"], str)
    assert isinstance(item["model_id"], str)

def test_shapenet_dataset_normalization(mock_shapenet_dir):
    ds = ShapeNet55Dataset(data_root=mock_shapenet_dir, pc_dir="shapenet_pc", split="train", n_points=2048)
    item = ds[0]
    pc = item["points"]
    assert abs(pc.mean(dim=0)).max() < 0.5
