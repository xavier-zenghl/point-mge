import torch
import pytest
import tempfile
import os
import numpy as np
from datasets.modelnet40 import ModelNet40Dataset

@pytest.fixture
def mock_modelnet_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        split_file = os.path.join(tmpdir, "train.txt")
        pc_dir = os.path.join(tmpdir, "modelnet40_pc")
        os.makedirs(pc_dir, exist_ok=True)
        with open(split_file, "w") as f:
            for i in range(10):
                label = i % 40
                fname = f"sample_{i:04d}"
                f.write(f"{fname}\t{label}\n")
                pc = np.random.randn(10000, 3).astype(np.float32)
                np.save(os.path.join(pc_dir, f"{fname}.npy"), pc)
        yield tmpdir

def test_modelnet40_length(mock_modelnet_dir):
    ds = ModelNet40Dataset(data_root=mock_modelnet_dir, pc_dir="modelnet40_pc", split="train", n_points=1024)
    assert len(ds) == 10

def test_modelnet40_item_shape(mock_modelnet_dir):
    ds = ModelNet40Dataset(data_root=mock_modelnet_dir, pc_dir="modelnet40_pc", split="train", n_points=1024)
    item = ds[0]
    assert item["points"].shape == (1024, 3)
    assert isinstance(item["label"], int)
    assert 0 <= item["label"] < 40
