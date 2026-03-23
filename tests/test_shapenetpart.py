import torch
import pytest
import tempfile
import os
import numpy as np
import json
from datasets.shapenetpart import ShapeNetPartDataset

@pytest.fixture
def mock_shapenetpart_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        split_file = os.path.join(tmpdir, "train_test_split", "shuffled_train_file_list.json")
        os.makedirs(os.path.dirname(split_file), exist_ok=True)
        cat_file = os.path.join(tmpdir, "synsetoffset2category.txt")
        with open(cat_file, "w") as f:
            f.write("Airplane\t02691156\n")
        data_dir = os.path.join(tmpdir, "02691156")
        os.makedirs(data_dir, exist_ok=True)
        file_list = []
        for i in range(5):
            fname = f"point_{i:04d}.txt"
            file_list.append(f"shape_data/02691156/{fname}")
            data = np.column_stack([np.random.randn(2048, 3), np.random.randn(2048, 3), np.random.randint(0, 4, 2048)])
            np.savetxt(os.path.join(data_dir, fname), data)
        with open(split_file, "w") as f:
            json.dump(file_list, f)
        yield tmpdir

def test_shapenetpart_length(mock_shapenetpart_dir):
    ds = ShapeNetPartDataset(data_root=mock_shapenetpart_dir, split="train", n_points=2048)
    assert len(ds) == 5

def test_shapenetpart_item_shape(mock_shapenetpart_dir):
    ds = ShapeNetPartDataset(data_root=mock_shapenetpart_dir, split="train", n_points=2048)
    item = ds[0]
    assert item["points"].shape == (2048, 3)
    assert item["normals"].shape == (2048, 3)
    assert item["seg_labels"].shape == (2048,)
    assert isinstance(item["category"], int)
