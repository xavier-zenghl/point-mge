import torch
import pytest
import tempfile
import os
import numpy as np
import h5py
from datasets.scanobjectnn import ScanObjectNNDataset

@pytest.fixture
def mock_scan_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = os.path.join(tmpdir, "training_objectdataset_augmentedrot_scale75.h5")
        n = 20
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("data", data=np.random.randn(n, 2048, 3).astype(np.float32))
            f.create_dataset("label", data=np.random.randint(0, 15, (n,)).astype(np.int64))
        yield tmpdir

def test_scanobjectnn_length(mock_scan_dir):
    ds = ScanObjectNNDataset(data_root=mock_scan_dir, split="train", variant="hardest")
    assert len(ds) == 20

def test_scanobjectnn_item_shape(mock_scan_dir):
    ds = ScanObjectNNDataset(data_root=mock_scan_dir, split="train", variant="hardest")
    item = ds[0]
    assert item["points"].shape == (2048, 3)
    assert isinstance(item["label"], int)
