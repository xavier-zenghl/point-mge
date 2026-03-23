import os
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset

VARIANT_FILES = {
    "hardest": {"train": "training_objectdataset_augmentedrot_scale75.h5", "test": "test_objectdataset_augmentedrot_scale75.h5"},
    "obj_bg": {"train": "training_objectdataset.h5", "test": "test_objectdataset.h5"},
    "obj_only": {"train": "training_objectdataset_augmented25rot.h5", "test": "test_objectdataset_augmented25rot.h5"},
}


class ScanObjectNNDataset(Dataset):
    NUM_CLASSES = 15

    def __init__(self, data_root: str, split: str = "train", variant: str = "hardest", n_points: int = 2048):
        assert variant in VARIANT_FILES, f"Unknown variant: {variant}"
        h5_file = os.path.join(data_root, VARIANT_FILES[variant][split])
        with h5py.File(h5_file, "r") as f:
            self.points = np.array(f["data"][:]).astype(np.float32)
            self.labels = np.array(f["label"][:]).astype(np.int64)
        self.n_points = n_points

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pc = self.points[idx][:self.n_points].copy()
        centroid = pc.mean(axis=0)
        pc = pc - centroid
        max_dist = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        if max_dist > 0:
            pc = pc / max_dist
        return {"points": torch.from_numpy(pc).float(), "label": int(self.labels[idx])}
