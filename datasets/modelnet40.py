import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ModelNet40Dataset(Dataset):
    def __init__(self, data_root: str, pc_dir: str = "modelnet40_pc", split: str = "train", n_points: int = 1024, augment: bool = False):
        self.data_root = data_root
        self.pc_dir = os.path.join(data_root, pc_dir)
        self.n_points = n_points
        self.augment = augment
        split_file = os.path.join(data_root, f"{split}.txt")
        self.samples = []
        with open(split_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                self.samples.append((parts[0], int(parts[1])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, label = self.samples[idx]
        pc = np.load(os.path.join(self.pc_dir, f"{name}.npy")).astype(np.float32)
        if pc.shape[0] > self.n_points:
            choice = np.random.choice(pc.shape[0], self.n_points, replace=False)
            pc = pc[choice]
        centroid = pc.mean(axis=0)
        pc = pc - centroid
        max_dist = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        if max_dist > 0:
            pc = pc / max_dist
        pc = torch.from_numpy(pc).float()
        return {"points": pc, "label": label}
