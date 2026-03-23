import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class ShapeNetPartDataset(Dataset):
    NUM_CATEGORIES = 16
    NUM_PARTS = 50

    def __init__(self, data_root: str, split: str = "train", n_points: int = 2048):
        self.data_root = data_root
        self.n_points = n_points
        cat_file = os.path.join(data_root, "synsetoffset2category.txt")
        self.categories = {}
        with open(cat_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                self.categories[parts[1]] = parts[0]
        self.cat2idx = {k: i for i, k in enumerate(sorted(self.categories.keys()))}
        split_map = {"train": "shuffled_train_file_list.json", "val": "shuffled_val_file_list.json", "test": "shuffled_test_file_list.json"}
        split_file = os.path.join(data_root, "train_test_split", split_map[split])
        with open(split_file, "r") as f:
            file_list = json.load(f)
        self.data = []
        for entry in file_list:
            parts = entry.split("/")
            synset = parts[1]
            fname = parts[2]
            self.data.append((synset, fname))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        synset, fname = self.data[idx]
        filepath = os.path.join(self.data_root, synset, fname)
        raw = np.loadtxt(filepath).astype(np.float32)
        pc = raw[:, :3]
        normals = raw[:, 3:6]
        seg_labels = raw[:, 6].astype(np.int64)
        if pc.shape[0] > self.n_points:
            choice = np.random.choice(pc.shape[0], self.n_points, replace=False)
            pc = pc[choice]
            normals = normals[choice]
            seg_labels = seg_labels[choice]
        centroid = pc.mean(axis=0)
        pc = pc - centroid
        max_dist = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        if max_dist > 0:
            pc = pc / max_dist
        return {
            "points": torch.from_numpy(pc).float(),
            "normals": torch.from_numpy(normals).float(),
            "seg_labels": torch.from_numpy(seg_labels).long(),
            "category": self.cat2idx[synset],
        }
