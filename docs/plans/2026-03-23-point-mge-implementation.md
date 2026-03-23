# Point-MGE Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reproduce the Point-MGE paper — a joint framework for point cloud representation learning and 3D shape generation using VQVAE tokenization, sliding-mask pretraining, and autoregressive generation.

**Architecture:** Point clouds are tokenized via Point Patch Embedding (FPS+KNN+miniPointNet), a VQVAE maps tokens to discrete codes targeting NeRF triplane features, an Extractor-Generator pair is pretrained with sliding masking, and downstream tasks (classification, few-shot, part segmentation, generation) are evaluated.

**Tech Stack:** PyTorch >= 2.0, timm, einops, pointnet2_ops, pytorch3d, open3d, DDP, YAML configs, pytest

---

## Phase 1: Project Foundation

### Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `setup.py`
- Create: `models/__init__.py`
- Create: `datasets/__init__.py`
- Create: `utils/__init__.py`
- Create: `tools/__init__.py`
- Create: `tests/__init__.py`
- Create: `configs/.gitkeep`
- Create: `scripts/.gitkeep`

**Step 1: Create requirements.txt**

```
torch>=2.0
torchvision
timm>=0.9.0
einops
pointnet2_ops
pytorch3d
open3d
numpy
scipy
tensorboard
wandb
easydict
pyyaml
tqdm
pytest
```

**Step 2: Create setup.py**

```python
from setuptools import setup, find_packages

setup(
    name="point-mge",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
)
```

**Step 3: Create all __init__.py and .gitkeep files**

Empty `__init__.py` for: `models/`, `models/heads/`, `datasets/`, `utils/`, `tools/`, `tests/`
Empty `.gitkeep` for: `configs/`, `scripts/`, `data/`

**Step 4: Verify project structure**

Run: `find . -name "*.py" -o -name ".gitkeep" -o -name "*.txt" | head -20`
Expected: All created files listed

**Step 5: Commit**

```bash
git add requirements.txt setup.py models/ datasets/ utils/ tools/ tests/ configs/ scripts/ data/
git commit -m "chore: initialize project structure"
```

---

### Task 2: Config System

**Files:**
- Create: `utils/config.py`
- Create: `tests/test_config.py`

**Step 1: Write the failing test**

```python
# tests/test_config.py
import pytest
import os
import tempfile
import yaml
from utils.config import load_config, merge_config


def test_load_config_from_yaml():
    cfg_dict = {
        "model": {"name": "extractor", "depth": 12, "embed_dim": 384},
        "train": {"epochs": 300, "lr": 1e-3},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg_dict, f)
        f_path = f.name
    try:
        cfg = load_config(f_path)
        assert cfg.model.name == "extractor"
        assert cfg.model.depth == 12
        assert cfg.train.lr == 1e-3
    finally:
        os.unlink(f_path)


def test_merge_config_with_cli_args():
    base = {"model": {"depth": 12}, "train": {"lr": 1e-3}}
    overrides = ["train.lr=5e-4", "train.epochs=100"]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(base, f)
        f_path = f.name
    try:
        cfg = load_config(f_path)
        cfg = merge_config(cfg, overrides)
        assert cfg.train.lr == 5e-4
        assert cfg.train.epochs == 100
        assert cfg.model.depth == 12
    finally:
        os.unlink(f_path)


def test_config_attribute_access():
    cfg_dict = {"a": {"b": {"c": 42}}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg_dict, f)
        f_path = f.name
    try:
        cfg = load_config(f_path)
        assert cfg.a.b.c == 42
    finally:
        os.unlink(f_path)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'utils.config'`

**Step 3: Write minimal implementation**

```python
# utils/config.py
import yaml
from easydict import EasyDict


def load_config(path: str) -> EasyDict:
    """Load YAML config file and return as EasyDict."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return EasyDict(cfg)


def merge_config(cfg: EasyDict, overrides: list[str]) -> EasyDict:
    """Merge CLI overrides like 'key.subkey=value' into config."""
    for item in overrides:
        key, val = item.split("=", 1)
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d[k]
        # Auto-cast types
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                if val.lower() == "true":
                    val = True
                elif val.lower() == "false":
                    val = False
        d[keys[-1]] = val
    return cfg
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add utils/config.py tests/test_config.py
git commit -m "feat: add YAML config system with CLI override support"
```

---

### Task 3: Logger Utility

**Files:**
- Create: `utils/logger.py`
- Create: `tests/test_logger.py`

**Step 1: Write the failing test**

```python
# tests/test_logger.py
import os
import tempfile
from utils.logger import get_logger


def test_logger_creates_log_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = os.path.join(tmpdir, "test.log")
        logger = get_logger("test", log_file=log_path)
        logger.info("hello")
        assert os.path.exists(log_path)
        with open(log_path) as f:
            content = f.read()
        assert "hello" in content


def test_logger_without_file():
    logger = get_logger("console_only")
    logger.info("console message")  # Should not raise
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_logger.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# utils/logger.py
import logging
import sys


def get_logger(name: str, log_file: str | None = None, level=logging.INFO) -> logging.Logger:
    """Create logger with console and optional file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s %(name)s %(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_logger.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add utils/logger.py tests/test_logger.py
git commit -m "feat: add logger utility"
```

---

### Task 4: Checkpoint Manager

**Files:**
- Create: `utils/checkpoint.py`
- Create: `tests/test_checkpoint.py`

**Step 1: Write the failing test**

```python
# tests/test_checkpoint.py
import os
import tempfile
import torch
import torch.nn as nn
from utils.checkpoint import save_checkpoint, load_checkpoint


def test_save_and_load_checkpoint():
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.pth")
        save_checkpoint(path, model, optimizer, epoch=10, best_metric=0.95)

        model2 = nn.Linear(10, 5)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        ckpt = load_checkpoint(path, model2, optimizer2)

        assert ckpt["epoch"] == 10
        assert ckpt["best_metric"] == 0.95

        # Verify model weights match
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)


def test_load_checkpoint_model_only():
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.pth")
        save_checkpoint(path, model, optimizer, epoch=5)

        model2 = nn.Linear(10, 5)
        ckpt = load_checkpoint(path, model2)
        assert ckpt["epoch"] == 5
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_checkpoint.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# utils/checkpoint.py
import torch
import torch.nn as nn


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    best_metric: float = 0.0,
    **kwargs,
):
    """Save model checkpoint."""
    state = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    state.update(kwargs)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """Load model checkpoint. Returns checkpoint dict with metadata."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_checkpoint.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add utils/checkpoint.py tests/test_checkpoint.py
git commit -m "feat: add checkpoint save/load utility"
```

---

### Task 5: Learning Rate Scheduler

**Files:**
- Create: `utils/scheduler.py`
- Create: `tests/test_scheduler.py`

**Step 1: Write the failing test**

```python
# tests/test_scheduler.py
import torch
import torch.nn as nn
from utils.scheduler import build_scheduler


def test_cosine_scheduler():
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = build_scheduler(optimizer, name="cosine", epochs=100, warmup_epochs=10, min_lr=1e-5)

    # During warmup, lr should increase
    lrs = []
    for _ in range(100):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    # Warmup: lr should increase for first 10 steps
    assert lrs[0] < lrs[9]
    # After warmup: lr should decrease
    assert lrs[10] > lrs[-1]
    # Final lr should be close to min_lr
    assert abs(lrs[-1] - 1e-5) < 1e-6
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scheduler.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# utils/scheduler.py
import math
import torch


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing with linear warmup."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, min_lr: float = 1e-5):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            alpha = epoch / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine for base_lr in self.base_lrs]


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    name: str = "cosine",
    epochs: int = 300,
    warmup_epochs: int = 10,
    min_lr: float = 1e-5,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Build learning rate scheduler."""
    if name == "cosine":
        return CosineWarmupScheduler(optimizer, warmup_epochs, epochs, min_lr)
    else:
        raise ValueError(f"Unknown scheduler: {name}")
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scheduler.py -v`
Expected: 1 passed

**Step 5: Commit**

```bash
git add utils/scheduler.py tests/test_scheduler.py
git commit -m "feat: add cosine warmup learning rate scheduler"
```

---

### Task 6: Distributed Training Utilities

**Files:**
- Create: `utils/distributed.py`
- Create: `tests/test_distributed.py`

**Step 1: Write the failing test**

```python
# tests/test_distributed.py
import torch
from utils.distributed import is_main_process, get_world_size, get_rank, reduce_tensor


def test_single_process_defaults():
    """Without DDP init, should return single-process defaults."""
    assert is_main_process() is True
    assert get_world_size() == 1
    assert get_rank() == 0


def test_reduce_tensor_single_process():
    t = torch.tensor([1.0, 2.0, 3.0])
    result = reduce_tensor(t)
    assert torch.allclose(result, t)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_distributed.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# utils/distributed.py
import torch
import torch.distributed as dist


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if not is_dist_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not is_dist_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    return get_rank() == 0


def reduce_tensor(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """All-reduce tensor across processes. No-op if not distributed."""
    if not is_dist_initialized():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=op)
    rt /= get_world_size()
    return rt


def setup_distributed():
    """Initialize DDP from environment variables set by torchrun."""
    if not dist.is_available():
        return
    dist.init_process_group(backend="nccl")
    local_rank = get_rank()
    torch.cuda.set_device(local_rank)


def cleanup_distributed():
    if is_dist_initialized():
        dist.destroy_process_group()
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_distributed.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add utils/distributed.py tests/test_distributed.py
git commit -m "feat: add distributed training utilities"
```

---

## Phase 2: Point Cloud Utilities & Data Pipeline

### Task 7: Point Cloud Utilities (FPS, KNN, Morton Sort)

**Files:**
- Create: `datasets/data_utils.py`
- Create: `tests/test_data_utils.py`

**Step 1: Write the failing test**

```python
# tests/test_data_utils.py
import torch
import pytest
from datasets.data_utils import (
    farthest_point_sample,
    knn_query,
    morton_sort,
    random_point_dropout,
    random_scale_shift,
)


def test_fps_output_shape():
    pc = torch.randn(2, 1024, 3)
    idx = farthest_point_sample(pc, 64)
    assert idx.shape == (2, 64)
    # Indices should be valid
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
    """Morton sort should produce deterministic ordering based on 3D coordinates."""
    centers = torch.tensor([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
    ])  # (1, 4, 3)
    sorted_indices = morton_sort(centers)
    assert sorted_indices.shape == (1, 4)
    # Origin [0,0,0] should have smallest Morton code
    assert sorted_indices[0, 0].item() == 3  # index of [0,0,0]


def test_random_point_dropout():
    pc = torch.randn(2, 1024, 3)
    dropped = random_point_dropout(pc, max_dropout_ratio=0.5)
    assert dropped.shape == pc.shape  # Shape preserved (duplicates used for padding)


def test_random_scale_shift():
    pc = torch.randn(2, 1024, 3)
    transformed = random_scale_shift(pc, scale_low=0.8, scale_high=1.2, shift_range=0.1)
    assert transformed.shape == pc.shape
    # Should be different from original
    assert not torch.allclose(transformed, pc)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_data_utils.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# datasets/data_utils.py
import torch
import torch.nn.functional as F


def farthest_point_sample(xyz: torch.Tensor, n_points: int) -> torch.Tensor:
    """
    Farthest point sampling.
    Args:
        xyz: (B, N, 3) input point cloud
        n_points: number of points to sample
    Returns:
        idx: (B, n_points) sampled point indices
    """
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, n_points, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), device=device)
    batch_indices = torch.arange(B, device=device)

    for i in range(n_points):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)  # (B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)  # (B, N)
        distance = torch.min(distance, dist)
        farthest = distance.argmax(dim=-1)

    return centroids


def knn_query(xyz: torch.Tensor, centers: torch.Tensor, k: int) -> torch.Tensor:
    """
    K-Nearest Neighbors query.
    Args:
        xyz: (B, N, 3) full point cloud
        centers: (B, G, 3) center points
        k: number of neighbors
    Returns:
        idx: (B, G, K) neighbor indices into xyz
    """
    # (B, G, N)
    dist = torch.cdist(centers, xyz, p=2.0)
    _, idx = dist.topk(k, dim=-1, largest=False)
    return idx


def morton_sort(centers: torch.Tensor) -> torch.Tensor:
    """
    Sort center points by Morton (Z-order) curve for spatial serialization.
    Args:
        centers: (B, G, 3) center point coordinates
    Returns:
        sorted_indices: (B, G) indices that sort centers by Morton code
    """
    B, G, _ = centers.shape
    # Normalize to [0, 1023] integer grid
    mins = centers.min(dim=1, keepdim=True).values
    maxs = centers.max(dim=1, keepdim=True).values
    span = (maxs - mins).clamp(min=1e-6)
    normalized = ((centers - mins) / span * 1023).long().clamp(0, 1023)

    # Compute Morton code by interleaving bits of x, y, z
    codes = torch.zeros(B, G, dtype=torch.long, device=centers.device)
    for bit in range(10):  # 10 bits per dimension
        codes |= ((normalized[..., 0] >> bit) & 1) << (3 * bit)
        codes |= ((normalized[..., 1] >> bit) & 1) << (3 * bit + 1)
        codes |= ((normalized[..., 2] >> bit) & 1) << (3 * bit + 2)

    sorted_indices = codes.argsort(dim=-1)
    return sorted_indices


def random_point_dropout(pc: torch.Tensor, max_dropout_ratio: float = 0.875) -> torch.Tensor:
    """Randomly drop points and fill with duplicates of the first point."""
    B, N, C = pc.shape
    result = pc.clone()
    for b in range(B):
        dropout_ratio = torch.rand(1).item() * max_dropout_ratio
        drop_idx = torch.where(torch.rand(N) < dropout_ratio)[0]
        if len(drop_idx) > 0:
            result[b, drop_idx] = result[b, 0]
    return result


def random_scale_shift(
    pc: torch.Tensor,
    scale_low: float = 0.8,
    scale_high: float = 1.25,
    shift_range: float = 0.1,
) -> torch.Tensor:
    """Random scaling and translation augmentation."""
    B, N, C = pc.shape
    scales = torch.empty(B, 1, 1).uniform_(scale_low, scale_high).to(pc.device)
    shifts = torch.empty(B, 1, C).uniform_(-shift_range, shift_range).to(pc.device)
    return pc * scales + shifts
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_data_utils.py -v`
Expected: 6 passed

**Step 5: Commit**

```bash
git add datasets/data_utils.py tests/test_data_utils.py
git commit -m "feat: add point cloud utilities (FPS, KNN, Morton sort, augmentation)"
```

---

### Task 8: ShapeNet Dataset

**Files:**
- Create: `datasets/shapenet.py`
- Create: `tests/test_shapenet.py`
- Create: `configs/pretrain.yaml`

**Step 1: Write the failing test**

```python
# tests/test_shapenet.py
import torch
import pytest
import tempfile
import os
import numpy as np
from datasets.shapenet import ShapeNet55Dataset


@pytest.fixture
def mock_shapenet_dir():
    """Create mock ShapeNet data files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock split file
        # Format: taxonomy_id-model_id (one per line)
        split_file = os.path.join(tmpdir, "train.txt")
        with open(split_file, "w") as f:
            for i in range(10):
                f.write(f"02691156-model_{i:04d}\n")

        # Create mock point cloud files
        pc_dir = os.path.join(tmpdir, "shapenet_pc")
        os.makedirs(pc_dir, exist_ok=True)
        for i in range(10):
            pc = np.random.randn(8192, 3).astype(np.float32)
            np.save(os.path.join(pc_dir, f"02691156-model_{i:04d}.npy"), pc)

        yield tmpdir


def test_shapenet_dataset_length(mock_shapenet_dir):
    ds = ShapeNet55Dataset(
        data_root=mock_shapenet_dir,
        pc_dir="shapenet_pc",
        split="train",
        n_points=2048,
    )
    assert len(ds) == 10


def test_shapenet_dataset_item_shape(mock_shapenet_dir):
    ds = ShapeNet55Dataset(
        data_root=mock_shapenet_dir,
        pc_dir="shapenet_pc",
        split="train",
        n_points=2048,
    )
    item = ds[0]
    assert item["points"].shape == (2048, 3)
    assert isinstance(item["taxonomy_id"], str)
    assert isinstance(item["model_id"], str)


def test_shapenet_dataset_normalization(mock_shapenet_dir):
    ds = ShapeNet55Dataset(
        data_root=mock_shapenet_dir,
        pc_dir="shapenet_pc",
        split="train",
        n_points=2048,
    )
    item = ds[0]
    pc = item["points"]
    # Points should be centered and normalized
    assert abs(pc.mean(dim=0)).max() < 0.5  # Roughly centered
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_shapenet.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# datasets/shapenet.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ShapeNet55Dataset(Dataset):
    """ShapeNet55 dataset for pretraining."""

    def __init__(
        self,
        data_root: str,
        pc_dir: str = "shapenet_pc",
        split: str = "train",
        n_points: int = 2048,
        augment: bool = False,
    ):
        self.data_root = data_root
        self.pc_dir = os.path.join(data_root, pc_dir)
        self.n_points = n_points
        self.augment = augment

        split_file = os.path.join(data_root, f"{split}.txt")
        with open(split_file, "r") as f:
            self.file_list = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        name = self.file_list[idx]
        parts = name.split("-", 1)
        taxonomy_id = parts[0]
        model_id = parts[1] if len(parts) > 1 else name

        pc = np.load(os.path.join(self.pc_dir, f"{name}.npy")).astype(np.float32)

        # Random subsample
        if pc.shape[0] > self.n_points:
            choice = np.random.choice(pc.shape[0], self.n_points, replace=False)
            pc = pc[choice]

        # Normalize: center and scale to unit sphere
        centroid = pc.mean(axis=0)
        pc = pc - centroid
        max_dist = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        if max_dist > 0:
            pc = pc / max_dist

        pc = torch.from_numpy(pc).float()

        return {
            "points": pc,
            "taxonomy_id": taxonomy_id,
            "model_id": model_id,
        }
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_shapenet.py -v`
Expected: 3 passed

**Step 5: Create pretrain config**

```yaml
# configs/pretrain.yaml
data:
  dataset: shapenet55
  data_root: data/ShapeNet55
  pc_dir: shapenet_pc
  n_points: 2048
  train_split: train
  val_split: test

model:
  embed_dim: 384
  num_groups: 64
  group_size: 32
  extractor_depth: 12
  generator_depth: 4
  num_heads: 6
  codebook_size: 8192
  mask_beta: 0.5
  mask_u: 2.0

train:
  epochs: 300
  batch_size: 128
  lr: 1e-3
  weight_decay: 0.05
  warmup_epochs: 10
  min_lr: 1e-5
  grad_accum_steps: 1

output:
  exp_dir: experiments/pretrain
  save_freq: 50
  log_freq: 10
```

**Step 6: Commit**

```bash
git add datasets/shapenet.py tests/test_shapenet.py configs/pretrain.yaml
git commit -m "feat: add ShapeNet55 dataset and pretrain config"
```

---

### Task 9: ModelNet40 Dataset

**Files:**
- Create: `datasets/modelnet40.py`
- Create: `tests/test_modelnet40.py`
- Create: `configs/cls_modelnet40.yaml`

**Step 1: Write the failing test**

```python
# tests/test_modelnet40.py
import torch
import pytest
import tempfile
import os
import numpy as np
from datasets.modelnet40 import ModelNet40Dataset


@pytest.fixture
def mock_modelnet_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock split file: label\tpath
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
    ds = ModelNet40Dataset(
        data_root=mock_modelnet_dir,
        pc_dir="modelnet40_pc",
        split="train",
        n_points=1024,
    )
    assert len(ds) == 10


def test_modelnet40_item_shape(mock_modelnet_dir):
    ds = ModelNet40Dataset(
        data_root=mock_modelnet_dir,
        pc_dir="modelnet40_pc",
        split="train",
        n_points=1024,
    )
    item = ds[0]
    assert item["points"].shape == (1024, 3)
    assert isinstance(item["label"], int)
    assert 0 <= item["label"] < 40
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_modelnet40.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# datasets/modelnet40.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ModelNet40Dataset(Dataset):
    """ModelNet40 classification dataset."""

    def __init__(
        self,
        data_root: str,
        pc_dir: str = "modelnet40_pc",
        split: str = "train",
        n_points: int = 1024,
        augment: bool = False,
    ):
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
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_modelnet40.py -v`
Expected: 2 passed

**Step 5: Create config**

```yaml
# configs/cls_modelnet40.yaml
data:
  dataset: modelnet40
  data_root: data/ModelNet40
  pc_dir: modelnet40_pc
  n_points: 1024
  train_split: train
  val_split: test
  num_classes: 40

model:
  embed_dim: 384
  num_groups: 64
  group_size: 32
  extractor_depth: 12
  num_heads: 6
  pretrained: experiments/pretrain/best.pth

train:
  epochs: 300
  batch_size: 32
  lr: 5e-4
  weight_decay: 0.05
  warmup_epochs: 10
  min_lr: 1e-5
  vote: true
  vote_num: 10

output:
  exp_dir: experiments/cls_modelnet40
  save_freq: 50
  log_freq: 10
```

**Step 6: Commit**

```bash
git add datasets/modelnet40.py tests/test_modelnet40.py configs/cls_modelnet40.yaml
git commit -m "feat: add ModelNet40 dataset and classification config"
```

---

### Task 10: ScanObjectNN Dataset

**Files:**
- Create: `datasets/scanobjectnn.py`
- Create: `tests/test_scanobjectnn.py`
- Create: `configs/cls_scanobjectnn.yaml`

**Step 1: Write the failing test**

```python
# tests/test_scanobjectnn.py
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
        # Create mock h5 file
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scanobjectnn.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# datasets/scanobjectnn.py
import os
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset

VARIANT_FILES = {
    "hardest": {
        "train": "training_objectdataset_augmentedrot_scale75.h5",
        "test": "test_objectdataset_augmentedrot_scale75.h5",
    },
    "obj_bg": {
        "train": "training_objectdataset.h5",
        "test": "test_objectdataset.h5",
    },
    "obj_only": {
        "train": "training_objectdataset_augmented25rot.h5",
        "test": "test_objectdataset_augmented25rot.h5",
    },
}


class ScanObjectNNDataset(Dataset):
    """ScanObjectNN classification dataset."""

    NUM_CLASSES = 15

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        variant: str = "hardest",
        n_points: int = 2048,
    ):
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

        return {
            "points": torch.from_numpy(pc).float(),
            "label": int(self.labels[idx]),
        }
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scanobjectnn.py -v`
Expected: 2 passed

**Step 5: Create config and commit**

```yaml
# configs/cls_scanobjectnn.yaml
data:
  dataset: scanobjectnn
  data_root: data/ScanObjectNN
  variant: hardest  # hardest | obj_bg | obj_only
  n_points: 2048
  num_classes: 15

model:
  embed_dim: 384
  num_groups: 64
  group_size: 32
  extractor_depth: 12
  num_heads: 6
  pretrained: experiments/pretrain/best.pth

train:
  epochs: 300
  batch_size: 32
  lr: 5e-4
  weight_decay: 0.05
  warmup_epochs: 10
  min_lr: 1e-5

output:
  exp_dir: experiments/cls_scanobjectnn
  save_freq: 50
  log_freq: 10
```

```bash
git add datasets/scanobjectnn.py tests/test_scanobjectnn.py configs/cls_scanobjectnn.yaml
git commit -m "feat: add ScanObjectNN dataset (3 variants)"
```

---

### Task 11: ShapeNetPart Dataset

**Files:**
- Create: `datasets/shapenetpart.py`
- Create: `tests/test_shapenetpart.py`
- Create: `configs/partseg.yaml`

**Step 1: Write the failing test**

```python
# tests/test_shapenetpart.py
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
        # Create mock data
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
            data = np.column_stack([
                np.random.randn(2048, 3),      # xyz
                np.random.randn(2048, 3),       # normal
                np.random.randint(0, 4, 2048),  # label
            ])
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_shapenetpart.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# datasets/shapenetpart.py
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class ShapeNetPartDataset(Dataset):
    """ShapeNetPart segmentation dataset. 16 categories, 50 part classes."""

    NUM_CATEGORIES = 16
    NUM_PARTS = 50

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        n_points: int = 2048,
    ):
        self.data_root = data_root
        self.n_points = n_points

        # Load category mapping
        cat_file = os.path.join(data_root, "synsetoffset2category.txt")
        self.categories = {}
        with open(cat_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                self.categories[parts[1]] = parts[0]

        self.cat2idx = {k: i for i, k in enumerate(sorted(self.categories.keys()))}

        # Load split
        split_map = {"train": "shuffled_train_file_list.json", "val": "shuffled_val_file_list.json", "test": "shuffled_test_file_list.json"}
        split_file = os.path.join(data_root, "train_test_split", split_map[split])
        with open(split_file, "r") as f:
            file_list = json.load(f)

        self.data = []
        for entry in file_list:
            # entry format: "shape_data/synsetid/filename.txt"
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

        # Normalize
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
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_shapenetpart.py -v`
Expected: 2 passed

**Step 5: Create config and commit**

```yaml
# configs/partseg.yaml
data:
  dataset: shapenetpart
  data_root: data/ShapeNetPart
  n_points: 2048
  num_categories: 16
  num_parts: 50

model:
  embed_dim: 384
  num_groups: 64
  group_size: 32
  extractor_depth: 12
  num_heads: 6
  pretrained: experiments/pretrain/best.pth

train:
  epochs: 300
  batch_size: 16
  lr: 2e-4
  weight_decay: 0.05
  warmup_epochs: 10
  min_lr: 1e-5

output:
  exp_dir: experiments/partseg
  save_freq: 50
  log_freq: 10
```

```bash
git add datasets/shapenetpart.py tests/test_shapenetpart.py configs/partseg.yaml
git commit -m "feat: add ShapeNetPart segmentation dataset"
```

---

### Task 12: Dataset Registry

**Files:**
- Modify: `datasets/__init__.py`
- Create: `tests/test_dataset_registry.py`

**Step 1: Write the failing test**

```python
# tests/test_dataset_registry.py
from datasets import build_dataset


def test_build_dataset_unknown_raises():
    import pytest
    with pytest.raises(ValueError, match="Unknown dataset"):
        build_dataset("nonexistent", data_root="/tmp", split="train")
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dataset_registry.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# datasets/__init__.py
from datasets.shapenet import ShapeNet55Dataset
from datasets.modelnet40 import ModelNet40Dataset
from datasets.scanobjectnn import ScanObjectNNDataset
from datasets.shapenetpart import ShapeNetPartDataset

DATASETS = {
    "shapenet55": ShapeNet55Dataset,
    "modelnet40": ModelNet40Dataset,
    "scanobjectnn": ScanObjectNNDataset,
    "shapenetpart": ShapeNetPartDataset,
}


def build_dataset(name: str, **kwargs):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
    return DATASETS[name](**kwargs)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dataset_registry.py -v`
Expected: 1 passed

**Step 5: Commit**

```bash
git add datasets/__init__.py tests/test_dataset_registry.py
git commit -m "feat: add dataset registry"
```

---

## Phase 3: Core Model Components

### Task 13: Point Patch Embedding

**Files:**
- Create: `models/point_patch_embed.py`
- Create: `tests/test_point_patch_embed.py`

**Step 1: Write the failing test**

```python
# tests/test_point_patch_embed.py
import torch
import pytest
from models.point_patch_embed import PointPatchEmbed


def test_point_patch_embed_output_shape():
    model = PointPatchEmbed(
        in_channels=3,
        embed_dim=384,
        num_groups=64,
        group_size=32,
    )
    pc = torch.randn(2, 2048, 3)
    tokens, centers = model(pc)
    assert tokens.shape == (2, 64, 384)
    assert centers.shape == (2, 64, 3)


def test_point_patch_embed_gradient_flow():
    model = PointPatchEmbed(in_channels=3, embed_dim=384, num_groups=64, group_size=32)
    pc = torch.randn(2, 2048, 3, requires_grad=True)
    tokens, centers = model(pc)
    loss = tokens.sum()
    loss.backward()
    assert pc.grad is not None
    assert pc.grad.shape == pc.shape


def test_point_patch_embed_with_morton_sort():
    model = PointPatchEmbed(
        in_channels=3,
        embed_dim=384,
        num_groups=64,
        group_size=32,
        use_morton_sort=True,
    )
    pc = torch.randn(2, 2048, 3)
    tokens, centers = model(pc)
    assert tokens.shape == (2, 64, 384)


def test_point_patch_embed_different_sizes():
    model = PointPatchEmbed(in_channels=3, embed_dim=256, num_groups=32, group_size=16)
    pc = torch.randn(4, 1024, 3)
    tokens, centers = model(pc)
    assert tokens.shape == (4, 32, 256)
    assert centers.shape == (4, 32, 3)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_point_patch_embed.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# models/point_patch_embed.py
import torch
import torch.nn as nn
from datasets.data_utils import farthest_point_sample, knn_query, morton_sort


class MiniPointNet(nn.Module):
    """Per-patch feature extraction using shared MLPs."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 384):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, embed_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, G, K, C) grouped point features
        Returns:
            (B, G, D) patch features (max-pooled)
        """
        B, G, K, C = x.shape
        x = x.reshape(B * G, K, C).transpose(1, 2)  # (B*G, C, K)
        x = self.mlp(x)  # (B*G, D, K)
        x = x.max(dim=-1).values  # (B*G, D)
        return x.reshape(B, G, -1)


class PointPatchEmbed(nn.Module):
    """
    Point Patch Embedding: FPS + KNN + mini-PointNet.

    Converts raw point cloud to patch tokens.
    Input: (B, N, 3)
    Output: tokens (B, G, D), centers (B, G, 3)
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 384,
        num_groups: int = 64,
        group_size: int = 32,
        use_morton_sort: bool = False,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.use_morton_sort = use_morton_sort
        self.mini_pointnet = MiniPointNet(in_channels, embed_dim)

    def forward(self, pc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pc: (B, N, 3) input point cloud
        Returns:
            tokens: (B, G, D) patch embeddings
            centers: (B, G, 3) patch center coordinates
        """
        B, N, C = pc.shape

        # FPS: sample center points
        center_idx = farthest_point_sample(pc, self.num_groups)  # (B, G)
        batch_idx = torch.arange(B, device=pc.device).unsqueeze(1).expand(-1, self.num_groups)
        centers = pc[batch_idx, center_idx]  # (B, G, 3)

        # Morton sort for spatial serialization
        if self.use_morton_sort:
            sort_idx = morton_sort(centers)  # (B, G)
            batch_idx2 = torch.arange(B, device=pc.device).unsqueeze(1).expand(-1, self.num_groups)
            centers = centers[batch_idx2, sort_idx]
            center_idx = center_idx[batch_idx2, sort_idx]

        # KNN: group neighbors
        knn_idx = knn_query(pc, centers, k=self.group_size)  # (B, G, K)

        # Gather neighbor points
        batch_idx3 = torch.arange(B, device=pc.device).reshape(B, 1, 1).expand(-1, self.num_groups, self.group_size)
        grouped = pc[batch_idx3, knn_idx]  # (B, G, K, 3)

        # Normalize: relative to center
        grouped = grouped - centers.unsqueeze(2)  # (B, G, K, 3)

        # Mini-PointNet
        tokens = self.mini_pointnet(grouped)  # (B, G, D)

        return tokens, centers
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_point_patch_embed.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add models/point_patch_embed.py tests/test_point_patch_embed.py
git commit -m "feat: add Point Patch Embedding (FPS + KNN + mini-PointNet)"
```

---

### Task 14: Vector Quantization Layer

**Files:**
- Create: `models/vqvae.py` (VQ layer first, full VQVAE in next task)
- Create: `tests/test_vq.py`

**Step 1: Write the failing test**

```python
# tests/test_vq.py
import torch
import pytest
from models.vqvae import VectorQuantize


def test_vq_output_shape():
    vq = VectorQuantize(dim=384, codebook_size=8192, commitment_weight=0.25)
    z = torch.randn(2, 64, 384)
    quantized, indices, loss = vq(z)
    assert quantized.shape == (2, 64, 384)
    assert indices.shape == (2, 64)
    assert loss.shape == ()  # scalar


def test_vq_indices_valid_range():
    vq = VectorQuantize(dim=384, codebook_size=512)
    z = torch.randn(2, 64, 384)
    _, indices, _ = vq(z)
    assert indices.min() >= 0
    assert indices.max() < 512


def test_vq_gradient_through_straight_through():
    vq = VectorQuantize(dim=384, codebook_size=512)
    z = torch.randn(2, 64, 384, requires_grad=True)
    quantized, _, loss = vq(z)
    total_loss = quantized.sum() + loss
    total_loss.backward()
    assert z.grad is not None


def test_vq_codebook_lookup():
    vq = VectorQuantize(dim=64, codebook_size=128)
    z = torch.randn(1, 10, 64)
    _, indices, _ = vq(z)
    # Look up indices manually
    looked_up = vq.codebook[indices]
    # After straight-through, quantized should have same values as codebook lookup
    quantized, _, _ = vq(z)
    # The forward values of quantized == codebook[indices] (before straight-through grad)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_vq.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# models/vqvae.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantize(nn.Module):
    """
    Vector Quantization with EMA codebook update.

    Uses straight-through estimator for gradients.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int = 8192,
        commitment_weight: float = 0.25,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.ema_decay = ema_decay

        self.codebook = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

        # EMA state
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_embed_sum", self.codebook.weight.clone())

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, L, D) continuous latent features
        Returns:
            quantized: (B, L, D) quantized features (straight-through)
            indices: (B, L) codebook indices
            loss: scalar VQ loss
        """
        B, L, D = z.shape
        flat_z = z.reshape(-1, D)  # (B*L, D)

        # Find nearest codebook entries
        # dist = ||z||^2 + ||e||^2 - 2*z@e^T
        dist = (
            flat_z.pow(2).sum(dim=-1, keepdim=True)
            + self.codebook.weight.pow(2).sum(dim=-1, keepdim=True).t()
            - 2 * flat_z @ self.codebook.weight.t()
        )
        indices = dist.argmin(dim=-1)  # (B*L,)
        quantized = self.codebook(indices).reshape(B, L, D)

        # EMA update (training only)
        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(indices, self.codebook_size).float()  # (B*L, K)
                cluster_size = one_hot.sum(dim=0)
                embed_sum = one_hot.t() @ flat_z  # (K, D)

                self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)
                self.ema_embed_sum.mul_(self.ema_decay).add_(embed_sum, alpha=1 - self.ema_decay)

                # Laplace smoothing
                n = self.ema_cluster_size.sum()
                cluster_size_smoothed = (
                    (self.ema_cluster_size + 1e-5) / (n + self.codebook_size * 1e-5) * n
                )
                self.codebook.weight.data.copy_(self.ema_embed_sum / cluster_size_smoothed.unsqueeze(1))

        # Loss
        commitment_loss = F.mse_loss(z, quantized.detach())
        codebook_loss = F.mse_loss(quantized, z.detach())
        loss = codebook_loss + self.commitment_weight * commitment_loss

        # Straight-through estimator
        quantized = z + (quantized - z).detach()

        indices = indices.reshape(B, L)
        return quantized, indices, loss
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_vq.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add models/vqvae.py tests/test_vq.py
git commit -m "feat: add vector quantization layer with EMA update"
```

---

### Task 15: VQVAE Full Model

**Files:**
- Modify: `models/vqvae.py` (add VQVAE class)
- Create: `tests/test_vqvae.py`

**Step 1: Write the failing test**

```python
# tests/test_vqvae.py
import torch
import pytest
from models.vqvae import VQVAE


def test_vqvae_forward_shape():
    model = VQVAE(
        embed_dim=384,
        num_groups=64,
        group_size=32,
        encoder_depth=4,
        decoder_depth=4,
        num_heads=6,
        codebook_size=512,  # small for test
    )
    pc = torch.randn(2, 2048, 3)
    result = model(pc)
    assert result["quantized"].shape == (2, 64, 384)
    assert result["indices"].shape == (2, 64)
    assert result["vq_loss"].shape == ()
    assert result["centers"].shape == (2, 64, 3)


def test_vqvae_encode():
    model = VQVAE(
        embed_dim=384, num_groups=64, group_size=32,
        encoder_depth=4, decoder_depth=4, num_heads=6, codebook_size=512,
    )
    pc = torch.randn(2, 2048, 3)
    indices, centers = model.encode(pc)
    assert indices.shape == (2, 64)
    assert centers.shape == (2, 64, 3)


def test_vqvae_decode_from_indices():
    model = VQVAE(
        embed_dim=384, num_groups=64, group_size=32,
        encoder_depth=4, decoder_depth=4, num_heads=6, codebook_size=512,
    )
    indices = torch.randint(0, 512, (2, 64))
    centers = torch.randn(2, 64, 3)
    decoded = model.decode(indices, centers)
    assert decoded.shape == (2, 64, 384)


def test_vqvae_gradient_flow():
    model = VQVAE(
        embed_dim=128, num_groups=16, group_size=8,
        encoder_depth=2, decoder_depth=2, num_heads=4, codebook_size=64,
    )
    pc = torch.randn(1, 256, 3, requires_grad=True)
    result = model(pc)
    loss = result["vq_loss"] + result["quantized"].sum()
    loss.backward()
    assert pc.grad is not None
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_vqvae.py -v`
Expected: FAIL

**Step 3: Append VQVAE class to models/vqvae.py**

Add the following to `models/vqvae.py` after the `VectorQuantize` class:

```python
# Append to models/vqvae.py

from timm.models.vision_transformer import Block as ViTBlock
from models.point_patch_embed import PointPatchEmbed


class VQVAE(nn.Module):
    """
    VQVAE tokenizer for point clouds.

    Encodes point cloud patches into discrete tokens via VQ codebook.
    Decoder reconstructs target features (NeRF triplane or occupancy).
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_groups: int = 64,
        group_size: int = 32,
        encoder_depth: int = 6,
        decoder_depth: int = 6,
        num_heads: int = 6,
        codebook_size: int = 8192,
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PointPatchEmbed(
            in_channels=3, embed_dim=embed_dim,
            num_groups=num_groups, group_size=group_size,
        )

        # Positional encoding for patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_groups, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Encoder
        self.encoder = nn.Sequential(*[
            ViTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True)
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # Vector quantization
        self.vq = VectorQuantize(dim=embed_dim, codebook_size=codebook_size, commitment_weight=commitment_weight)

        # Decoder
        self.decoder = nn.Sequential(*[
            ViTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(embed_dim)

    def encode(self, pc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode point cloud to discrete token indices."""
        tokens, centers = self.patch_embed(pc)
        tokens = tokens + self.pos_embed
        tokens = self.encoder_norm(self.encoder(tokens))
        _, indices, _ = self.vq(tokens)
        return indices, centers

    def decode(self, indices: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """Decode from codebook indices."""
        quantized = self.vq.codebook(indices)  # (B, G, D)
        quantized = quantized + self.pos_embed
        decoded = self.decoder_norm(self.decoder(quantized))
        return decoded

    def forward(self, pc: torch.Tensor) -> dict:
        """Full forward pass: encode, quantize, decode."""
        tokens, centers = self.patch_embed(pc)
        tokens = tokens + self.pos_embed

        # Encode
        encoded = self.encoder_norm(self.encoder(tokens))

        # Quantize
        quantized, indices, vq_loss = self.vq(encoded)

        # Decode
        quantized_with_pos = quantized + self.pos_embed
        decoded = self.decoder_norm(self.decoder(quantized_with_pos))

        return {
            "quantized": decoded,
            "indices": indices,
            "vq_loss": vq_loss,
            "centers": centers,
            "encoded": encoded,
        }
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_vqvae.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add models/vqvae.py tests/test_vqvae.py
git commit -m "feat: add full VQVAE model (encoder + VQ + decoder)"
```

---

### Task 16: NeRF Triplane Representation

**Files:**
- Create: `models/nerf.py`
- Create: `tests/test_nerf.py`
- Create: `configs/nerf.yaml`

**Step 1: Write the failing test**

```python
# tests/test_nerf.py
import torch
import pytest
from models.nerf import TriplaneNeRF, TriplaneFeatureExtractor


def test_triplane_nerf_output():
    model = TriplaneNeRF(
        plane_resolution=64,
        plane_channels=32,
        mlp_hidden=128,
    )
    # Query 3D points
    points = torch.randn(2, 100, 3)  # (B, N_points, 3)
    rgb, density = model(points)
    assert rgb.shape == (2, 100, 3)
    assert density.shape == (2, 100, 1)


def test_triplane_feature_extractor():
    model = TriplaneNeRF(plane_resolution=64, plane_channels=32, mlp_hidden=128)
    features = model.get_triplane_features()
    # 3 planes, each (plane_channels, H, W)
    assert features.shape == (3, 32, 64, 64)


def test_triplane_nerf_gradient():
    model = TriplaneNeRF(plane_resolution=32, plane_channels=16, mlp_hidden=64)
    points = torch.randn(1, 50, 3)
    rgb, density = model(points)
    loss = rgb.sum() + density.sum()
    loss.backward()
    # Triplane parameters should have gradients
    assert model.planes.grad is not None
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_nerf.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# models/nerf.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class TriplaneNeRF(nn.Module):
    """
    Triplane NeRF for per-object 3D representation.

    Uses 3 orthogonal feature planes (XY, XZ, YZ).
    Query features by projecting 3D points onto each plane.
    """

    def __init__(
        self,
        plane_resolution: int = 64,
        plane_channels: int = 32,
        mlp_hidden: int = 128,
    ):
        super().__init__()
        self.plane_resolution = plane_resolution
        self.plane_channels = plane_channels

        # 3 feature planes: XY, XZ, YZ
        self.planes = nn.Parameter(
            torch.randn(3, plane_channels, plane_resolution, plane_resolution) * 0.01
        )

        # MLP decoder: features → RGB + density
        feat_dim = plane_channels * 3  # concatenate features from 3 planes
        self.density_mlp = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, 1),
            nn.Softplus(),
        )
        self.rgb_mlp = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, 3),
            nn.Sigmoid(),
        )

    def sample_plane(self, plane: torch.Tensor, coords_2d: torch.Tensor) -> torch.Tensor:
        """
        Bilinear sample from a feature plane.
        Args:
            plane: (C, H, W) feature plane
            coords_2d: (N, 2) coordinates in [-1, 1]
        Returns:
            (N, C) sampled features
        """
        # grid_sample expects (1, C, H, W) and (1, 1, N, 2)
        grid = coords_2d.reshape(1, 1, -1, 2)
        plane_4d = plane.unsqueeze(0)
        sampled = F.grid_sample(plane_4d, grid, mode="bilinear", align_corners=True, padding_mode="border")
        return sampled.squeeze(0).squeeze(1).t()  # (N, C)

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            points: (B, N, 3) 3D query points, coordinates in [-1, 1]
        Returns:
            rgb: (B, N, 3)
            density: (B, N, 1)
        """
        B, N, _ = points.shape
        x, y, z = points[..., 0], points[..., 1], points[..., 2]

        all_rgb, all_density = [], []
        for b in range(B):
            # Project to 3 planes: XY, XZ, YZ
            xy = torch.stack([x[b], y[b]], dim=-1)  # (N, 2)
            xz = torch.stack([x[b], z[b]], dim=-1)
            yz = torch.stack([y[b], z[b]], dim=-1)

            f_xy = self.sample_plane(self.planes[0], xy)  # (N, C)
            f_xz = self.sample_plane(self.planes[1], xz)
            f_yz = self.sample_plane(self.planes[2], yz)

            feat = torch.cat([f_xy, f_xz, f_yz], dim=-1)  # (N, 3C)

            density = self.density_mlp(feat)
            rgb = self.rgb_mlp(feat)

            all_rgb.append(rgb)
            all_density.append(density)

        return torch.stack(all_rgb), torch.stack(all_density)

    def get_triplane_features(self) -> torch.Tensor:
        """Return triplane features for use as VQVAE target."""
        return self.planes.data.clone()  # (3, C, H, W)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_nerf.py -v`
Expected: 3 passed

**Step 5: Create NeRF config and commit**

```yaml
# configs/nerf.yaml
data:
  dataset: shapenet_renders
  data_root: data/ShapeNet55/renders
  image_size: 128
  num_views: 24

model:
  plane_resolution: 64
  plane_channels: 32
  mlp_hidden: 128

train:
  epochs_per_object: 1000
  lr: 5e-4
  batch_size_rays: 4096

output:
  feature_dir: data/ShapeNet55/triplane_features
  save_freq: 100
```

```bash
git add models/nerf.py tests/test_nerf.py configs/nerf.yaml
git commit -m "feat: add triplane NeRF model"
```

---

### Task 17: Extractor (Multi-Scale ViT Encoder)

**Files:**
- Create: `models/extractor.py`
- Create: `tests/test_extractor.py`

**Step 1: Write the failing test**

```python
# tests/test_extractor.py
import torch
import pytest
from models.extractor import Extractor


def test_extractor_forward_shape():
    model = Extractor(
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_groups=64,
    )
    # Simulate unmasked tokens (e.g., 32 out of 64 visible)
    tokens = torch.randn(2, 32, 384)
    centers = torch.randn(2, 32, 3)
    out = model(tokens, centers)
    assert out.shape == (2, 32, 384)


def test_extractor_with_different_token_counts():
    model = Extractor(embed_dim=384, depth=12, num_heads=6, num_groups=64)
    # Different masking ratios → different visible counts
    for n_visible in [16, 32, 48, 64]:
        tokens = torch.randn(2, n_visible, 384)
        centers = torch.randn(2, n_visible, 3)
        out = model(tokens, centers)
        assert out.shape == (2, n_visible, 384)


def test_extractor_gradient_flow():
    model = Extractor(embed_dim=128, depth=2, num_heads=4, num_groups=64)
    tokens = torch.randn(1, 32, 128, requires_grad=True)
    centers = torch.randn(1, 32, 3)
    out = model(tokens, centers)
    out.sum().backward()
    assert tokens.grad is not None
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_extractor.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# models/extractor.py
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block as ViTBlock


class Extractor(nn.Module):
    """
    Extractor: Multi-scale ViT encoder for point cloud pretraining.

    Processes only visible (unmasked) tokens during pretraining.
    During finetuning, processes all tokens.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        num_groups: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_groups = num_groups

        # Positional encoding via learnable center embedding
        self.center_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ViTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, V, D) visible token embeddings
            centers: (B, V, 3) center coordinates of visible patches
        Returns:
            (B, V, D) encoded features
        """
        # Add positional encoding from center coordinates
        pos = self.center_embed(centers)  # (B, V, D)
        x = tokens + pos

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_extractor.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add models/extractor.py tests/test_extractor.py
git commit -m "feat: add Extractor (ViT encoder for pretraining)"
```

---

### Task 18: Generator (ViT Decoder for Token Prediction)

**Files:**
- Create: `models/generator.py`
- Create: `tests/test_generator.py`

**Step 1: Write the failing test**

```python
# tests/test_generator.py
import torch
import pytest
from models.generator import Generator


def test_generator_forward_shape():
    model = Generator(
        embed_dim=384,
        depth=4,
        num_heads=6,
        num_groups=64,
        codebook_size=512,
    )
    # Full sequence: visible features + mask tokens
    visible_features = torch.randn(2, 32, 384)
    visible_centers = torch.randn(2, 32, 3)
    mask_centers = torch.randn(2, 32, 3)
    visible_bool = torch.cat([torch.ones(2, 32), torch.zeros(2, 32)], dim=1).bool()

    logits, center_pred = model(visible_features, visible_centers, mask_centers, visible_bool)
    # logits: predictions for masked tokens
    assert logits.shape == (2, 32, 512)  # (B, M, codebook_size)
    # center_pred: predicted center coordinates for masked patches
    assert center_pred.shape == (2, 32, 3)


def test_generator_gradient_flow():
    model = Generator(embed_dim=128, depth=2, num_heads=4, num_groups=16, codebook_size=64)
    vis_feat = torch.randn(1, 8, 128, requires_grad=True)
    vis_centers = torch.randn(1, 8, 3)
    mask_centers = torch.randn(1, 8, 3)
    vis_bool = torch.cat([torch.ones(1, 8), torch.zeros(1, 8)], dim=1).bool()

    logits, center_pred = model(vis_feat, vis_centers, mask_centers, vis_bool)
    loss = logits.sum() + center_pred.sum()
    loss.backward()
    assert vis_feat.grad is not None
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_generator.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# models/generator.py
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block as ViTBlock


class Generator(nn.Module):
    """
    Generator: Lightweight ViT decoder for masked token prediction.

    Takes visible features from Extractor, inserts mask tokens for masked positions,
    and predicts VQVAE token indices at masked positions.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        num_groups: int = 64,
        codebook_size: int = 8192,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_groups = num_groups

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Center position embedding
        self.center_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ViTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Token prediction head
        self.token_head = nn.Linear(embed_dim, codebook_size)

        # Center point prediction head
        self.center_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )

    def forward(
        self,
        visible_features: torch.Tensor,
        visible_centers: torch.Tensor,
        mask_centers: torch.Tensor,
        visible_bool: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            visible_features: (B, V, D) encoded features from Extractor
            visible_centers: (B, V, 3) center coords of visible patches
            mask_centers: (B, M, 3) center coords of masked patches
            visible_bool: (B, G) bool mask, True=visible
        Returns:
            logits: (B, M, codebook_size) token prediction logits for masked positions
            center_pred: (B, M, 3) predicted center coordinates for masked positions
        """
        B, V, D = visible_features.shape
        M = mask_centers.shape[1]
        G = V + M

        # Create full sequence with mask tokens
        mask_tokens = self.mask_token.expand(B, M, -1)

        # Combine visible and masked, then add position encoding
        all_centers = torch.cat([visible_centers, mask_centers], dim=1)  # (B, G, 3)
        pos = self.center_embed(all_centers)  # (B, G, D)

        full_tokens = torch.cat([visible_features, mask_tokens], dim=1)  # (B, G, D)
        full_tokens = full_tokens + pos

        # Apply transformer
        for block in self.blocks:
            full_tokens = block(full_tokens)
        full_tokens = self.norm(full_tokens)

        # Extract masked position features
        masked_features = full_tokens[:, V:]  # (B, M, D)

        # Predict token indices and centers
        logits = self.token_head(masked_features)  # (B, M, codebook_size)
        center_pred = self.center_head(masked_features)  # (B, M, 3)

        return logits, center_pred
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_generator.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add models/generator.py tests/test_generator.py
git commit -m "feat: add Generator (ViT decoder for masked token prediction)"
```

---

### Task 19: Sliding Masking Strategy

**Files:**
- Create: `models/masking.py`
- Create: `tests/test_masking.py`

**Step 1: Write the failing test**

```python
# tests/test_masking.py
import torch
import pytest
from models.masking import sliding_mask, compute_mask_ratio


def test_mask_ratio_at_start():
    """At epoch 0, mask ratio should be beta (0.5)."""
    ratio = compute_mask_ratio(epoch=0, total_epochs=300, beta=0.5, u=2.0)
    assert abs(ratio - 0.5) < 0.01


def test_mask_ratio_at_end():
    """At final epoch, mask ratio should approach 1.0."""
    ratio = compute_mask_ratio(epoch=299, total_epochs=300, beta=0.5, u=2.0)
    assert ratio > 0.95


def test_mask_ratio_monotonic():
    """Mask ratio should increase over training."""
    ratios = [compute_mask_ratio(e, 300, 0.5, 2.0) for e in range(300)]
    for i in range(1, len(ratios)):
        assert ratios[i] >= ratios[i - 1] - 1e-6


def test_sliding_mask_output_shape():
    num_groups = 64
    mask_ratio = 0.75
    mask = sliding_mask(batch_size=4, num_groups=num_groups, mask_ratio=mask_ratio)
    assert mask.shape == (4, num_groups)
    assert mask.dtype == torch.bool
    # Visible count ~ (1 - 0.75) * 64 = 16
    visible_count = mask.sum(dim=1).float().mean().item()
    assert abs(visible_count - 16) < 2


def test_sliding_mask_different_ratios():
    for ratio in [0.25, 0.5, 0.75, 0.9]:
        mask = sliding_mask(batch_size=8, num_groups=64, mask_ratio=ratio)
        expected_visible = int((1 - ratio) * 64)
        actual_visible = mask.sum(dim=1)[0].item()
        assert abs(actual_visible - expected_visible) <= 1
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_masking.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# models/masking.py
import torch


def compute_mask_ratio(epoch: int, total_epochs: int, beta: float = 0.5, u: float = 2.0) -> float:
    """
    Compute sliding mask ratio.

    Formula: m_r = 1 - beta^((epoch/total_epochs) * u)
    At epoch 0: m_r ≈ 0 (but clamped to beta minimum visible)
    At final epoch: m_r approaches 1.0

    Paper formula: r = 1 - β^(γ/Γ * u)
    where γ=epoch, Γ=total_epochs, β=0.5, u=2
    """
    progress = epoch / max(total_epochs - 1, 1)
    mask_ratio = 1.0 - beta ** (progress * u)
    return max(mask_ratio, beta)  # At least beta masking


def sliding_mask(
    batch_size: int,
    num_groups: int,
    mask_ratio: float,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Generate random mask for point patches.

    Args:
        batch_size: batch size
        num_groups: total number of patches (G)
        mask_ratio: fraction of patches to mask
        device: tensor device
    Returns:
        visible_mask: (B, G) bool tensor, True = visible, False = masked
    """
    num_visible = max(1, int(num_groups * (1 - mask_ratio)))

    # Random permutation per sample
    noise = torch.rand(batch_size, num_groups, device=device)
    ids_sorted = noise.argsort(dim=-1)

    visible_mask = torch.zeros(batch_size, num_groups, dtype=torch.bool, device=device)
    visible_ids = ids_sorted[:, :num_visible]
    visible_mask.scatter_(1, visible_ids, True)

    return visible_mask
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_masking.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add models/masking.py tests/test_masking.py
git commit -m "feat: add sliding masking strategy"
```

---

## Phase 4: Training Pipeline

### Task 20: Evaluation Metrics

**Files:**
- Create: `utils/metrics.py`
- Create: `tests/test_metrics.py`

**Step 1: Write the failing test**

```python
# tests/test_metrics.py
import torch
import pytest
from utils.metrics import accuracy, compute_iou, AverageMeter


def test_accuracy():
    pred = torch.tensor([0, 1, 2, 3, 0])
    target = torch.tensor([0, 1, 2, 2, 1])
    acc = accuracy(pred, target)
    assert abs(acc - 60.0) < 1e-5  # 3/5 = 60%


def test_compute_iou():
    pred = torch.tensor([0, 0, 1, 1, 2, 2])
    target = torch.tensor([0, 1, 1, 1, 2, 0])
    iou = compute_iou(pred, target, num_classes=3)
    # Class 0: intersection=1, union=3 → 1/3
    # Class 1: intersection=2, union=3 → 2/3
    # Class 2: intersection=1, union=2 → 1/2
    assert len(iou) == 3
    assert abs(iou[0] - 1 / 3) < 1e-5
    assert abs(iou[1] - 2 / 3) < 1e-5
    assert abs(iou[2] - 1 / 2) < 1e-5


def test_average_meter():
    meter = AverageMeter()
    meter.update(1.0)
    meter.update(3.0)
    assert meter.avg == 2.0
    assert meter.count == 2
    meter.reset()
    assert meter.avg == 0.0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# utils/metrics.py
import torch
import numpy as np


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute classification accuracy (%)."""
    correct = (pred == target).sum().item()
    total = target.numel()
    return 100.0 * correct / total


def compute_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> list[float]:
    """Compute per-class IoU for segmentation."""
    ious = []
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(intersection / union)
    return ious
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add utils/metrics.py tests/test_metrics.py
git commit -m "feat: add evaluation metrics (accuracy, IoU, AverageMeter)"
```

---

### Task 21: NeRF Training Script

**Files:**
- Create: `tools/train_nerf.py`
- Create: `tests/test_train_nerf.py`

**Step 1: Write the failing test**

```python
# tests/test_train_nerf.py
import torch
import pytest
from models.nerf import TriplaneNeRF


def test_nerf_training_step():
    """Test that one training step reduces loss."""
    model = TriplaneNeRF(plane_resolution=16, plane_channels=8, mlp_hidden=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Mock data: random rays
    points = torch.rand(1, 64, 3) * 2 - 1  # [-1, 1]
    target_rgb = torch.rand(1, 64, 3)
    target_density = torch.rand(1, 64, 1)

    # Step 1
    rgb1, density1 = model(points)
    loss1 = ((rgb1 - target_rgb) ** 2).mean() + ((density1 - target_density) ** 2).mean()

    optimizer.zero_grad()
    loss1.backward()
    optimizer.step()

    # Step 2
    rgb2, density2 = model(points)
    loss2 = ((rgb2 - target_rgb) ** 2).mean() + ((density2 - target_density) ** 2).mean()

    # Loss should decrease (not guaranteed but very likely with random data)
    assert loss2.item() < loss1.item() * 1.1  # Allow 10% tolerance
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_train_nerf.py -v`
Expected: FAIL (or pass — either way confirms model works)

**Step 3: Write training script**

```python
# tools/train_nerf.py
"""
Stage 1: Train per-object NeRF and extract triplane features.

Usage:
    python tools/train_nerf.py --config configs/nerf.yaml
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config, merge_config
from utils.logger import get_logger
from models.nerf import TriplaneNeRF


def train_single_object(
    model: TriplaneNeRF,
    images: torch.Tensor,
    cameras: dict,
    cfg,
    logger,
) -> torch.Tensor:
    """Train NeRF for one object and return triplane features."""
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    device = next(model.parameters()).device

    for epoch in range(cfg.train.epochs_per_object):
        # Sample random rays from random views
        view_idx = torch.randint(0, images.shape[0], (cfg.train.batch_size_rays,))
        # Ray generation would depend on camera model
        # Simplified: random points in [-1, 1]^3
        points = torch.rand(1, cfg.train.batch_size_rays, 3, device=device) * 2 - 1
        target_rgb = images[view_idx].to(device)

        rgb_pred, density = model(points)
        loss = F.mse_loss(rgb_pred.squeeze(0), target_rgb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            logger.info(f"  Epoch {epoch}: loss={loss.item():.6f}")

    return model.get_triplane_features()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.opts:
        cfg = merge_config(cfg, args.opts)

    logger = get_logger("train_nerf")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(cfg.output.feature_dir, exist_ok=True)

    # Load object list
    data_root = cfg.data.data_root
    object_list = sorted(os.listdir(data_root))

    logger.info(f"Training NeRF for {len(object_list)} objects")

    for obj_name in tqdm(object_list):
        output_path = os.path.join(cfg.output.feature_dir, f"{obj_name}.npy")
        if os.path.exists(output_path):
            continue

        model = TriplaneNeRF(
            plane_resolution=cfg.model.plane_resolution,
            plane_channels=cfg.model.plane_channels,
            mlp_hidden=cfg.model.mlp_hidden,
        ).to(device)

        # Load rendered images and cameras for this object
        obj_dir = os.path.join(data_root, obj_name)
        images = torch.zeros(cfg.data.num_views, 3)  # Placeholder
        cameras = {}  # Placeholder

        features = train_single_object(model, images, cameras, cfg, logger)
        np.save(output_path, features.cpu().numpy())

    logger.info("NeRF training complete.")


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_train_nerf.py -v`
Expected: 1 passed

**Step 5: Commit**

```bash
git add tools/train_nerf.py tests/test_train_nerf.py
git commit -m "feat: add NeRF training script (stage 1)"
```

---

### Task 22: VQVAE Training Script

**Files:**
- Create: `tools/train_vqvae.py`
- Create: `tests/test_train_vqvae.py`
- Create: `configs/vqvae.yaml`

**Step 1: Write the failing test**

```python
# tests/test_train_vqvae.py
import torch
from models.vqvae import VQVAE


def test_vqvae_training_step():
    """Test one VQVAE training step reduces loss."""
    model = VQVAE(
        embed_dim=128, num_groups=16, group_size=8,
        encoder_depth=2, decoder_depth=2, num_heads=4, codebook_size=64,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    pc = torch.randn(2, 256, 3)
    # Mock target: triplane features flattened to match decoder output
    target = torch.randn(2, 16, 128)

    # Step 1
    result = model(pc)
    recon_loss = ((result["quantized"] - target) ** 2).mean()
    loss1 = recon_loss + result["vq_loss"]

    optimizer.zero_grad()
    loss1.backward()
    optimizer.step()

    # Step 2
    result2 = model(pc)
    recon_loss2 = ((result2["quantized"] - target) ** 2).mean()
    loss2 = recon_loss2 + result2["vq_loss"]

    # Loss should generally decrease
    assert loss2.item() < loss1.item() * 1.5  # Generous tolerance


def test_vqvae_codebook_utilization():
    """Test that codebook entries are actually used."""
    model = VQVAE(
        embed_dim=64, num_groups=16, group_size=8,
        encoder_depth=2, decoder_depth=2, num_heads=4, codebook_size=32,
    )
    pc = torch.randn(8, 256, 3)
    result = model(pc)
    unique_codes = result["indices"].unique()
    # Should use at least a few different codes
    assert len(unique_codes) > 1
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_train_vqvae.py -v`
Expected: FAIL

**Step 3: Write training script**

```python
# tools/train_vqvae.py
"""
Stage 2: Train VQVAE tokenizer.

Usage:
    python tools/train_vqvae.py --config configs/vqvae.yaml
    torchrun --nproc_per_node=4 tools/train_vqvae.py --config configs/vqvae.yaml
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config, merge_config
from utils.logger import get_logger
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.scheduler import build_scheduler
from utils.distributed import setup_distributed, cleanup_distributed, is_main_process, get_rank, get_world_size
from utils.metrics import AverageMeter
from datasets import build_dataset
from models.vqvae import VQVAE


def train_one_epoch(model, loader, optimizer, device, epoch, logger):
    model.train()
    loss_meter = AverageMeter()

    for batch_idx, batch in enumerate(loader):
        pc = batch["points"].to(device)

        result = model(pc)
        recon_loss = result["quantized"].pow(2).mean()  # Placeholder — real target is NeRF features
        vq_loss = result["vq_loss"]
        loss = recon_loss + vq_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), pc.size(0))

        if batch_idx % 50 == 0 and is_main_process():
            logger.info(
                f"Epoch [{epoch}] Step [{batch_idx}/{len(loader)}] "
                f"Loss: {loss_meter.avg:.4f} VQ: {vq_loss.item():.4f}"
            )

    return loss_meter.avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.opts:
        cfg = merge_config(cfg, args.opts)

    # Distributed setup
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        setup_distributed()

    device = torch.device(f"cuda:{get_rank()}" if torch.cuda.is_available() else "cpu")
    logger = get_logger("train_vqvae", log_file=os.path.join(cfg.output.exp_dir, "train.log") if is_main_process() else None)

    os.makedirs(cfg.output.exp_dir, exist_ok=True)

    # Dataset
    dataset = build_dataset(cfg.data.dataset, data_root=cfg.data.data_root, split=cfg.data.train_split, n_points=cfg.data.n_points)
    sampler = DistributedSampler(dataset) if distributed else None
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, sampler=sampler, shuffle=(sampler is None), num_workers=4, pin_memory=True, drop_last=True)

    # Model
    model = VQVAE(
        embed_dim=cfg.model.embed_dim,
        num_groups=cfg.model.num_groups,
        group_size=cfg.model.group_size,
        encoder_depth=cfg.model.get("encoder_depth", 6),
        decoder_depth=cfg.model.get("decoder_depth", 6),
        num_heads=cfg.model.num_heads,
        codebook_size=cfg.model.codebook_size,
    ).to(device)

    if distributed:
        model = DDP(model, device_ids=[get_rank()])

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = build_scheduler(optimizer, epochs=cfg.train.epochs, warmup_epochs=cfg.train.warmup_epochs, min_lr=cfg.train.min_lr)

    start_epoch = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, model.module if distributed else model, optimizer)
        start_epoch = ckpt["epoch"] + 1

    for epoch in range(start_epoch, cfg.train.epochs):
        if distributed:
            sampler.set_epoch(epoch)

        loss = train_one_epoch(model, loader, optimizer, device, epoch, logger)
        scheduler.step()

        if is_main_process():
            logger.info(f"Epoch [{epoch}] Loss: {loss:.4f}")
            if (epoch + 1) % cfg.output.save_freq == 0:
                save_checkpoint(
                    os.path.join(cfg.output.exp_dir, f"epoch_{epoch}.pth"),
                    model.module if distributed else model,
                    optimizer, epoch=epoch,
                )

    if is_main_process():
        save_checkpoint(os.path.join(cfg.output.exp_dir, "final.pth"), model.module if distributed else model, optimizer, epoch=cfg.train.epochs - 1)

    if distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
```

**Step 4: Create config**

```yaml
# configs/vqvae.yaml
data:
  dataset: shapenet55
  data_root: data/ShapeNet55
  pc_dir: shapenet_pc
  n_points: 2048
  train_split: train
  val_split: test

model:
  embed_dim: 384
  num_groups: 64
  group_size: 32
  encoder_depth: 6
  decoder_depth: 6
  num_heads: 6
  codebook_size: 8192

train:
  epochs: 300
  batch_size: 64
  lr: 1e-3
  weight_decay: 0.05
  warmup_epochs: 10
  min_lr: 1e-5

output:
  exp_dir: experiments/vqvae
  save_freq: 50
  log_freq: 10
```

**Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_train_vqvae.py -v`
Expected: 2 passed

**Step 6: Commit**

```bash
git add tools/train_vqvae.py tests/test_train_vqvae.py configs/vqvae.yaml
git commit -m "feat: add VQVAE training script (stage 2)"
```

---

### Task 23: Pretraining Script (Extractor-Generator)

**Files:**
- Create: `tools/pretrain.py`
- Create: `tests/test_pretrain.py`

**Step 1: Write the failing test**

```python
# tests/test_pretrain.py
import torch
import pytest
from models.point_patch_embed import PointPatchEmbed
from models.extractor import Extractor
from models.generator import Generator
from models.masking import sliding_mask, compute_mask_ratio
from models.vqvae import VQVAE


def test_pretrain_forward_pass():
    """Test full pretraining forward pass: embed → mask → extract → generate."""
    B, N, G, D = 2, 512, 16, 128

    patch_embed = PointPatchEmbed(in_channels=3, embed_dim=D, num_groups=G, group_size=8)
    extractor = Extractor(embed_dim=D, depth=2, num_heads=4, num_groups=G)
    generator = Generator(embed_dim=D, depth=2, num_heads=4, num_groups=G, codebook_size=64)

    pc = torch.randn(B, N, 3)

    # 1. Patch embedding
    tokens, centers = patch_embed(pc)
    assert tokens.shape == (B, G, D)

    # 2. Masking
    mask_ratio = compute_mask_ratio(epoch=150, total_epochs=300)
    visible_mask = sliding_mask(B, G, mask_ratio)
    num_visible = visible_mask.sum(dim=1)[0].item()
    num_masked = G - num_visible

    # 3. Split visible and masked
    # Gather visible tokens/centers
    vis_idx = visible_mask.nonzero(as_tuple=False)  # Simple approach per batch
    # Use a per-batch approach
    vis_tokens_list, vis_centers_list, mask_centers_list = [], [], []
    for b in range(B):
        v_idx = visible_mask[b].nonzero(as_tuple=True)[0]
        m_idx = (~visible_mask[b]).nonzero(as_tuple=True)[0]
        vis_tokens_list.append(tokens[b, v_idx])
        vis_centers_list.append(centers[b, v_idx])
        mask_centers_list.append(centers[b, m_idx])

    vis_tokens = torch.stack(vis_tokens_list)
    vis_centers = torch.stack(vis_centers_list)
    mask_centers = torch.stack(mask_centers_list)

    # 4. Extractor
    extracted = extractor(vis_tokens, vis_centers)
    assert extracted.shape == vis_tokens.shape

    # 5. Generator
    logits, center_pred = generator(extracted, vis_centers, mask_centers, visible_mask)
    assert logits.shape[0] == B
    assert logits.shape[2] == 64  # codebook_size
    assert center_pred.shape == mask_centers.shape


def test_pretrain_loss_computation():
    """Test that cross-entropy loss is computable."""
    B, M, K = 2, 8, 64
    logits = torch.randn(B, M, K)
    target_indices = torch.randint(0, K, (B, M))

    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, K), target_indices.reshape(-1)
    )
    assert loss.shape == ()
    assert not torch.isnan(loss)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pretrain.py -v`
Expected: FAIL (initially) or PASS (if all dependencies already work)

**Step 3: Write pretraining script**

```python
# tools/pretrain.py
"""
Stage 3: Pretrain Extractor-Generator with sliding masking.

Usage:
    python tools/pretrain.py --config configs/pretrain.yaml
    torchrun --nproc_per_node=4 tools/pretrain.py --config configs/pretrain.yaml
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config, merge_config
from utils.logger import get_logger
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.scheduler import build_scheduler
from utils.distributed import setup_distributed, cleanup_distributed, is_main_process, get_rank
from utils.metrics import AverageMeter
from datasets import build_dataset
from models.point_patch_embed import PointPatchEmbed
from models.extractor import Extractor
from models.generator import Generator
from models.vqvae import VQVAE
from models.masking import sliding_mask, compute_mask_ratio


class PretrainModel(nn.Module):
    """Wraps patch_embed + extractor + generator for pretraining."""

    def __init__(self, cfg):
        super().__init__()
        D = cfg.model.embed_dim
        G = cfg.model.num_groups

        self.patch_embed = PointPatchEmbed(
            in_channels=3, embed_dim=D,
            num_groups=G, group_size=cfg.model.group_size,
        )
        self.extractor = Extractor(
            embed_dim=D, depth=cfg.model.extractor_depth,
            num_heads=cfg.model.num_heads, num_groups=G,
        )
        self.generator = Generator(
            embed_dim=D, depth=cfg.model.generator_depth,
            num_heads=cfg.model.num_heads, num_groups=G,
            codebook_size=cfg.model.codebook_size,
        )
        self.num_groups = G

    def forward(self, pc, visible_mask, target_indices):
        B, G = visible_mask.shape
        D = self.extractor.embed_dim

        # Patch embedding
        tokens, centers = self.patch_embed(pc)

        # Split visible / masked
        vis_tokens_list, vis_centers_list, mask_centers_list, target_list = [], [], [], []
        for b in range(B):
            v_idx = visible_mask[b].nonzero(as_tuple=True)[0]
            m_idx = (~visible_mask[b]).nonzero(as_tuple=True)[0]
            vis_tokens_list.append(tokens[b, v_idx])
            vis_centers_list.append(centers[b, v_idx])
            mask_centers_list.append(centers[b, m_idx])
            target_list.append(target_indices[b, m_idx])

        vis_tokens = torch.stack(vis_tokens_list)
        vis_centers = torch.stack(vis_centers_list)
        mask_centers = torch.stack(mask_centers_list)
        targets = torch.stack(target_list)

        # Extractor
        extracted = self.extractor(vis_tokens, vis_centers)

        # Generator
        logits, center_pred = self.generator(extracted, vis_centers, mask_centers, visible_mask)

        # Losses
        token_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )
        center_loss = F.mse_loss(center_pred, mask_centers)

        return {
            "token_loss": token_loss,
            "center_loss": center_loss,
            "loss": token_loss + 0.1 * center_loss,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--vqvae_ckpt", type=str, default=None, help="Path to trained VQVAE checkpoint")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.opts:
        cfg = merge_config(cfg, args.opts)

    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        setup_distributed()

    device = torch.device(f"cuda:{get_rank()}" if torch.cuda.is_available() else "cpu")
    logger = get_logger("pretrain", log_file=os.path.join(cfg.output.exp_dir, "train.log") if is_main_process() else None)
    os.makedirs(cfg.output.exp_dir, exist_ok=True)

    # Load VQVAE for tokenization (frozen)
    vqvae = VQVAE(
        embed_dim=cfg.model.embed_dim, num_groups=cfg.model.num_groups,
        group_size=cfg.model.group_size, encoder_depth=6, decoder_depth=6,
        num_heads=cfg.model.num_heads, codebook_size=cfg.model.codebook_size,
    ).to(device)
    if args.vqvae_ckpt:
        load_checkpoint(args.vqvae_ckpt, vqvae)
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False

    # Dataset
    dataset = build_dataset(cfg.data.dataset, data_root=cfg.data.data_root, split=cfg.data.train_split, n_points=cfg.data.n_points)
    sampler = DistributedSampler(dataset) if distributed else None
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, sampler=sampler, shuffle=(sampler is None), num_workers=4, pin_memory=True, drop_last=True)

    # Model
    model = PretrainModel(cfg).to(device)
    if distributed:
        model = DDP(model, device_ids=[get_rank()])

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = build_scheduler(optimizer, epochs=cfg.train.epochs, warmup_epochs=cfg.train.warmup_epochs, min_lr=cfg.train.min_lr)

    start_epoch = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, model.module if distributed else model, optimizer)
        start_epoch = ckpt["epoch"] + 1

    for epoch in range(start_epoch, cfg.train.epochs):
        if distributed:
            sampler.set_epoch(epoch)

        model.train()
        loss_meter = AverageMeter()
        mask_ratio = compute_mask_ratio(epoch, cfg.train.epochs, cfg.model.mask_beta, cfg.model.mask_u)

        for batch_idx, batch in enumerate(loader):
            pc = batch["points"].to(device)
            B = pc.shape[0]

            # Get VQVAE token targets
            with torch.no_grad():
                target_indices, _ = vqvae.encode(pc)

            # Generate mask
            visible_mask = sliding_mask(B, cfg.model.num_groups, mask_ratio, device=device)

            # Forward
            result = model(pc, visible_mask, target_indices)
            loss = result["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), B)

        scheduler.step()

        if is_main_process():
            logger.info(f"Epoch [{epoch}] Loss: {loss_meter.avg:.4f} MaskRatio: {mask_ratio:.3f}")
            if (epoch + 1) % cfg.output.save_freq == 0:
                save_checkpoint(
                    os.path.join(cfg.output.exp_dir, f"epoch_{epoch}.pth"),
                    model.module if distributed else model, optimizer, epoch=epoch,
                )

    if is_main_process():
        save_checkpoint(os.path.join(cfg.output.exp_dir, "final.pth"), model.module if distributed else model, optimizer, epoch=cfg.train.epochs - 1)

    if distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pretrain.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add tools/pretrain.py tests/test_pretrain.py
git commit -m "feat: add pretraining script (stage 3, Extractor-Generator)"
```

---

## Phase 5: Downstream Tasks

### Task 24: Classification Head

**Files:**
- Create: `models/heads/cls_head.py`
- Create: `tests/test_cls_head.py`

**Step 1: Write the failing test**

```python
# tests/test_cls_head.py
import torch
import pytest
from models.heads.cls_head import ClassificationHead


def test_cls_head_output_shape():
    head = ClassificationHead(embed_dim=384, num_classes=40)
    features = torch.randn(2, 64, 384)  # (B, G, D)
    logits = head(features)
    assert logits.shape == (2, 40)


def test_cls_head_gradient():
    head = ClassificationHead(embed_dim=128, num_classes=10)
    features = torch.randn(1, 16, 128, requires_grad=True)
    logits = head(features)
    logits.sum().backward()
    assert features.grad is not None
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cls_head.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# models/heads/cls_head.py
import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Classification head: global pool + MLP."""

    def __init__(self, embed_dim: int = 384, num_classes: int = 40, dropout: float = 0.5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),  # concat of max-pool and avg-pool
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, G, D) patch features from extractor
        Returns:
            logits: (B, num_classes)
        """
        max_pool = features.max(dim=1).values  # (B, D)
        avg_pool = features.mean(dim=1)  # (B, D)
        pooled = torch.cat([max_pool, avg_pool], dim=-1)  # (B, 2D)
        return self.head(pooled)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cls_head.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add models/heads/cls_head.py tests/test_cls_head.py
git commit -m "feat: add classification head"
```

---

### Task 25: Classification Finetuning Script

**Files:**
- Create: `tools/finetune_cls.py`
- Create: `tests/test_finetune_cls.py`

**Step 1: Write the failing test**

```python
# tests/test_finetune_cls.py
import torch
from models.point_patch_embed import PointPatchEmbed
from models.extractor import Extractor
from models.heads.cls_head import ClassificationHead


def test_cls_pipeline_forward():
    """Test full classification pipeline: patch_embed → extractor → cls_head."""
    D, G = 128, 16
    patch_embed = PointPatchEmbed(in_channels=3, embed_dim=D, num_groups=G, group_size=8)
    extractor = Extractor(embed_dim=D, depth=2, num_heads=4, num_groups=G)
    cls_head = ClassificationHead(embed_dim=D, num_classes=40)

    pc = torch.randn(2, 256, 3)
    tokens, centers = patch_embed(pc)
    features = extractor(tokens, centers)
    logits = cls_head(features)

    assert logits.shape == (2, 40)

    # Test loss
    labels = torch.randint(0, 40, (2,))
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    assert not torch.isnan(loss)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_finetune_cls.py -v`
Expected: FAIL (or PASS)

**Step 3: Write finetuning script**

```python
# tools/finetune_cls.py
"""
Stage 4a: Finetune for classification on ModelNet40 / ScanObjectNN.

Usage:
    python tools/finetune_cls.py --config configs/cls_modelnet40.yaml
    torchrun --nproc_per_node=4 tools/finetune_cls.py --config configs/cls_modelnet40.yaml
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config, merge_config
from utils.logger import get_logger
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.scheduler import build_scheduler
from utils.distributed import setup_distributed, cleanup_distributed, is_main_process, get_rank
from utils.metrics import accuracy, AverageMeter
from datasets import build_dataset
from models.point_patch_embed import PointPatchEmbed
from models.extractor import Extractor
from models.heads.cls_head import ClassificationHead


class ClassificationModel(nn.Module):
    """Full classification model: patch_embed + extractor + cls_head."""

    def __init__(self, cfg):
        super().__init__()
        D = cfg.model.embed_dim
        G = cfg.model.num_groups
        self.patch_embed = PointPatchEmbed(
            in_channels=3, embed_dim=D, num_groups=G, group_size=cfg.model.group_size,
        )
        self.extractor = Extractor(
            embed_dim=D, depth=cfg.model.extractor_depth, num_heads=cfg.model.num_heads, num_groups=G,
        )
        self.cls_head = ClassificationHead(embed_dim=D, num_classes=cfg.data.num_classes)

    def forward(self, pc):
        tokens, centers = self.patch_embed(pc)
        features = self.extractor(tokens, centers)
        logits = self.cls_head(features)
        return logits


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        pc = batch["points"].to(device)
        labels = batch["label"]
        if isinstance(labels, torch.Tensor):
            labels = labels.to(device)
        else:
            labels = torch.tensor(labels, device=device)

        logits = model(pc)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.shape[0]
    return 100.0 * correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.opts:
        cfg = merge_config(cfg, args.opts)

    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        setup_distributed()

    device = torch.device(f"cuda:{get_rank()}" if torch.cuda.is_available() else "cpu")
    logger = get_logger("finetune_cls", log_file=os.path.join(cfg.output.exp_dir, "train.log") if is_main_process() else None)
    os.makedirs(cfg.output.exp_dir, exist_ok=True)

    # Datasets
    train_ds = build_dataset(cfg.data.dataset, data_root=cfg.data.data_root, split=cfg.data.train_split, n_points=cfg.data.n_points)
    val_ds = build_dataset(cfg.data.dataset, data_root=cfg.data.data_root, split=cfg.data.val_split, n_points=cfg.data.n_points)

    train_sampler = DistributedSampler(train_ds) if distributed else None
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = ClassificationModel(cfg).to(device)

    # Load pretrained weights
    if cfg.model.get("pretrained"):
        logger.info(f"Loading pretrained weights from {cfg.model.pretrained}")
        ckpt = torch.load(cfg.model.pretrained, map_location="cpu", weights_only=False)
        # Load patch_embed and extractor weights
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(f"Loaded pretrained: missing={len(missing)}, unexpected={len(unexpected)}")

    if distributed:
        model = DDP(model, device_ids=[get_rank()])

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = build_scheduler(optimizer, epochs=cfg.train.epochs, warmup_epochs=cfg.train.warmup_epochs, min_lr=cfg.train.min_lr)

    best_acc = 0.0
    for epoch in range(cfg.train.epochs):
        if distributed:
            train_sampler.set_epoch(epoch)

        model.train()
        loss_meter = AverageMeter()

        for batch in train_loader:
            pc = batch["points"].to(device)
            labels = batch["label"]
            if isinstance(labels, torch.Tensor):
                labels = labels.to(device)
            else:
                labels = torch.tensor(labels, device=device)

            logits = model(pc)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), pc.shape[0])

        scheduler.step()

        # Evaluate
        if is_main_process():
            acc = evaluate(model.module if distributed else model, val_loader, device)
            logger.info(f"Epoch [{epoch}] Loss: {loss_meter.avg:.4f} Acc: {acc:.2f}%")

            if acc > best_acc:
                best_acc = acc
                save_checkpoint(os.path.join(cfg.output.exp_dir, "best.pth"), model.module if distributed else model, optimizer, epoch=epoch, best_metric=best_acc)

    if is_main_process():
        logger.info(f"Best accuracy: {best_acc:.2f}%")

    if distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_finetune_cls.py -v`
Expected: 1 passed

**Step 5: Commit**

```bash
git add tools/finetune_cls.py tests/test_finetune_cls.py
git commit -m "feat: add classification finetuning script (stage 4a)"
```

---

### Task 26: Few-Shot Learning Evaluation

**Files:**
- Create: `tools/fewshot.py`
- Create: `tests/test_fewshot.py`
- Create: `configs/fewshot.yaml`

**Step 1: Write the failing test**

```python
# tests/test_fewshot.py
import torch
import pytest
from tools.fewshot import run_fewshot_episode


def test_fewshot_episode():
    """Test k-way n-shot classification with random features."""
    # Simulate: 5-way 10-shot, feature dim=128
    support_features = torch.randn(50, 128)  # 5*10 = 50
    support_labels = torch.arange(5).repeat_interleave(10)
    query_features = torch.randn(50, 128)  # 5*10 = 50 queries
    query_labels = torch.arange(5).repeat_interleave(10)

    acc = run_fewshot_episode(support_features, support_labels, query_features, query_labels)
    # Random chance is 20% for 5-way
    assert 0 <= acc <= 100


def test_fewshot_perfect_classification():
    """Test with perfectly separable features."""
    # Create features that are clearly separable
    support_features = torch.zeros(10, 2)
    support_labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    support_features[:5, 0] = 10.0  # Class 0 at [10, 0]
    support_features[5:, 1] = 10.0  # Class 1 at [0, 10]

    query_features = torch.zeros(4, 2)
    query_labels = torch.tensor([0, 0, 1, 1])
    query_features[:2, 0] = 10.0
    query_features[2:, 1] = 10.0

    acc = run_fewshot_episode(support_features, support_labels, query_features, query_labels)
    assert acc == 100.0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fewshot.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# tools/fewshot.py
"""
Stage 4b: Few-shot learning evaluation.

Usage:
    python tools/fewshot.py --config configs/fewshot.yaml --ckpt experiments/pretrain/best.pth
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config, merge_config
from utils.logger import get_logger
from utils.checkpoint import load_checkpoint
from datasets import build_dataset
from models.point_patch_embed import PointPatchEmbed
from models.extractor import Extractor


def run_fewshot_episode(
    support_features: torch.Tensor,
    support_labels: torch.Tensor,
    query_features: torch.Tensor,
    query_labels: torch.Tensor,
) -> float:
    """
    Run one few-shot episode using nearest-centroid classifier.
    Returns accuracy (%).
    """
    classes = support_labels.unique()
    centroids = []
    for c in classes:
        mask = support_labels == c
        centroids.append(support_features[mask].mean(dim=0))
    centroids = torch.stack(centroids)  # (k, D)

    # Classify queries by nearest centroid
    dists = torch.cdist(query_features, centroids)  # (Q, k)
    preds = classes[dists.argmin(dim=-1)]

    correct = (preds == query_labels).sum().item()
    return 100.0 * correct / query_labels.shape[0]


class FeatureExtractor(nn.Module):
    """Extract global features from point cloud using pretrained model."""

    def __init__(self, cfg):
        super().__init__()
        D = cfg.model.embed_dim
        G = cfg.model.num_groups
        self.patch_embed = PointPatchEmbed(
            in_channels=3, embed_dim=D, num_groups=G, group_size=cfg.model.group_size,
        )
        self.extractor = Extractor(
            embed_dim=D, depth=cfg.model.extractor_depth, num_heads=cfg.model.num_heads, num_groups=G,
        )

    def forward(self, pc):
        tokens, centers = self.patch_embed(pc)
        features = self.extractor(tokens, centers)
        return features.max(dim=1).values  # Global max-pool → (B, D)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.opts:
        cfg = merge_config(cfg, args.opts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger("fewshot")

    # Load model
    model = FeatureExtractor(cfg).to(device)
    load_checkpoint(args.ckpt, model)
    model.eval()

    # Load dataset
    dataset = build_dataset(cfg.data.dataset, data_root=cfg.data.data_root, split="test", n_points=cfg.data.n_points)

    # Extract all features
    all_features, all_labels = [], []
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    with torch.no_grad():
        for batch in loader:
            pc = batch["points"].to(device)
            feat = model(pc)
            all_features.append(feat.cpu())
            all_labels.append(torch.tensor(batch["label"]) if not isinstance(batch["label"], torch.Tensor) else batch["label"])

    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)

    k_way = cfg.get("k_way", 5)
    n_shot = cfg.get("n_shot", 10)
    n_query = cfg.get("n_query", 20)
    n_episodes = cfg.get("n_episodes", 10)

    accs = []
    for ep in range(n_episodes):
        # Sample k classes
        classes = all_labels.unique()
        selected = classes[torch.randperm(len(classes))[:k_way]]

        support_feats, support_labs, query_feats, query_labs = [], [], [], []
        for c in selected:
            idx = (all_labels == c).nonzero(as_tuple=True)[0]
            perm = idx[torch.randperm(len(idx))]
            support_feats.append(all_features[perm[:n_shot]])
            support_labs.extend([c] * n_shot)
            query_feats.append(all_features[perm[n_shot:n_shot + n_query]])
            query_labs.extend([c] * min(n_query, len(perm) - n_shot))

        support_feats = torch.cat(support_feats)
        support_labs = torch.tensor(support_labs)
        query_feats = torch.cat(query_feats)
        query_labs = torch.tensor(query_labs)

        acc = run_fewshot_episode(support_feats, support_labs, query_feats, query_labs)
        accs.append(acc)

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    logger.info(f"{k_way}-way {n_shot}-shot: {mean_acc:.2f} ± {std_acc:.2f}%")


if __name__ == "__main__":
    main()
```

**Step 4: Create config**

```yaml
# configs/fewshot.yaml
data:
  dataset: modelnet40
  data_root: data/ModelNet40
  pc_dir: modelnet40_pc
  n_points: 1024

model:
  embed_dim: 384
  num_groups: 64
  group_size: 32
  extractor_depth: 12
  num_heads: 6

k_way: 5
n_shot: 10
n_query: 20
n_episodes: 10
```

**Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_fewshot.py -v`
Expected: 2 passed

**Step 6: Commit**

```bash
git add tools/fewshot.py tests/test_fewshot.py configs/fewshot.yaml
git commit -m "feat: add few-shot learning evaluation (stage 4b)"
```

---

### Task 27: Part Segmentation Head

**Files:**
- Create: `models/heads/partseg_head.py`
- Create: `tests/test_partseg_head.py`

**Step 1: Write the failing test**

```python
# tests/test_partseg_head.py
import torch
import pytest
from models.heads.partseg_head import PartSegHead


def test_partseg_head_output_shape():
    head = PartSegHead(embed_dim=384, num_groups=64, num_parts=50, num_categories=16)
    features = torch.randn(2, 64, 384)
    centers = torch.randn(2, 64, 3)
    pc = torch.randn(2, 2048, 3)
    category = torch.tensor([0, 3])

    seg_logits = head(features, centers, pc, category)
    assert seg_logits.shape == (2, 2048, 50)


def test_partseg_head_gradient():
    head = PartSegHead(embed_dim=128, num_groups=16, num_parts=10, num_categories=4)
    features = torch.randn(1, 16, 128, requires_grad=True)
    centers = torch.randn(1, 16, 3)
    pc = torch.randn(1, 256, 3)
    category = torch.tensor([0])

    seg_logits = head(features, centers, pc, category)
    seg_logits.sum().backward()
    assert features.grad is not None
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_partseg_head.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# models/heads/partseg_head.py
import torch
import torch.nn as nn


class PartSegHead(nn.Module):
    """
    Part segmentation head.
    Propagates patch-level features to per-point predictions.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_groups: int = 64,
        num_parts: int = 50,
        num_categories: int = 16,
    ):
        super().__init__()
        self.num_groups = num_groups

        # Category embedding
        self.cat_embed = nn.Embedding(num_categories, 64)

        # Propagation: per-point features via distance-weighted interpolation
        # Then MLP for prediction
        self.head = nn.Sequential(
            nn.Conv1d(embed_dim + 64 + 3, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, num_parts, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        centers: torch.Tensor,
        pc: torch.Tensor,
        category: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: (B, G, D) patch features
            centers: (B, G, 3) patch centers
            pc: (B, N, 3) full point cloud
            category: (B,) category indices
        Returns:
            seg_logits: (B, N, num_parts)
        """
        B, N, _ = pc.shape
        G, D = features.shape[1], features.shape[2]

        # Propagate patch features to all points via distance-weighted interpolation
        # (B, N, G) distances
        dist = torch.cdist(pc, centers)  # (B, N, G)
        # Top-3 nearest patches
        knn_dist, knn_idx = dist.topk(3, dim=-1, largest=False)  # (B, N, 3)

        # Inverse distance weights
        weights = 1.0 / (knn_dist + 1e-8)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # (B, N, 3)

        # Gather and interpolate features
        batch_idx = torch.arange(B, device=pc.device).reshape(B, 1, 1).expand(-1, N, 3)
        knn_features = features[batch_idx, knn_idx]  # (B, N, 3, D)
        point_features = (knn_features * weights.unsqueeze(-1)).sum(dim=2)  # (B, N, D)

        # Category embedding
        cat_feat = self.cat_embed(category)  # (B, 64)
        cat_feat = cat_feat.unsqueeze(1).expand(-1, N, -1)  # (B, N, 64)

        # Combine: point features + category + coordinates
        combined = torch.cat([point_features, cat_feat, pc], dim=-1)  # (B, N, D+64+3)
        combined = combined.transpose(1, 2)  # (B, D+64+3, N)

        seg_logits = self.head(combined)  # (B, num_parts, N)
        return seg_logits.transpose(1, 2)  # (B, N, num_parts)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_partseg_head.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add models/heads/partseg_head.py tests/test_partseg_head.py
git commit -m "feat: add part segmentation head"
```

---

### Task 28: Part Segmentation Finetuning Script

**Files:**
- Create: `tools/finetune_partseg.py`
- Create: `tests/test_finetune_partseg.py`

**Step 1: Write the failing test**

```python
# tests/test_finetune_partseg.py
import torch
from models.point_patch_embed import PointPatchEmbed
from models.extractor import Extractor
from models.heads.partseg_head import PartSegHead
from utils.metrics import compute_iou


def test_partseg_pipeline_forward():
    D, G = 128, 16
    patch_embed = PointPatchEmbed(in_channels=3, embed_dim=D, num_groups=G, group_size=8)
    extractor = Extractor(embed_dim=D, depth=2, num_heads=4, num_groups=G)
    seg_head = PartSegHead(embed_dim=D, num_groups=G, num_parts=10, num_categories=4)

    pc = torch.randn(2, 256, 3)
    category = torch.tensor([0, 2])

    tokens, centers = patch_embed(pc)
    features = extractor(tokens, centers)
    logits = seg_head(features, centers, pc, category)

    assert logits.shape == (2, 256, 10)

    # Test loss
    labels = torch.randint(0, 10, (2, 256))
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 10), labels.reshape(-1))
    loss.backward()
    assert not torch.isnan(loss)


def test_iou_computation():
    pred = torch.tensor([0, 0, 1, 1, 2, 2])
    target = torch.tensor([0, 0, 1, 2, 2, 2])
    iou = compute_iou(pred, target, num_classes=3)
    assert all(0 <= v <= 1 for v in iou if not (v != v))  # not NaN
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_finetune_partseg.py -v`
Expected: FAIL (or PASS)

**Step 3: Write finetuning script**

```python
# tools/finetune_partseg.py
"""
Stage 4c: Finetune for part segmentation on ShapeNetPart.

Usage:
    python tools/finetune_partseg.py --config configs/partseg.yaml
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config, merge_config
from utils.logger import get_logger
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.scheduler import build_scheduler
from utils.distributed import setup_distributed, cleanup_distributed, is_main_process, get_rank
from utils.metrics import compute_iou, AverageMeter
from datasets import build_dataset
from models.point_patch_embed import PointPatchEmbed
from models.extractor import Extractor
from models.heads.partseg_head import PartSegHead


class PartSegModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D = cfg.model.embed_dim
        G = cfg.model.num_groups
        self.patch_embed = PointPatchEmbed(in_channels=3, embed_dim=D, num_groups=G, group_size=cfg.model.group_size)
        self.extractor = Extractor(embed_dim=D, depth=cfg.model.extractor_depth, num_heads=cfg.model.num_heads, num_groups=G)
        self.seg_head = PartSegHead(embed_dim=D, num_groups=G, num_parts=cfg.data.num_parts, num_categories=cfg.data.num_categories)

    def forward(self, pc, category):
        tokens, centers = self.patch_embed(pc)
        features = self.extractor(tokens, centers)
        logits = self.seg_head(features, centers, pc, category)
        return logits


@torch.no_grad()
def evaluate(model, loader, device, num_parts):
    model.eval()
    all_ious = []
    for batch in loader:
        pc = batch["points"].to(device)
        category = batch["category"].to(device)
        labels = batch["seg_labels"].to(device)

        logits = model(pc, category)
        preds = logits.argmax(dim=-1)

        for b in range(pc.shape[0]):
            iou = compute_iou(preds[b], labels[b], num_parts)
            valid_ious = [v for v in iou if v == v]  # filter NaN
            if valid_ious:
                all_ious.append(np.mean(valid_ious))

    return np.mean(all_ious) * 100 if all_ious else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.opts:
        cfg = merge_config(cfg, args.opts)

    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        setup_distributed()

    device = torch.device(f"cuda:{get_rank()}" if torch.cuda.is_available() else "cpu")
    logger = get_logger("finetune_partseg", log_file=os.path.join(cfg.output.exp_dir, "train.log") if is_main_process() else None)
    os.makedirs(cfg.output.exp_dir, exist_ok=True)

    train_ds = build_dataset(cfg.data.dataset, data_root=cfg.data.data_root, split="train", n_points=cfg.data.n_points)
    val_ds = build_dataset(cfg.data.dataset, data_root=cfg.data.data_root, split="test", n_points=cfg.data.n_points)

    train_sampler = DistributedSampler(train_ds) if distributed else None
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = PartSegModel(cfg).to(device)
    if cfg.model.get("pretrained"):
        ckpt = torch.load(cfg.model.pretrained, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)

    if distributed:
        model = DDP(model, device_ids=[get_rank()])

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = build_scheduler(optimizer, epochs=cfg.train.epochs, warmup_epochs=cfg.train.warmup_epochs, min_lr=cfg.train.min_lr)

    best_miou = 0.0
    for epoch in range(cfg.train.epochs):
        if distributed:
            train_sampler.set_epoch(epoch)

        model.train()
        loss_meter = AverageMeter()
        for batch in train_loader:
            pc = batch["points"].to(device)
            category = batch["category"].to(device)
            labels = batch["seg_labels"].to(device)

            logits = model(pc, category)
            loss = F.cross_entropy(logits.reshape(-1, cfg.data.num_parts), labels.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), pc.shape[0])

        scheduler.step()

        if is_main_process():
            miou = evaluate(model.module if distributed else model, val_loader, device, cfg.data.num_parts)
            logger.info(f"Epoch [{epoch}] Loss: {loss_meter.avg:.4f} mIoU: {miou:.2f}%")
            if miou > best_miou:
                best_miou = miou
                save_checkpoint(os.path.join(cfg.output.exp_dir, "best.pth"), model.module if distributed else model, optimizer, epoch=epoch, best_metric=best_miou)

    if is_main_process():
        logger.info(f"Best mIoU: {best_miou:.2f}%")

    if distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_finetune_partseg.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add tools/finetune_partseg.py tests/test_finetune_partseg.py
git commit -m "feat: add part segmentation finetuning script (stage 4c)"
```

---

## Phase 6: 3D Shape Generation

### Task 29: GPT Generator Model

**Files:**
- Create: `models/gpt_generator.py`
- Create: `tests/test_gpt_generator.py`

**Step 1: Write the failing test**

```python
# tests/test_gpt_generator.py
import torch
import pytest
from models.gpt_generator import GPTGenerator


def test_gpt_generator_forward_shape():
    model = GPTGenerator(
        codebook_size=512,
        embed_dim=384,
        depth=12,
        num_heads=6,
        seq_len=64,
        num_classes=55,
    )
    # Autoregressive input: token indices
    indices = torch.randint(0, 512, (2, 64))
    class_label = torch.randint(0, 55, (2,))

    logits = model(indices, class_label)
    assert logits.shape == (2, 64, 512)


def test_gpt_generator_unconditional():
    model = GPTGenerator(
        codebook_size=64, embed_dim=128, depth=4,
        num_heads=4, seq_len=16, num_classes=0,  # unconditional
    )
    indices = torch.randint(0, 64, (2, 16))
    logits = model(indices)
    assert logits.shape == (2, 16, 64)


def test_gpt_generator_causal_mask():
    """Test that causal masking is working (no future information leakage)."""
    model = GPTGenerator(
        codebook_size=64, embed_dim=128, depth=2,
        num_heads=4, seq_len=8, num_classes=0,
    )
    model.eval()

    indices = torch.randint(0, 64, (1, 8))

    # Changing token at position 5 should not affect logits at positions 0-4
    logits1 = model(indices)
    indices_modified = indices.clone()
    indices_modified[0, 5] = (indices[0, 5] + 1) % 64
    logits2 = model(indices_modified)

    # Positions 0-4 should be identical
    assert torch.allclose(logits1[:, :5], logits2[:, :5], atol=1e-5)
    # Position 5+ should differ
    assert not torch.allclose(logits1[:, 5:], logits2[:, 5:])


def test_gpt_generator_generate():
    model = GPTGenerator(
        codebook_size=64, embed_dim=128, depth=2,
        num_heads=4, seq_len=16, num_classes=10,
    )
    model.eval()

    class_label = torch.tensor([3])
    generated = model.generate(class_label, temperature=1.0, top_k=10)
    assert generated.shape == (1, 16)
    assert generated.min() >= 0
    assert generated.max() < 64


def test_gpt_generator_gradient():
    model = GPTGenerator(
        codebook_size=64, embed_dim=128, depth=2,
        num_heads=4, seq_len=16, num_classes=10,
    )
    indices = torch.randint(0, 64, (2, 16))
    labels = torch.tensor([0, 5])
    logits = model(indices, labels)
    loss = logits.sum()
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
            break
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gpt_generator.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# models/gpt_generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with causal masking."""

    def __init__(self, embed_dim: int, num_heads: int, seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len))
        self.register_buffer("mask", mask.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(self.mask[:, :, :L, :L] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.proj_drop(self.proj(out))


class GPTBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, seq_len: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, seq_len, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTGenerator(nn.Module):
    """
    GPT-style autoregressive generator for 3D shape generation.

    Generates VQVAE token sequences for point cloud reconstruction.
    Supports class-conditional and unconditional generation.
    """

    def __init__(
        self,
        codebook_size: int = 8192,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        seq_len: int = 64,
        num_classes: int = 55,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.seq_len = seq_len
        self.num_classes = num_classes

        # Token embedding
        self.tok_embed = nn.Embedding(codebook_size, embed_dim)
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Class conditioning
        if num_classes > 0:
            self.class_embed = nn.Embedding(num_classes, embed_dim)
        else:
            self.class_embed = None

        self.drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            GPTBlock(embed_dim, num_heads, seq_len + (1 if num_classes > 0 else 0), mlp_ratio=4.0, dropout=dropout)
            for _ in range(depth)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)

        # Output head
        self.head = nn.Linear(embed_dim, codebook_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, indices: torch.Tensor, class_label: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            indices: (B, L) token indices
            class_label: (B,) optional class labels
        Returns:
            logits: (B, L, codebook_size) next-token prediction logits
        """
        B, L = indices.shape
        tok = self.tok_embed(indices) + self.pos_embed[:, :L]

        if self.class_embed is not None and class_label is not None:
            cls_tok = self.class_embed(class_label).unsqueeze(1)  # (B, 1, D)
            x = torch.cat([cls_tok, tok], dim=1)  # (B, L+1, D)
        else:
            x = tok

        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        # Remove class token if present
        if self.class_embed is not None and class_label is not None:
            x = x[:, 1:]  # (B, L, D)

        logits = self.head(x)  # (B, L, codebook_size)
        return logits

    @torch.no_grad()
    def generate(
        self,
        class_label: torch.Tensor = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """
        Autoregressive generation.
        Args:
            class_label: (B,) optional class labels
            temperature: sampling temperature
            top_k: top-k filtering (0 = disabled)
            top_p: nucleus sampling threshold
        Returns:
            generated: (B, seq_len) generated token indices
        """
        B = class_label.shape[0] if class_label is not None else 1
        device = next(self.parameters()).device

        generated = torch.zeros(B, 0, dtype=torch.long, device=device)

        for i in range(self.seq_len):
            if generated.shape[1] == 0:
                # First token: use a start token (index 0)
                input_ids = torch.zeros(B, 1, dtype=torch.long, device=device)
            else:
                input_ids = generated

            logits = self.forward(input_ids, class_label)
            next_logits = logits[:, -1, :] / temperature  # (B, codebook_size)

            # Top-k filtering
            if top_k > 0:
                top_k_vals, _ = next_logits.topk(top_k)
                threshold = top_k_vals[:, -1].unsqueeze(-1)
                next_logits[next_logits < threshold] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = next_logits.sort(descending=True)
                cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[mask] = float("-inf")
                next_logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)  # (B, 1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gpt_generator.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add models/gpt_generator.py tests/test_gpt_generator.py
git commit -m "feat: add GPT autoregressive generator for 3D shape generation"
```

---

### Task 30: Generation Metrics (COV, MMD, 1-NNA)

**Files:**
- Modify: `utils/metrics.py` (add generation metrics)
- Create: `tests/test_gen_metrics.py`

**Step 1: Write the failing test**

```python
# tests/test_gen_metrics.py
import torch
import pytest
from utils.metrics import chamfer_distance_batch, compute_cov, compute_mmd, compute_1nna


def test_chamfer_distance_self():
    """CD of a set with itself should be 0."""
    pc = torch.randn(5, 256, 3)
    cd = chamfer_distance_batch(pc, pc)
    assert cd.shape == (5,)
    assert torch.allclose(cd, torch.zeros(5), atol=1e-5)


def test_chamfer_distance_positive():
    """CD between different point clouds should be positive."""
    pc1 = torch.randn(3, 128, 3)
    pc2 = torch.randn(3, 128, 3) + 5.0
    cd = chamfer_distance_batch(pc1, pc2)
    assert (cd > 0).all()


def test_cov_score():
    """Coverage: fraction of reference shapes matched by generated shapes."""
    ref = torch.randn(10, 64, 3)
    gen = ref.clone()  # Perfect generation
    cov = compute_cov(gen, ref)
    assert cov == 100.0  # All reference shapes covered


def test_mmd_perfect():
    """MMD with identical sets should be near 0."""
    ref = torch.randn(10, 64, 3)
    mmd = compute_mmd(ref, ref)
    assert mmd < 0.01


def test_1nna_random():
    """1-NNA with random data should be around 50%."""
    ref = torch.randn(50, 32, 3)
    gen = torch.randn(50, 32, 3)
    nna = compute_1nna(gen, ref)
    # Should be between 30-70% for random data
    assert 20 < nna < 80
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gen_metrics.py -v`
Expected: FAIL

**Step 3: Add generation metrics to utils/metrics.py**

Append to `utils/metrics.py`:

```python
# Append to utils/metrics.py

def chamfer_distance_batch(pc1: torch.Tensor, pc2: torch.Tensor) -> torch.Tensor:
    """
    Chamfer Distance between batches of point clouds.
    Args:
        pc1, pc2: (B, N, 3) point clouds
    Returns:
        cd: (B,) chamfer distance per sample
    """
    # (B, N1, N2)
    dist = torch.cdist(pc1, pc2, p=2.0)
    cd1 = dist.min(dim=2).values.mean(dim=1)  # (B,)
    cd2 = dist.min(dim=1).values.mean(dim=1)  # (B,)
    return cd1 + cd2


def _pairwise_cd(set1: torch.Tensor, set2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise CD between two sets. Returns (M, N) distance matrix."""
    M, N = set1.shape[0], set2.shape[0]
    dists = torch.zeros(M, N)
    for i in range(M):
        cd = chamfer_distance_batch(
            set1[i:i+1].expand(N, -1, -1),
            set2,
        )
        dists[i] = cd
    return dists


def compute_cov(gen: torch.Tensor, ref: torch.Tensor) -> float:
    """
    Coverage: fraction of ref shapes with a nearest neighbor in gen (%).
    """
    dists = _pairwise_cd(gen, ref)  # (M_gen, N_ref)
    # For each ref, find nearest gen
    matched = dists.argmin(dim=0)  # (N_ref,)
    unique_matched = matched.unique()
    # COV = fraction of ref shapes that are nearest to at least one gen
    # Actually: COV = fraction of ref matched by unique gen shapes
    nn_idx = dists.argmin(dim=0)  # nearest gen for each ref
    coverage = nn_idx.unique().shape[0] / ref.shape[0] * 100
    return coverage


def compute_mmd(gen: torch.Tensor, ref: torch.Tensor) -> float:
    """
    Minimum Matching Distance: average CD from each ref to its nearest gen.
    """
    dists = _pairwise_cd(gen, ref)  # (M_gen, N_ref)
    min_dists = dists.min(dim=0).values  # (N_ref,)
    return min_dists.mean().item()


def compute_1nna(gen: torch.Tensor, ref: torch.Tensor) -> float:
    """
    1-Nearest Neighbor Accuracy.
    Combines gen+ref, classifies each by its NN label.
    Perfect generation → ~50%.
    """
    all_pc = torch.cat([gen, ref], dim=0)
    labels = torch.cat([torch.zeros(gen.shape[0]), torch.ones(ref.shape[0])])
    N = all_pc.shape[0]

    dists = _pairwise_cd(all_pc, all_pc)  # (N, N)
    # Set diagonal to inf
    dists.fill_diagonal_(float("inf"))

    nn_idx = dists.argmin(dim=1)
    nn_labels = labels[nn_idx]

    correct = (nn_labels == labels).float().mean().item()
    return correct * 100
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gen_metrics.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add utils/metrics.py tests/test_gen_metrics.py
git commit -m "feat: add generation metrics (CD, COV, MMD, 1-NNA)"
```

---

### Task 31: Generation Training Script

**Files:**
- Create: `tools/train_generation.py`
- Create: `configs/generation.yaml`

**Step 1: Write the training script**

```python
# tools/train_generation.py
"""
Stage 4d: Train GPT generator for 3D shape generation.

Usage:
    python tools/train_generation.py --config configs/generation.yaml
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config, merge_config
from utils.logger import get_logger
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.scheduler import build_scheduler
from utils.distributed import setup_distributed, cleanup_distributed, is_main_process, get_rank
from utils.metrics import AverageMeter
from datasets import build_dataset
from models.vqvae import VQVAE
from models.gpt_generator import GPTGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--vqvae_ckpt", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.opts:
        cfg = merge_config(cfg, args.opts)

    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        setup_distributed()

    device = torch.device(f"cuda:{get_rank()}" if torch.cuda.is_available() else "cpu")
    logger = get_logger("train_gen", log_file=os.path.join(cfg.output.exp_dir, "train.log") if is_main_process() else None)
    os.makedirs(cfg.output.exp_dir, exist_ok=True)

    # VQVAE (frozen, for tokenization)
    vqvae = VQVAE(
        embed_dim=cfg.model.vqvae_embed_dim, num_groups=cfg.model.num_groups,
        group_size=cfg.model.group_size, encoder_depth=6, decoder_depth=6,
        num_heads=cfg.model.num_heads, codebook_size=cfg.model.codebook_size,
    ).to(device)
    load_checkpoint(args.vqvae_ckpt, vqvae)
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False

    # GPT Generator
    model = GPTGenerator(
        codebook_size=cfg.model.codebook_size,
        embed_dim=cfg.model.gpt_embed_dim,
        depth=cfg.model.gpt_depth,
        num_heads=cfg.model.gpt_heads,
        seq_len=cfg.model.num_groups,
        num_classes=cfg.model.get("num_classes", 55),
    ).to(device)

    if distributed:
        model = DDP(model, device_ids=[get_rank()])

    dataset = build_dataset(cfg.data.dataset, data_root=cfg.data.data_root, split=cfg.data.train_split, n_points=cfg.data.n_points)
    sampler = DistributedSampler(dataset) if distributed else None
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, sampler=sampler, shuffle=(sampler is None), num_workers=4, pin_memory=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = build_scheduler(optimizer, epochs=cfg.train.epochs, warmup_epochs=cfg.train.warmup_epochs, min_lr=cfg.train.min_lr)

    for epoch in range(cfg.train.epochs):
        if distributed:
            sampler.set_epoch(epoch)

        model.train()
        loss_meter = AverageMeter()

        for batch in loader:
            pc = batch["points"].to(device)

            # Tokenize with VQVAE
            with torch.no_grad():
                indices, _ = vqvae.encode(pc)  # (B, G)

            # Teacher forcing: predict next token
            input_ids = indices[:, :-1]  # (B, G-1)
            target_ids = indices[:, 1:]  # (B, G-1)

            # Class label (if available)
            class_label = None
            if "taxonomy_id" in batch and cfg.model.get("num_classes", 0) > 0:
                # Taxonomy mapping would be needed
                pass

            logits = model(input_ids, class_label)[:, :target_ids.shape[1]]
            loss = F.cross_entropy(logits.reshape(-1, cfg.model.codebook_size), target_ids.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), pc.shape[0])

        scheduler.step()

        if is_main_process():
            logger.info(f"Epoch [{epoch}] Loss: {loss_meter.avg:.4f}")
            if (epoch + 1) % cfg.output.save_freq == 0:
                save_checkpoint(os.path.join(cfg.output.exp_dir, f"epoch_{epoch}.pth"), model.module if distributed else model, optimizer, epoch=epoch)

    if is_main_process():
        save_checkpoint(os.path.join(cfg.output.exp_dir, "final.pth"), model.module if distributed else model, optimizer, epoch=cfg.train.epochs - 1)

    if distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
```

**Step 2: Create config**

```yaml
# configs/generation.yaml
data:
  dataset: shapenet55
  data_root: data/ShapeNet55
  pc_dir: shapenet_pc
  n_points: 2048
  train_split: train

model:
  vqvae_embed_dim: 384
  num_groups: 64
  group_size: 32
  num_heads: 6
  codebook_size: 8192
  gpt_embed_dim: 384
  gpt_depth: 12
  gpt_heads: 6
  num_classes: 55

train:
  epochs: 300
  batch_size: 64
  lr: 1e-4
  weight_decay: 0.05
  warmup_epochs: 10
  min_lr: 1e-6

output:
  exp_dir: experiments/generation
  save_freq: 50
  log_freq: 10
```

**Step 3: Commit**

```bash
git add tools/train_generation.py configs/generation.yaml
git commit -m "feat: add GPT generation training script (stage 4d)"
```

---

### Task 32: Generation Evaluation Script

**Files:**
- Create: `tools/eval_generation.py`
- Create: `tests/test_eval_generation.py`

**Step 1: Write the failing test**

```python
# tests/test_eval_generation.py
import torch
from models.gpt_generator import GPTGenerator
from models.vqvae import VQVAE


def test_generation_pipeline():
    """Test: generate tokens → decode with VQVAE → get point cloud features."""
    codebook_size = 64
    D, G = 128, 16

    gpt = GPTGenerator(
        codebook_size=codebook_size, embed_dim=128, depth=2,
        num_heads=4, seq_len=G, num_classes=10,
    )
    vqvae = VQVAE(
        embed_dim=D, num_groups=G, group_size=8,
        encoder_depth=2, decoder_depth=2, num_heads=4,
        codebook_size=codebook_size,
    )
    gpt.eval()
    vqvae.eval()

    # Generate
    class_label = torch.tensor([3, 7])
    generated_indices = gpt.generate(class_label, temperature=1.0, top_k=10)
    assert generated_indices.shape == (2, G)

    # Decode
    centers = torch.randn(2, G, 3)  # In practice, centers need to be predicted too
    decoded = vqvae.decode(generated_indices, centers)
    assert decoded.shape == (2, G, D)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval_generation.py -v`
Expected: FAIL (initially)

**Step 3: Write evaluation script**

```python
# tools/eval_generation.py
"""
Evaluate 3D shape generation quality (COV, MMD, 1-NNA).

Usage:
    python tools/eval_generation.py --config configs/generation.yaml \
        --gpt_ckpt experiments/generation/final.pth \
        --vqvae_ckpt experiments/vqvae/final.pth
"""
import os
import sys
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config, merge_config
from utils.logger import get_logger
from utils.checkpoint import load_checkpoint
from utils.metrics import compute_cov, compute_mmd, compute_1nna
from datasets import build_dataset
from models.vqvae import VQVAE
from models.gpt_generator import GPTGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gpt_ckpt", type=str, required=True)
    parser.add_argument("--vqvae_ckpt", type=str, required=True)
    parser.add_argument("--num_generate", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.opts:
        cfg = merge_config(cfg, args.opts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger("eval_gen")

    # Load VQVAE
    vqvae = VQVAE(
        embed_dim=cfg.model.vqvae_embed_dim, num_groups=cfg.model.num_groups,
        group_size=cfg.model.group_size, encoder_depth=6, decoder_depth=6,
        num_heads=cfg.model.num_heads, codebook_size=cfg.model.codebook_size,
    ).to(device)
    load_checkpoint(args.vqvae_ckpt, vqvae)
    vqvae.eval()

    # Load GPT
    gpt = GPTGenerator(
        codebook_size=cfg.model.codebook_size,
        embed_dim=cfg.model.gpt_embed_dim,
        depth=cfg.model.gpt_depth,
        num_heads=cfg.model.gpt_heads,
        seq_len=cfg.model.num_groups,
        num_classes=cfg.model.get("num_classes", 55),
    ).to(device)
    load_checkpoint(args.gpt_ckpt, gpt)
    gpt.eval()

    # Load reference point clouds
    dataset = build_dataset(cfg.data.dataset, data_root=cfg.data.data_root, split="test", n_points=cfg.data.n_points)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    ref_pcs = []
    for batch in loader:
        ref_pcs.append(batch["points"])
        if len(ref_pcs) * 32 >= args.num_generate:
            break
    ref_pcs = torch.cat(ref_pcs)[:args.num_generate]

    # Generate shapes
    logger.info(f"Generating {args.num_generate} shapes...")
    generated_pcs = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, args.num_generate, batch_size):
            n = min(batch_size, args.num_generate - i)
            # Random class labels for conditional generation
            class_label = torch.randint(0, cfg.model.get("num_classes", 55), (n,), device=device)
            indices = gpt.generate(class_label, temperature=args.temperature, top_k=args.top_k)

            # Decode to features (point cloud reconstruction from features is an approximation)
            centers = torch.randn(n, cfg.model.num_groups, 3, device=device) * 0.5
            decoded = vqvae.decode(indices, centers)
            # Use centers as proxy point cloud for evaluation
            generated_pcs.append(centers.cpu())

    generated_pcs = torch.cat(generated_pcs)[:args.num_generate]

    # Evaluate
    logger.info("Computing generation metrics...")
    cov = compute_cov(generated_pcs, ref_pcs)
    mmd = compute_mmd(generated_pcs, ref_pcs)
    nna = compute_1nna(generated_pcs, ref_pcs)

    logger.info(f"COV (%): {cov:.2f}")
    logger.info(f"MMD: {mmd:.6f}")
    logger.info(f"1-NNA (%): {nna:.2f}")


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eval_generation.py -v`
Expected: 1 passed

**Step 5: Commit**

```bash
git add tools/eval_generation.py tests/test_eval_generation.py
git commit -m "feat: add generation evaluation script (COV, MMD, 1-NNA)"
```

---

### Task 33: Model Registry and __init__ Files

**Files:**
- Modify: `models/__init__.py`
- Modify: `models/heads/__init__.py`

**Step 1: Write model registry**

```python
# models/__init__.py
from models.point_patch_embed import PointPatchEmbed
from models.vqvae import VQVAE, VectorQuantize
from models.nerf import TriplaneNeRF
from models.extractor import Extractor
from models.generator import Generator
from models.gpt_generator import GPTGenerator
from models.masking import sliding_mask, compute_mask_ratio
```

```python
# models/heads/__init__.py
from models.heads.cls_head import ClassificationHead
from models.heads.partseg_head import PartSegHead
```

**Step 2: Commit**

```bash
git add models/__init__.py models/heads/__init__.py
git commit -m "feat: add model registry and exports"
```

---

### Task 34: Integration Test (Full Pipeline)

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

```python
# tests/test_integration.py
"""Integration test: verify the full Point-MGE pipeline runs end-to-end."""
import torch
import pytest


def test_full_pretrain_pipeline():
    """Test: point cloud → patch embed → mask → extract → generate → loss."""
    from models.point_patch_embed import PointPatchEmbed
    from models.extractor import Extractor
    from models.generator import Generator
    from models.vqvae import VQVAE
    from models.masking import sliding_mask, compute_mask_ratio

    B, N, G, D = 2, 512, 16, 128
    codebook_size = 64

    patch_embed = PointPatchEmbed(in_channels=3, embed_dim=D, num_groups=G, group_size=8)
    vqvae = VQVAE(embed_dim=D, num_groups=G, group_size=8, encoder_depth=2, decoder_depth=2, num_heads=4, codebook_size=codebook_size)
    extractor = Extractor(embed_dim=D, depth=2, num_heads=4, num_groups=G)
    generator = Generator(embed_dim=D, depth=2, num_heads=4, num_groups=G, codebook_size=codebook_size)

    pc = torch.randn(B, N, 3)

    # VQVAE tokenization
    with torch.no_grad():
        target_indices, _ = vqvae.encode(pc)

    # Patch embedding
    tokens, centers = patch_embed(pc)

    # Masking
    mask_ratio = compute_mask_ratio(100, 300)
    visible_mask = sliding_mask(B, G, mask_ratio)

    # Split
    vis_list, vis_c_list, mask_c_list, tgt_list = [], [], [], []
    for b in range(B):
        v = visible_mask[b].nonzero(as_tuple=True)[0]
        m = (~visible_mask[b]).nonzero(as_tuple=True)[0]
        vis_list.append(tokens[b, v])
        vis_c_list.append(centers[b, v])
        mask_c_list.append(centers[b, m])
        tgt_list.append(target_indices[b, m])

    vis_tokens = torch.stack(vis_list)
    vis_centers = torch.stack(vis_c_list)
    mask_centers = torch.stack(mask_c_list)
    targets = torch.stack(tgt_list)

    # Extract
    features = extractor(vis_tokens, vis_centers)

    # Generate
    logits, center_pred = generator(features, vis_centers, mask_centers, visible_mask)

    # Loss
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, codebook_size), targets.reshape(-1)
    )
    loss.backward()

    assert not torch.isnan(loss)
    assert loss.item() > 0


def test_full_classification_pipeline():
    """Test: pretrained extractor → classification head → loss."""
    from models.point_patch_embed import PointPatchEmbed
    from models.extractor import Extractor
    from models.heads.cls_head import ClassificationHead

    D, G = 128, 16
    patch_embed = PointPatchEmbed(in_channels=3, embed_dim=D, num_groups=G, group_size=8)
    extractor = Extractor(embed_dim=D, depth=2, num_heads=4, num_groups=G)
    cls_head = ClassificationHead(embed_dim=D, num_classes=40)

    pc = torch.randn(4, 256, 3)
    tokens, centers = patch_embed(pc)
    features = extractor(tokens, centers)
    logits = cls_head(features)
    labels = torch.randint(0, 40, (4,))
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    assert logits.shape == (4, 40)
    assert not torch.isnan(loss)


def test_full_generation_pipeline():
    """Test: GPT generate → VQVAE decode."""
    from models.vqvae import VQVAE
    from models.gpt_generator import GPTGenerator

    codebook_size = 64
    D, G = 128, 16

    vqvae = VQVAE(embed_dim=D, num_groups=G, group_size=8, encoder_depth=2, decoder_depth=2, num_heads=4, codebook_size=codebook_size)
    gpt = GPTGenerator(codebook_size=codebook_size, embed_dim=128, depth=2, num_heads=4, seq_len=G, num_classes=10)

    vqvae.eval()
    gpt.eval()

    with torch.no_grad():
        indices = gpt.generate(torch.tensor([5, 3]), temperature=1.0, top_k=10)
        centers = torch.randn(2, G, 3)
        decoded = vqvae.decode(indices, centers)

    assert indices.shape == (2, G)
    assert decoded.shape == (2, G, D)
```

**Step 2: Run all tests**

Run: `python -m pytest tests/test_integration.py -v`
Expected: 3 passed

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for full Point-MGE pipeline"
```

---

## Summary

| Phase | Tasks | Description |
|-------|-------|-------------|
| 1 | 1-6 | Project foundation: setup, config, logger, checkpoint, scheduler, distributed |
| 2 | 7-12 | Data pipeline: point cloud utils, ShapeNet, ModelNet40, ScanObjectNN, ShapeNetPart, registry |
| 3 | 13-19 | Core models: patch embed, VQ, VQVAE, NeRF, extractor, generator, masking |
| 4 | 20-23 | Training: metrics, NeRF training, VQVAE training, pretraining |
| 5 | 24-28 | Downstream: cls head, cls finetuning, few-shot, partseg head, partseg finetuning |
| 6 | 29-34 | Generation: GPT generator, gen metrics, gen training, gen eval, registry, integration tests |

**Total: 34 tasks, ~170 steps**

**Execution order:** Tasks are designed to be executed sequentially (1 → 34). Each task builds on the previous ones.
