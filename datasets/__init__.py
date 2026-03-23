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
