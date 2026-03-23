import pytest
from datasets import build_dataset

def test_build_dataset_unknown_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        build_dataset("nonexistent", data_root="/tmp", split="train")
