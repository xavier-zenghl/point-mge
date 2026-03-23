import pytest
import os
import tempfile
import yaml
from utils.config import load_config, merge_config


def test_load_config_from_yaml():
    cfg_dict = {"model": {"name": "extractor", "depth": 12, "embed_dim": 384}, "train": {"epochs": 300, "lr": 1e-3}}
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
