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
