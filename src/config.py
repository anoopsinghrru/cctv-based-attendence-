"""Load config.yaml and resolve project paths."""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    # DIP_CONFIG lets the tests (and alternative deployments) point at another config file
    path = Path(path or os.environ.get("DIP_CONFIG") or ROOT / "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # make every path absolute so scripts work from any working directory
    cfg["paths"] = {k: str((ROOT / v).resolve()) for k, v in cfg["paths"].items()}
    cfg["models"]["dir"] = str((ROOT / cfg["models"]["dir"]).resolve())
    for key in ("faces", "snapshots", "reports"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["db"]).parent.mkdir(parents=True, exist_ok=True)
    return cfg
