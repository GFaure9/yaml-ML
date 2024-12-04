from typing import Dict, Any
from src.types import LoadedYAMLType
import yaml


def load_yaml_config(filepath: str) -> LoadedYAMLType:
    with open(filepath, "r") as f:
        data = yaml.safe_load(f)

    logs: bool = data["logs"]
    trg_var: str = data["target_var"]
    loading_cfg: Dict[str, str] = data["loading"]
    prepro_cfg: Dict[str, Dict[str, Any]] = data["preprocessing"]
    split_cfg: Dict[str, Dict[str, Any]] = data["dataset"]["split"]
    model_cfg: Dict[str, Dict[str, Dict[str, Any]]] = data["model"]
    score: str = data["score"]

    return logs, trg_var, loading_cfg, prepro_cfg, split_cfg, model_cfg, score