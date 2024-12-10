from typing import Dict, Any, Union, List
from yaml_ml.special_types import LoadedYAMLType
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
    score: Union[str, List[str]] = data["score"]
    output_folder: str = data["output_folder"]
    name: str = data["name"]

    return logs, trg_var, loading_cfg, prepro_cfg, split_cfg, model_cfg, score, output_folder, name