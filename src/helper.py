import yaml
from pathlib import Path


with open(f"{Path(__file__).parent}/modules.yaml", "r") as f:
    modules = yaml.safe_load(f)