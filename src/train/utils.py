import yaml
from typing import Any


def load_yaml_file(file_path: str) -> dict[str, Any]:
    """
    Load data from a YAML file.
    """
    with open(file_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
