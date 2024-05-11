"""
This module provides utility functions for data loading and file handling.
"""
from typing import Any
import os
import pandas as pd
import yaml


def get_data(path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    """
    if os.path.exists(path):
        return pd.read_csv(path)
    return None  


def load_yaml_file(file_path: str) -> dict[str, Any]:
    """
    Load data from a YAML file.
    """
    with open(file_path, "r", encoding='utf-8') as stream:  # Specify encoding
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None  
