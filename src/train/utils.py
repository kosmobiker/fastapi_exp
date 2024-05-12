"""
This module provides utility functions for data loading and file handling.
"""

import os
import pandas as pd


def get_data(path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    """
    if os.path.exists(path):
        return pd.read_csv(path)
    return None
