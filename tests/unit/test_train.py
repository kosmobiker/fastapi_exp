import unittest
import yaml
from unittest.mock import patch, mock_open
import pandas as pd
from src.train.utils import get_data, load_yaml_file


class TestGetData(unittest.TestCase):
    def test_happy_path(self):
        with patch("os.path.exists", return_value=True) as mock_exists:
            # Create a mock pandas DataFrame
            mock_df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

            # Set the return value of pd.read_csv to the mock DataFrame
            with patch("pd.read_csv", return_value=mock_df) as mock_read_csv:
                result = get_data("path/to/file")
                self.assertEqual(result, mock_df)
                assert len(result) == 3


class TestGetData(unittest.TestCase):
    def test_sad_path(self):
        with patch("os.path.exists", return_value=False) as mock_exists:
            with self.assertRaises(FileNotFoundError):
                get_data("path/to/file")


class TestLoadYamlFile(unittest.TestCase):

    @patch("builtins.open", new_callable=mock_open, read_data="key: value")
    def test_load_yaml_file_success(self, mock_file_open):
        file_path = "test.yaml"
        expected_result = {"key": "value"}

        result = load_yaml_file(file_path)

        mock_file_open.assert_called_once_with(file_path, "r")
        self.assertEqual(result, expected_result)

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_load_yaml_file_file_not_found(self, mock_file_open):
        file_path = "nonexistent.yaml"

        with self.assertRaises(FileNotFoundError):
            load_yaml_file(file_path)
