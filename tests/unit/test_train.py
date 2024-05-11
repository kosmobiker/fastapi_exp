import unittest
import pandas as pd
import os
import tempfile

import yaml
from src.train.utils import get_data, load_yaml_file
from tests.functional.test_train import _fake_get_data


class TestGetData(unittest.TestCase):
    def setUp(self):
        self.test_data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        self.test_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        self.test_data.to_csv(self.test_file.name, index=False)

    def tearDown(self):
        os.unlink(self.test_file.name)

    def test_get_data(self):
        result = get_data(self.test_file.name)
        pd.testing.assert_frame_equal(result, self.test_data)

    def test_get_data_non_existent_file(self):
        result = get_data("non_existent_file.csv")
        self.assertIsNone(result)


class TestLoadYamlFile(unittest.TestCase):
    def setUp(self):
        self.test_data = {"A": 1, "B": 2}
        self.test_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
        with open(self.test_file.name, "w") as f:
            yaml.dump(self.test_data, f)

    def tearDown(self):
        os.unlink(self.test_file.name)

    def test_load_yaml_file(self):
        result = load_yaml_file(self.test_file.name)
        self.assertEqual(result, self.test_data)

    def test_load_yaml_file_non_existent_file(self):
        with self.assertRaises(FileNotFoundError):
            load_yaml_file("non_existent_file.yaml")

    def test_load_yaml_file_invalid_yaml(self):
        invalid_yaml_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
        with open(invalid_yaml_file.name, "w") as f:
            f.write("{unclosed_brace")
        result = load_yaml_file(invalid_yaml_file.name)
        os.unlink(invalid_yaml_file.name)
        self.assertIsNone(result)


class TestTrainer:
    def test_fake_get_data(self):
        df = _fake_get_data(1000)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (1000, 32)
        assert df["income"].dtype == "float"
