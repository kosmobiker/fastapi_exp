import unittest
import pandas as pd
import os
import tempfile

from src.train.utils import get_data
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


class TestTrainer:
    def test_fake_get_data(self):
        df = _fake_get_data(1000)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (1000, 32)
        assert df["income"].dtype == "float"
