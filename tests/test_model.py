from unittest.mock import patch

import pytest

from app.ml.model import predict_proba
from tests.conftest import generate_synthetic_user


@pytest.mark.parametrize(
    "proba_value, expected_label, expected_risk",
    [
        (0.9, 1, "high"),
        (0.6, 1, "medium"),
        (0.2, 0, "low"),
    ],
)
@patch("app.ml.model.joblib.load")
@patch("app.ml.model.model")
@patch("app.ml.model.preprocessor")  # Mock the preprocessor directly
def test_predict_proba(
    mock_preprocessor,
    mock_model,
    mock_joblib,
    proba_value,
    expected_label,
    expected_risk,
):
    # Given
    user_data = generate_synthetic_user()
    features_dict = user_data["features"]

    # Mock the preprocessor transform method
    mock_preprocessor.transform.return_value = [[1, 2, 3]]  # Mock transformed values
    mock_model.predict_proba.return_value = [[1 - proba_value, proba_value]]

    # When
    label, proba, risk = predict_proba(features_dict)

    # Then
    assert label == expected_label
    assert proba == proba_value
    assert risk == expected_risk
