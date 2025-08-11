import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_healthcheck():
    """
    Given: The API is running.
    When: A GET request is made to the /healthcheck endpoint.
    Then: The response should have a 200 status code, a JSON body with {"status": "ok"},
          and the content-type header should be "application/json".
    """
    # Given
    endpoint = "/healthcheck"

    # When
    response = client.get(endpoint)

    # Then
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert "application/json" in response.headers["content-type"]


def test_healthcheck_post_method_not_allowed():
    """
    Given: The API is running.
    When: A POST request is made to the /healthcheck endpoint.
    Then: The response should have a 405 status code (Method Not Allowed).
    """
    # Given
    endpoint = "/healthcheck"

    # When
    response = client.post(endpoint)

    # Then
    assert response.status_code == 405


@pytest.mark.parametrize(
    "endpoint,expected_status",
    [
        ("/healthcheck", 200),
        ("/nonexistent", 404),
    ],
)
def test_endpoint_status_codes(endpoint, expected_status):
    """
    Given: The API is running.
    When: A GET request is made to various endpoints.
    Then: The response should have the expected status code.
    """
    # When
    response = client.get(endpoint)

    # Then
    assert response.status_code == expected_status
