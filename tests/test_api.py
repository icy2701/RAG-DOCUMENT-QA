import sys
import os
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from main import app


# scope="module" loads models once for all tests rather than per test
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_200(client):
    response = client.post("/ask", json={"question": "What is machine learning?", "top_k": 3})
    assert response.status_code == 200


def test_ask_response_has_answer_key(client):
    response = client.post("/ask", json={"question": "What is overfitting?", "top_k": 3})
    data = response.json()
    assert "answer" in data
    assert isinstance(data["answer"], str)


def test_ask_response_has_sources_key(client):
    response = client.post("/ask", json={"question": "What are Python data types?", "top_k": 3})
    data = response.json()
    assert "sources" in data
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) == 3


def test_ask_rejects_missing_question(client):
    response = client.post("/ask", json={})
    assert response.status_code == 422