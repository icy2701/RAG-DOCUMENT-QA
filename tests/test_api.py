import pytest
import sys
import os

# Add the src/ folder to Python's search path
# os.path.dirname(__file__)        → tests/
# os.path.join(..., '..', 'src')   → src/
# sys.path.append(...)             → Python can now find main.py inside src/
# Without this, "from main import app" would fail because Python
# doesn't know where to look for main.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# TestClient is FastAPI's built-in testing tool
# It lets us send fake HTTP requests to our app without
# running a real server — perfect for automated testing
# Think of it as a robot that pretends to be a browser
from fastapi.testclient import TestClient

# We import the actual FastAPI app object from main.py
# This is the same app that runs when you do uvicorn src.main:app
from main import app

# ── SHARED TEST CLIENT (FIXTURE) ──────────────────────────────────────────────

# @pytest.fixture is a special decorator that creates reusable test setup
# scope="module" means this fixture is created ONCE for the entire test file
# and shared across all test functions — not recreated for each test
# This is important because loading models takes 10-30 seconds
# We don't want to reload them for every single test function

# The `with TestClient(app) as c:` syntax is critical
# Using TestClient as a context manager (with block) properly triggers
# the lifespan function in main.py — which loads our models
# Without the `with` block, startup never fires, vectorstore stays None,
# and every /ask test crashes with "NoneType has no attribute similarity_search"
# yield c passes the ready client to every test function that requests it
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

# ── TEST 1: HEALTH ENDPOINT ───────────────────────────────────────────────────

# Every test function must start with test_ so pytest finds it automatically
# We pass `client` as a parameter — pytest sees this matches our fixture name
# and automatically provides the TestClient we created above
# This test verifies two things:
#   1. /health returns HTTP status 200 (success)
#   2. The response body is exactly {"status": "ok"}
def test_health_returns_200(client):
    # client.get("/health") sends a GET request to our /health endpoint
    # exactly like typing http://127.0.0.1:8000/health in a browser
    response = client.get("/health")

    # assert means "this must be true — if not, the test fails immediately"
    # response.status_code is the HTTP status number the server returned
    # 200 = success, 404 = not found, 422 = invalid input, 500 = server error
    assert response.status_code == 200

    # response.json() converts the response body from JSON string to Python dict
    # We check the entire dictionary matches exactly what we expect
    assert response.json() == {"status": "ok"}

# ── TEST 2: ASK ENDPOINT RETURNS 200 ─────────────────────────────────────────

# This test checks that /ask accepts a valid question and returns 200
# We use an ML question because we know ml_basics.txt covers it
# We only check the status code here — not the answer content
# because answer content can vary slightly between runs
def test_ask_returns_200(client):
    # client.post() sends a POST request with a JSON body
    # json= automatically converts the Python dictionary to JSON format
    # FastAPI receives this, validates it against our Question schema,
    # and passes it to the ask() function
    response = client.post(
        "/ask",
        json={"question": "What is machine learning?", "top_k": 3}
    )

    # If this is 200 we know the entire pipeline ran without crashing:
    # FAISS search worked, flan-t5 generated something, response was built
    assert response.status_code == 200

# ── TEST 3: ASK RESPONSE HAS ANSWER KEY ──────────────────────────────────────

# This test checks the STRUCTURE of the response not just the status code
# We verify the response body contains an "answer" key
# and that its value is actually a string
# We don't check the exact answer text because:
#   - flan-t5 might generate slightly different wording each time
#   - what matters is that an answer EXISTS and is a string
def test_ask_response_has_answer_key(client):
    response = client.post(
        "/ask",
        json={"question": "What is overfitting?", "top_k": 3}
    )

    assert response.status_code == 200

    # Parse the response body as a Python dictionary
    data = response.json()

    # Check the "answer" key exists in the dictionary
    assert "answer" in data

    # isinstance(data["answer"], str) checks that the value is a string type
    # not None, not a number, not a list — a proper text string
    assert isinstance(data["answer"], str)

# ── TEST 4: ASK RESPONSE HAS SOURCES KEY ─────────────────────────────────────

# This test checks that sources are returned correctly
# Sources are what make RAG trustworthy — every answer shows evidence
# We verify three things:
#   1. "sources" key exists in the response
#   2. Its value is a list
#   3. The list has exactly 3 items (because we sent top_k=3)
def test_ask_response_has_sources_key(client):
    response = client.post(
        "/ask",
        json={"question": "What are Python data types?", "top_k": 3}
    )

    assert response.status_code == 200

    data = response.json()

    # Check "sources" key exists
    assert "sources" in data

    # Check it's a list not some other type
    assert isinstance(data["sources"], list)

    # Check we got exactly top_k=3 sources back
    # If this fails it means FAISS returned wrong number of results
    assert len(data["sources"]) == 3

# ── TEST 5: ASK REJECTS MISSING QUESTION ─────────────────────────────────────

# This test checks that our input validation works correctly
# We send an empty JSON body with no question field
# FastAPI should automatically return 422 (Unprocessable Entity)
# because "question" is a required field in our Question BaseModel
# This proves our schema validation is working — bad input is rejected
# before it reaches our RAG pipeline
# 422 is the standard HTTP code for "I understood you but your data is wrong"
def test_ask_rejects_missing_question(client):
    # Send completely empty body — no question, no top_k
    response = client.post("/ask", json={})

    # FastAPI's Pydantic validation catches this automatically
    # Our ask() function never even runs
    assert response.status_code == 422