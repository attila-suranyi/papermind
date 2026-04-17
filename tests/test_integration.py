import os
import shutil
from pathlib import Path

import pytest
import requests
from fastapi.testclient import TestClient

from config import load_settings


def is_ollama_running():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


@pytest.fixture(scope="session", autouse=True)
def check_ollama():
    """Verify Ollama is running before starting any tests."""
    if not is_ollama_running():
        pytest.fail("Ollama is not running. Please start Ollama before running integration tests.")


@pytest.fixture
def test_env():
    """Sets up a real but temporary testing environment using the test config."""
    os.environ["APP_ENV"] = "test"

    test_settings = load_settings("test")

    # Import app after setting APP_ENV so lifespan picks up the test config
    from app.api import app

    with TestClient(app) as client:
        yield {"client": client}

    # Teardown: Remove directories defined in test.yaml
    if test_settings.VECTOR_DB_DIR.exists():
        shutil.rmtree(test_settings.VECTOR_DB_DIR)
    if test_settings.DOCS_DIR.exists():
        shutil.rmtree(test_settings.DOCS_DIR)


def test_answer_is_given(test_env):
    """
    Test the full RAG flow via API: Index a PDF and ask a question.
    """
    client = test_env["client"]

    # 1. Discover the PDF file starting with "short_" in the tests directory
    test_dir = Path("tests")
    pdf_files = list(test_dir.glob("short_*.pdf"))

    assert len(pdf_files) > 0, f"No PDF file starting with 'short_' found in {test_dir}"
    pdf_path = pdf_files[0]

    # 2. Call the /index endpoint
    with open(pdf_path, "rb") as f:
        files = {"file": (pdf_path.name, f, "application/pdf")}
        response = client.post("/index", files=files)

    assert response.status_code == 202
    assert response.json()["status"] == "ok"
    assert response.json()["filename"] == pdf_path.name

    # 3. Query the documents via /answer endpoint
    query = "What is the main topic of the document?"
    response = client.post("/answer", json={"query": query})

    assert response.status_code == 200
    answer_data = response.json()
    assert "answer" in answer_data
    answer = answer_data["answer"]

    assert isinstance(answer, str)
    assert len(answer.strip()) > 0

    print(f"\nQuery: {query}")
    print(f"Answer: {answer}")
