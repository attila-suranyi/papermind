from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api import app


@pytest.fixture
def client():
    """Provides a TestClient for FastAPI app.

    Note: We're using a context manager if we want to trigger lifespan events,
    but for this simple endpoint check, direct TestClient is often enough.
    However, our /index endpoint uses app.state which is set in lifespan.
    """
    with TestClient(app) as c:
        # Mock pipelines to avoid heavy dependencies in tests
        c.app.state.ingestion_pipeline = MagicMock()
        c.app.state.retrieval_pipeline = MagicMock()
        yield c


@pytest.fixture
def dummy_pdf(tmp_path, client):
    """Creates a temporary dummy PDF file for testing and cleans up the copy in DOCS_DIR."""
    pdf_file = tmp_path / "test_file.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF")
    yield pdf_file

    # Clean up the file if it was moved to DOCS_DIR by the API
    docs_pdf = client.app.state.settings.DOCS_DIR / pdf_file.name
    if docs_pdf.exists():
        docs_pdf.unlink()


def test_health(client):
    """Verifies the health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_pdf(client, dummy_pdf):
    """Verifies that uploading a PDF returns 202 and correct filename."""
    with open(dummy_pdf, "rb") as f:
        files = {"file": (dummy_pdf.name, f, "application/pdf")}
        response = client.post("/index", files=files)

    assert response.status_code == 202
    assert response.json()["status"] == "ok"
    assert response.json()["filename"] == dummy_pdf.name


def test_upload_invalid_file(client, tmp_path):
    """Verifies that non-PDF files are rejected."""
    txt_file = tmp_path / "test_file.txt"
    txt_file.write_text("not a pdf")

    with open(txt_file, "rb") as f:
        files = {"file": (txt_file.name, f, "text/plain")}
        response = client.post("/index", files=files)

    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]


def test_answer_success(client):
    """Verifies the answer endpoint."""
    client.app.state.retrieval_pipeline.get_answer.return_value = "Mocked answer"

    response = client.post("/answer", json={"query": "What is the meaning of life?"})

    assert response.status_code == 200
    assert response.json() == {"query": "What is the meaning of life?", "answer": "Mocked answer"}
    client.app.state.retrieval_pipeline.get_answer.assert_called_once_with(
        "What is the meaning of life?"
    )


def test_answer_failure(client):
    """Verifies the answer endpoint failure case."""
    client.app.state.retrieval_pipeline.get_answer.return_value = None

    response = client.post("/answer", json={"query": "fail?"})

    assert response.status_code == 500
    assert "failed to produce an answer" in response.json()["detail"]
