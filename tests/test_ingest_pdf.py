import json
from unittest.mock import patch

import pytest

from app.ingestion.ingest_pdf import flatten_metadata, ingest_pdf, ingest_pdfs
from app.model import Chunk


def test_ingest_pdf_raises_when_file_missing(tmp_path):
    """ingest_pdf should raise FileNotFoundError for a non-existent path."""
    with pytest.raises(FileNotFoundError):
        ingest_pdf(tmp_path / "missing.pdf")


def test_ingest_pdfs_raises_when_dir_missing(tmp_path):
    """ingest_pdfs should raise FileNotFoundError for a non-existent directory."""
    with pytest.raises(FileNotFoundError):
        ingest_pdfs(tmp_path / "no_such_dir")


def test_ingest_pdfs_raises_when_path_is_file(tmp_path):
    """ingest_pdfs should raise NotADirectoryError when given a file path."""
    f = tmp_path / "file.txt"
    f.write_text("hello")
    with pytest.raises(NotADirectoryError):
        ingest_pdfs(f)


def test_ingest_pdfs_returns_empty_when_no_pdfs(tmp_path):
    """ingest_pdfs should return an empty dict when the directory has no PDFs."""
    result = ingest_pdfs(tmp_path)
    assert result == {}


def test_ingest_pdfs_calls_ingest_pdf_for_each_pdf(tmp_path):
    """ingest_pdfs should call ingest_pdf once per PDF file found."""
    (tmp_path / "a.pdf").write_bytes(b"%PDF")
    (tmp_path / "b.pdf").write_bytes(b"%PDF")

    fake_chunks = [Chunk(text="chunk", metadata={})]

    with patch("app.ingestion.ingest_pdf.ingest_pdf", return_value=fake_chunks) as mock_ingest:
        result = ingest_pdfs(tmp_path)

    assert mock_ingest.call_count == 2
    assert set(result.keys()) == {"a.pdf", "b.pdf"}


def test_ingest_pdfs_skips_failed_pdf(tmp_path):
    """ingest_pdfs should skip a PDF that raises an exception and continue with others."""
    (tmp_path / "good.pdf").write_bytes(b"%PDF")
    (tmp_path / "bad.pdf").write_bytes(b"%PDF")

    def side_effect(path):
        if path.name == "bad.pdf":
            raise RuntimeError("parse error")
        return [Chunk(text="ok", metadata={})]

    with patch("app.ingestion.ingest_pdf.ingest_pdf", side_effect=side_effect):
        result = ingest_pdfs(tmp_path)

    assert "good.pdf" in result
    assert "bad.pdf" not in result


def test_flatten_metadata_keeps_primitives():
    """flatten_metadata should keep primitive values as-is."""
    meta = {"title": "Test", "page": 1, "flag": True, "score": 0.9, "empty": None}
    result = flatten_metadata(meta)
    assert result == meta


def test_flatten_metadata_serializes_complex_values():
    """flatten_metadata should JSON-serialize non-primitive values."""
    meta = {"tags": ["a", "b"], "nested": {"x": 1}}
    result = flatten_metadata(meta)
    assert result["tags"] == json.dumps(["a", "b"])
    assert result["nested"] == json.dumps({"x": 1})
