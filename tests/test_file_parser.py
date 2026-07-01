"""Integration tests for file parsing — requires sample fixtures."""
import os
import pytest
from src.file_parser import parse_file

FIXTURE_PDF = os.path.join(os.path.dirname(__file__), "fixtures", "sample.pdf")


@pytest.mark.skipif(not os.path.exists(FIXTURE_PDF), reason="No fixture PDF found")
def test_parse_pdf_returns_docs():
    docs = parse_file(FIXTURE_PDF, images_dir=os.path.dirname(FIXTURE_PDF))
    assert len(docs) > 0
    assert all("text" in d and "source" in d for d in docs)
    assert all(len(d["text"]) > 0 for d in docs)
