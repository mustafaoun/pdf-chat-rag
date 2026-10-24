from pathlib import Path
import fitz
import pytest

from RAG import rag_utils_ocr
from RAG.session_utils import build_uploaded_index, upload_digest


def test_native_text_does_not_require_ocr(tmp_path, monkeypatch):
    source = tmp_path / "native.pdf"
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), "A native PDF with selectable text.")
        doc.save(source)
    monkeypatch.setattr(rag_utils_ocr, "ocr_page", lambda page: pytest.fail("Unexpected OCR"))
    pages = rag_utils_ocr.extract_pages(source)
    assert "selectable text" in pages[0]["text"]
    assert pages[0]["method"] == "native"


def test_mixed_document_uses_ocr_for_missing_page(tmp_path, monkeypatch):
    source = tmp_path / "mixed.pdf"
    with fitz.open() as doc:
        doc.new_page().insert_text((72, 72), "Native page")
        doc.new_page()
        doc.save(source)
    calls = []
    monkeypatch.setattr(rag_utils_ocr, "ocr_page", lambda page: calls.append(page.number) or "Scanned text")
    pages = rag_utils_ocr.extract_pages(source)
    assert calls == [1]
    assert [p["method"] for p in pages] == ["native", "ocr"]


def test_upload_file_removed_even_on_failure():
    paths = []
    def fail(path):
        paths.append(Path(path))
        assert paths[-1].read_bytes() == b"test"
        raise ValueError("Extraction failed")
    with pytest.raises(ValueError):
        build_uploaded_index(b"test", fail)
    assert not paths[0].exists()


def test_uploads_use_unique_paths():
    paths = []
    def read(path):
        paths.append(path)
        return Path(path).read_bytes()
    assert build_uploaded_index(b"a", read) == b"a"
    assert build_uploaded_index(b"b", read) == b"b"
    assert paths[0] != paths[1]
    assert upload_digest(b"a") != upload_digest(b"b")


def test_environment_ocr_override(monkeypatch):
    monkeypatch.setenv("TESSERACT_CMD", "/custom/tesseract")
    assert rag_utils_ocr.tesseract_command() == "/custom/tesseract"
