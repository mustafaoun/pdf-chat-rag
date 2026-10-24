"""Portable, page-aware PDF extraction with optional local OCR and embeddings."""
import os
from pathlib import Path
import shutil
import uuid


def tesseract_command():
    configured = os.getenv("TESSERACT_CMD")
    if configured:
        return configured
    found = shutil.which("tesseract")
    if found:
        return found
    windows = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    return str(windows) if windows.is_file() else None


def ocr_page(page):
    import io
    import fitz
    import pytesseract
    from PIL import Image

    command = tesseract_command()
    if not command:
        raise ValueError("Scanned page requires Tesseract. Install it or set TESSERACT_CMD.")
    pytesseract.pytesseract.tesseract_cmd = command
    image = Image.open(io.BytesIO(page.get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")))
    language = os.getenv("OCR_LANGUAGE", "eng")
    try:
        return pytesseract.image_to_string(image, lang=language, config="--psm 6").strip()
    except Exception as exc:
        raise ValueError(f"OCR failed. Check Tesseract and the installed '{language}' language data.") from exc


def extract_pages(file_path, use_ocr=True):
    """OCR only pages lacking text, including scanned pages in mixed PDFs."""
    import fitz

    pages = []
    with fitz.open(file_path) as document:
        for number, page in enumerate(document):
            text = page.get_text("text").strip()
            method = "native"
            if not text and use_ocr:
                text = ocr_page(page)
                method = "ocr"
            if text:
                pages.append({"text": text, "page": number + 1, "method": method})
    if not pages:
        raise ValueError("EXTRACTION FAILED: PDF has no readable text.")
    return pages


def process_pdf_to_vectorstore(file_path, use_ocr=True):
    # Defer model/framework imports so local extraction can be tested offline.
    from langchain_core.documents import Document
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    pages = extract_pages(file_path, use_ocr=use_ocr)
    documents = [Document(page_content=p["text"], metadata={
        "page": p["page"], "method": p["method"]
    }) for p in pages]
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Unique collections prevent one upload/session from retrieving another's pages.
    return Chroma.from_documents(chunks, embeddings, collection_name="pdf-" + uuid.uuid4().hex)
