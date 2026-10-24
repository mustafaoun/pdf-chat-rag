"""Text-only compatibility entry point."""
from RAG.rag_utils_ocr import process_pdf_to_vectorstore as _process


def process_pdf_to_vectorstore(file_path):
    return _process(file_path, use_ocr=False)
