# PDF Chat with OCR fallback

A Streamlit PDF question-answering app with local text extraction, local sentence-transformer embeddings, Chroma retrieval, and Groq-hosted answer generation.

## Data flow and privacy

The PDF is extracted locally. Pages without extractable text fall back to local Tesseract OCR, including scanned pages inside otherwise text-based PDFs. Chunks and embeddings are indexed in a per-document Chroma collection.

**Retrieved text excerpts and your question are sent to Groq.** This is not fully offline or suitable for confidential documents without reviewing the provider's terms and your data policy. Model weights may download from Hugging Face on first use. The original PDF is temporarily written locally and deleted after indexing, including when processing fails. The index and chat remain available during the Streamlit session; they are not secure-erasure guarantees.

## Run locally

Use Python 3.10 or newer:

```bash
git clone https://github.com/mustafaoun/pdf-chat-rag.git
cd pdf-chat-rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `GROQ_API_KEY` in `.env`, then run:

```bash
streamlit run app.py
```

On Windows activate with `.venv\Scripts\activate`. Install [Tesseract](https://tesseract-ocr.github.io/tessdoc/Installation.html) for scanned PDFs. It is discovered through PATH or optional `TESSERACT_CMD`. Set `OCR_LANGUAGE=eng` or another installed language, such as `eng+ara`. Poppler and a personal machine path are not needed by the extraction path.

`GROQ_MODEL` defaults to `llama-3.3-70b-versatile`; set it to a model supported by your account if that model is unavailable. First-time embedding downloads and OCR can be slow; no latency benchmark is claimed.

Upload one PDF, wait for indexing, then ask questions. Identical uploads reuse the current session's index across reruns. Replacing/removing a document clears its chat and attempts to delete its previous collection. Unique collection names avoid reusing a shared default document collection.

## Structure

- `app.py`: Streamlit UI and Groq retrieval/generation chain.
- `RAG/rag_utils_ocr.py`: page-aware extraction, OCR, chunking, local embeddings, Chroma.
- `RAG/session_utils.py`: upload hashing and temporary-file lifecycle.
- `tests/test_processing.py`: extraction and resource-cleanup regression tests.

## Test extraction without a hosted model

```bash
pip install -r requirements-dev.txt
pytest -q
```

The tests cover native-text PDFs, mixed-page OCR dispatch, unique temporary files, cleanup after failure, and Tesseract configuration. They mock OCR recognition and do not call Groq or download embedding weights. Run a real scanned PDF and a hosted-model question manually to validate those external dependencies.

## Limitations

Retrieval can miss relevant passages and answers can be wrong. Low temperature does not guarantee factuality. Treat document instructions as untrusted, verify consequential answers against the original document, and avoid exposing this unauthenticated demo as a multi-user production service.
