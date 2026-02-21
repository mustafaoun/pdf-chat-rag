# PDF Chat RAG

A local-first Retrieval-Augmented Generation (RAG) app that lets you upload a PDF and chat with its content. It supports both text-based PDFs and scanned images (via OCR) and uses a vector store + LLM chain to produce grounded answers.

---

## Demo
Run locally and open http://localhost:8501 after following setup.

## Features
- Upload any PDF and ask questions about its contents
- Automatic text extraction (pdfplumber / PyPDFLoader)
- OCR fallback for scanned PDFs (PyMuPDF + Tesseract, pdf2image + Poppler fallback)
- Local embeddings (HuggingFace `all-MiniLM-L6-v2`) and Chroma vector store
- LangChain LCEL pipeline and Groq `ChatGroq` LLM integration
- Session chat history via Streamlit

## Quick setup (local)
1. Clone the repo:

```bash
git clone https://github.com/mustafaoun/pdf-chat-rag.git
cd pdf-chat-rag
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows
.\\.venv\\Scripts\\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and add your API key:

```bash
copy .env.example .env
# Edit .env and set GROQ_API_KEY=your_key_here
```

> Important: Never commit `.env` to the repo. `.gitignore` already ignores it.

4. (Optional) For OCR of scanned Arabic PDFs install Tesseract (and Arabic language data):

- Windows: install Tesseract and add to PATH (`C:\\Program Files\\Tesseract-OCR`)
- Download `ara.traineddata` into `C:\\Program Files\\Tesseract-OCR\\tessdata\\` to enable Arabic OCR

If you prefer not to install Tesseract, use Google Drive → Open with Google Docs → Download as PDF (Google OCR) to convert scanned PDFs into text-based PDFs.

5. Run app:

```bash
streamlit run app.py
```

Open `http://localhost:8501` and upload a PDF.

## Hiding your API key (recommended)
- Use the `.env` method for local development (already supported).
- For the GitHub repository, do NOT commit `.env`. Instead:
  - Add a `.env.example` with the placeholder (this repo includes it).
  - When using GitHub Actions or any CI, set `GROQ_API_KEY` as a repository secret and reference it in workflows.

## How to publish to GitHub (safe)
1. Initialize git (if not already):

```bash
git init
git add .
git commit -m "Initial commit"
```

2. Add remote and push (only after confirming `.gitignore` and `.env`):

```bash
git branch -M main
git remote add origin https://github.com/mustafaoun/pdf-chat-rag.git
git push -u origin main
```

If `temp.pdf` or other sensitive files were accidentally committed, remove them from history:

```bash
git rm --cached temp.pdf
git commit -m "Remove temp PDF"
# For full history rewrite (use with caution):
# git filter-branch --index-filter 'git rm --cached --ignore-unmatch temp.pdf' -- --all
```

## Make this repo popular (tips)
- Write a clear, attractive `README.md` with screenshots or GIFs.
- Add a short demo GIF of the app in use (upload to README via GitHub). Use `screencastify` or `kap`.
- Add topics/tags on GitHub: `pdf`, `rAG`, `streamlit`, `ocr`, `langchain`.
- Create a short tutorial or blog post and share on Twitter, Reddit (r/MachineLearning, r/LanguageTechnology), Hacker News.
- Add a LICENSE (MIT or Apache-2.0) so others can use it freely.
- Add a minimal GitHub Action to run basic lint/tests and show CI passing.

## Security notes
- Do not commit API keys or private data.
- Use GitHub secrets to store `GROQ_API_KEY` for workflows.

## License
Add a license file if you want to make this repo open-source-friendly (MIT recommended).

---

If you want, I can:
- Create a `README` demo GIF and add it to the repo
- Create a simple GitHub Actions workflow (CI) that runs a small smoke test
- Help you run the `git` commands here (I won't push keys) or prepare a deployable release
