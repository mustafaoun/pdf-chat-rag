## PDF Chat RAG: Local-First Document Intelligence

Most RAG tutorials assume your data is perfectly clean. Real-world documents aren't. They are messy, scanned, or image-only PDFs that standard loaders can't handle. 

**PDF Chat RAG** was built to bridge that gap. It is a high-performance, private-by-design application that allows you to have sub-second conversations with any PDF—whether it’s a digital native or a scanned photocopy—without your data ever leaving your device for embedding or processing.

---

### Key Features

* **Privacy-First Architecture:** Uses local embeddings via HuggingFace. Your sensitive documents stay on your machine.
* **Intelligent OCR Fallback:** Integrated with **Tesseract OCR** to handle scanned pages and image-heavy PDFs that break traditional parsers.
* **Sub-Second Inference:** Leverages **Groq API with Llama 3.3 70b** for lightning-fast responses without the high costs of OpenAI.
* **Grounded Accuracy:** Optimized with a low temperature (0.2) to ensure the model stays faithful to the document and avoids hallucinations.
* **Streamlit Interface:** A clean, intuitive UI for seamless document uploading and chatting.

---

### Technical Decisions

| Component | Choice | Why? |
| :--- | :--- | :--- |
| **Orchestration** | LangChain (LCEL) | Modular, readable, and easy to extend the RAG chain. |
| **Embeddings** | HuggingFace (Local) | Eliminates the privacy risk of sending raw text to external APIs. |
| **LLM** | Llama 3.3 70b (via Groq) | Speed. Getting 70b-tier performance at sub-second speeds is a game changer. |
| **OCR Engine** | Tesseract | The "unglamorous" necessity for handling messy, real-world documents. |
| **Vector Store** | FAISS | Lightweight and efficient for local document indexing. |

---

### Project Structure

```text
├── OCR/               # Logic for Tesseract integration and image processing
├── RAG/               # Core LangChain logic and vector store management
├── assets/            # Screenshots and project visuals
├── app.py             # Main Streamlit application entry point
├── requirements.txt   # Project dependencies
└── .env.example       # Template for environment variables (Groq API Key)
```

---

### Getting Started

#### 1. Prerequisites
* Python 3.9+
* **Tesseract OCR** installed on your system.

#### 2. Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/mustafaoun/pdf-chat-rag.git
   cd pdf-chat-rag
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your environment variables:
   * Rename `.env.example` to `.env`.
   * Add your `GROQ_API_KEY`.

#### 3. Run the App
```bash
streamlit run app.py
```

---

###  The Lesson
The biggest takeaway from this project wasn't the "intelligent" part—it was the engineering required to handle the edge cases. Handling scanned PDFs and ensuring data privacy transformed a simple chatbot into a robust tool.

**What about you?** If you've built RAG pipelines, do you prefer local embeddings for privacy or cloud-based solutions for scalability? Let's discuss in the issues!

---
