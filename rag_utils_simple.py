from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import pdfplumber
import os

def process_pdf_to_vectorstore(file_path):
    """
    Simple PDF processing without OCR complexity
    """
    print(f"\n📄 Starting PDF processing: {file_path}")
    documents = []
    total_text = ""
    
    # Method 1: Try pdfplumber (best for regular PDFs)
    print("🔄 Attempting text extraction with pdfplumber...")
    try:
        with pdfplumber.open(file_path) as pdf:
            num_pages = len(pdf.pages)
            print(f"📊 PDF has {num_pages} pages")
            
            for idx, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    print(f"✅ Page {idx + 1}: Extracted {len(text)} characters")
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": file_path, "page": idx, "method": "pdfplumber"}
                    ))
                    total_text += text
                else:
                    print(f"⚠️ Page {idx + 1}: Empty or no text")
    except Exception as e:
        print(f"❌ pdfplumber failed: {str(e)}")
    
    # Method 2: Try PyPDFLoader as fallback
    if not total_text.strip():
        print("🔄 Fallback: Attempting with PyPDFLoader...")
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            total_text = "".join([doc.page_content for doc in documents])
            if total_text.strip():
                print(f"✅ PyPDFLoader extracted {len(total_text)} characters")
            else:
                print("⚠️ PyPDFLoader also returned empty text")
        except Exception as e:
            print(f"❌ PyPDFLoader failed: {str(e)}")
    
    # Final check
    if not documents:
        raise ValueError("❌ FAILED: Could not extract any text from PDF using text-based methods (pdfplumber, PyPDFLoader). PDF might be image-only or encrypted.")
    
    total_text = total_text.strip()
    if not total_text:
        raise ValueError("❌ FAILED: PDF loaded but contains no readable text. File may be: image-only, password-protected, or corrupted/blank.")
    
    print(f"\n✅ Successfully extracted {len(total_text)} total characters from PDF")
    print(f"📊 Split into chunks with size=1000, overlap=200...")
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)
    
    if not docs:
        raise ValueError("❌ Failed to split documents into chunks.")
    
    print(f"✅ Created {len(docs)} chunks for vector store")
    print(f"🔄 Creating embeddings with HuggingFace model...")
    
    # Create embeddings
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("✅ Embeddings model loaded")
    except Exception as e:
        raise ValueError(f"❌ Failed to load embeddings model: {str(e)}")
    
    # Create vector store
    try:
        print(f"🔄 Building vector store with {len(docs)} documents...")
        vectorstore = Chroma.from_documents(docs, embeddings)
        print("✅ Vector store created successfully!")
    except Exception as e:
        raise ValueError(f"❌ Failed to create vector store: {str(e)}")
    
    return vectorstore
