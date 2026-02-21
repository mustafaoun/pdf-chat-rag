from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import pdfplumber
import os
import subprocess

# Set up OCR - CRITICAL: Add Tesseract to PATH FIRST before importing pytesseract
print("\n" + "="*60)
print("🔧 INITIALIZING OCR SUPPORT")
print("="*60)

TESSERACT_DIR = r'C:\Program Files\Tesseract-OCR'
TESSERACT_EXE = os.path.join(TESSERACT_DIR, 'tesseract.exe')
POPPLER_PATH = r"D:\One Drive\OneDrive\سطح المكتب\Release-25.12.0-0\poppler-25.12.0\Library\bin"

# Add Tesseract to PATH BEFORE importing pytesseract
os.environ['PATH'] = TESSERACT_DIR + os.pathsep + os.environ.get('PATH', '')
print(f"✅ Added Tesseract to PATH")

# NOW import pytesseract
try:
    import pytesseract
    pytesseract.pytesseract.pytesseract_cmd = TESSERACT_EXE
    
    # Verify Tesseract works
    result = subprocess.run([TESSERACT_EXE, '--version'], capture_output=True, timeout=5, text=True)
    if result.returncode == 0:
        print(f"✅ Tesseract verified and working")
        TESSERACT_OK = True
    else:
        print(f"❌ Tesseract verification failed")
        TESSERACT_OK = False
except Exception as e:
    print(f"❌ Tesseract error: {e}")
    TESSERACT_OK = False

# Configure Poppler
if os.path.exists(POPPLER_PATH):
    os.environ['PATH'] = POPPLER_PATH + os.pathsep + os.environ['PATH']
    print(f"✅ Poppler available")
    POPPLER_OK = True
else:
    print(f"⚠️ Poppler not found")
    POPPLER_OK = False

# Try importing pdf2image
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_OK = True
    print("✅ pdf2image available")
except ImportError:
    PDF2IMAGE_OK = False
    print("❌ pdf2image not available")

# Try importing PyMuPDF
try:
    import fitz
    from PIL import Image
    import io
    PYMUPDF_OK = True
    print("✅ PyMuPDF available")
except ImportError:
    PYMUPDF_OK = False
    print("❌ PyMuPDF not available")

print("="*60 + "\n")


def ocr_with_pymupdf(file_path):
    """OCR using PyMuPDF + Tesseract"""
    if not (PYMUPDF_OK and TESSERACT_OK):
        return ""
    
    try:
        print("🔄 Running OCR with PyMuPDF...")
        import fitz
        from PIL import Image
        import io
        
        pdf_doc = fitz.open(file_path)
        total_pages = len(pdf_doc)
        print(f"   📄 {total_pages} pages to process")
        
        extracted_text = ""
        for page_num in range(total_pages):
            page = pdf_doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
            img_data = pix.tobytes("ppm")
            img = Image.open(io.BytesIO(img_data))
            
            # Enhance image contrast for better OCR
            try:
                from PIL import ImageEnhance
                contrast = ImageEnhance.Contrast(img)
                img = contrast.enhance(1.8)
            except:
                pass
            
            # Use English-only OCR (Arabic data not installed)
            text = pytesseract.image_to_string(img, lang='eng', config='--psm 6')
            if text.strip():
                extracted_text += f"\n--- Page {page_num + 1} ---\n{text}"
                print(f"   ✅ Page {page_num + 1}: {len(text)} chars")
            else:
                print(f"   ⚠️  Page {page_num + 1}: No text")
        
        pdf_doc.close()
        return extracted_text.strip()
    except Exception as e:
        print(f"   ❌ PyMuPDF OCR failed: {e}")
        return ""


def ocr_with_pdf2image(file_path):
    """OCR using pdf2image + Tesseract"""
    if not (PDF2IMAGE_OK and POPPLER_OK and TESSERACT_OK):
        return ""
    
    try:
        print("🔄 Running OCR with pdf2image + poppler...")
        from pdf2image import convert_from_path
        
        images = convert_from_path(file_path, dpi=200)
        print(f"   📄 Converted to {len(images)} images")
        
        extracted_text = ""
        for idx, image in enumerate(images):
            # Enhance contrast
            try:
                from PIL import ImageEnhance
                contrast = ImageEnhance.Contrast(image)
                image = contrast.enhance(1.8)
            except:
                pass
            
            # Use English-only OCR with better PSM
            text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
            if text.strip():
                extracted_text += f"\n--- Page {idx + 1} ---\n{text}"
                print(f"   ✅ Page {idx + 1}: {len(text)} chars")
            else:
                print(f"   ⚠️  Page {idx + 1}: No text")
        
        return extracted_text.strip()
    except Exception as e:
        print(f"   ❌ pdf2image OCR failed: {e}")
        return ""


def process_pdf_to_vectorstore(file_path):
    """
    Complete PDF processing with fallback chain
    """
    print(f"\n📥 Processing: {file_path}\n")
    documents = []
    total_text = ""
    extraction_method = None
    
    # ===== STAGE 1: Text Extraction (for native PDFs) =====
    print("STAGE 1: Text-based extraction")
    print("-" * 40)
    
    # Try pdfplumber
    try:
        print("🔄 Trying pdfplumber...")
        with pdfplumber.open(file_path) as pdf:
            num_pages = len(pdf.pages)
            print(f"   📄 {num_pages} pages detected")
            
            for idx, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    print(f"   ✅ Page {idx + 1}: {len(text)} chars extracted")
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": file_path, "page": idx}
                    ))
                    total_text += text
            
            if total_text.strip():
                extraction_method = "pdfplumber"
                print(f"✅ SUCCESS: pdfplumber extracted {len(total_text)} chars\n")
    except Exception as e:
        print(f"   ⚠️  pdfplumber failed: {e}")
    
    # Try PyPDFLoader fallback
    if not total_text.strip():
        try:
            print("🔄 Trying PyPDFLoader...")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            total_text = "".join([doc.page_content for doc in documents])
            
            if total_text.strip():
                extraction_method = "PyPDFLoader"
                print(f"✅ SUCCESS: PyPDFLoader extracted {len(total_text)} chars\n")
            else:
                print("   ⚠️  PyPDFLoader returned empty text")
        except Exception as e:
            print(f"   ⚠️  PyPDFLoader failed: {e}")
    
    # ===== STAGE 2: OCR (for scanned/image PDFs) =====
    if not total_text.strip():
        print("\nSTAGE 2: OCR extraction (image-based)")
        print("-" * 40)
        
        # Try PyMuPDF OCR first
        if PYMUPDF_OK and TESSERACT_OK:
            ocr_text = ocr_with_pymupdf(file_path)
            if ocr_text:
                documents = [Document(
                    page_content=ocr_text,
                    metadata={"source": file_path, "method": "PyMuPDF-OCR"}
                )]
                total_text = ocr_text
                extraction_method = "PyMuPDF-OCR"
                print(f"✅ SUCCESS: PyMuPDF OCR extracted {len(total_text)} chars\n")
        
        # Try pdf2image OCR as fallback
        if not total_text.strip() and PDF2IMAGE_OK and POPPLER_OK and TESSERACT_OK:
            ocr_text = ocr_with_pdf2image(file_path)
            if ocr_text:
                documents = [Document(
                    page_content=ocr_text,
                    metadata={"source": file_path, "method": "pdf2image-OCR"}
                )]
                total_text = ocr_text
                extraction_method = "pdf2image-OCR"
                print(f"✅ SUCCESS: pdf2image OCR extracted {len(total_text)} chars\n")
    
    # ===== FINAL VALIDATION =====
    print("STAGE 3: Validation & Vectorization")
    print("-" * 40)
    
    if not documents or not total_text.strip():
        error_details = f"\nMethod: {extraction_method or 'None'}"
        if not TESSERACT_OK:
            error_details += "\n❌ Tesseract not working"
        if not POPPLER_OK:
            error_details += "\n❌ Poppler not found"
        if not PDF2IMAGE_OK:
            error_details += "\n❌ pdf2image not available"
        if not PYMUPDF_OK:
            error_details += "\n❌ PyMuPDF not available"
        
        raise ValueError(f"❌ EXTRACTION FAILED: Could not extract any text from PDF.{error_details}\n\nPlease try: Google Docs → download as PDF, or use ILovePDF.com for OCR conversion")
    
    # Split into chunks
    print(f"🔄 Chunk splitting ({len(total_text)} chars)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)
    print(f"✅ Created {len(docs)} chunks")
    
    # Create embeddings
    print("🔄 Creating embeddings...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("✅ Embeddings ready")
    except Exception as e:
        raise ValueError(f"❌ Embeddings failed: {str(e)}")
    
    # Create vectorstore
    print("🔄 Building vector store...")
    try:
        vectorstore = Chroma.from_documents(docs, embeddings)
        print(f"✅ Vector store created ({len(docs)} documents)\n")
        return vectorstore
    except Exception as e:
        raise ValueError(f"❌ Vector store failed: {str(e)}")
