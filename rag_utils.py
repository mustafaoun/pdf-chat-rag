from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import pdfplumber
import os
import subprocess
import io
import fitz  # PyMuPDF
from PIL import Image

# Configure poppler path
poppler_path = r"D:\One Drive\OneDrive\سطح المكتب\Release-25.12.0-0\poppler-25.12.0\Library\bin"
if os.path.exists(poppler_path):
    os.environ['PATH'] = poppler_path + os.pathsep + os.environ['PATH']
    print(f"✅ Poppler configured at: {poppler_path}")
else:
    print(f"⚠️ Poppler path not found: {poppler_path}")

try:
    import pytesseract
    pytesseract.pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    # Verify Tesseract is accessible
    try:
        result = subprocess.run([r'C:\Program Files\Tesseract-OCR\tesseract.exe', '--version'], 
                       capture_output=True, timeout=5, text=True)
        if result.returncode == 0:
            print("✅ Tesseract OCR is ready")
            OCR_AVAILABLE = True
        else:
            print("⚠️ Tesseract not responding properly")
            OCR_AVAILABLE = False
    except Exception as e:
        print(f"❌ Tesseract not accessible: {str(e)}")
        OCR_AVAILABLE = False
except ImportError as e:
    print(f"⚠️ Tesseract import failed: {str(e)}")
    OCR_AVAILABLE = False
except Exception as e:
    print(f"⚠️ Error loading Tesseract: {str(e)}")
    OCR_AVAILABLE = False


def extract_text_with_ocr_pymupdf(file_path: str) -> str:
    """
    استخدم OCR لاستخراج النص من الصور في PDF باستخدام PyMuPDF
    هذه الطريقة أكثر استقراراً ولا تحتاج poppler
    """
    if not OCR_AVAILABLE:
        print("⚠️ OCR not available - Tesseract not configured")
        return ""
    
    try:
        print(f"🔍 Starting OCR with PyMuPDF from {file_path}...")
        pdf_doc = fitz.open(file_path)
        total_pages = len(pdf_doc)
        print(f"📊 PDF has {total_pages} page(s)")
        
        extracted_text = ""
        
        for page_num in range(total_pages):
            print(f"📄 Processing page {page_num + 1}/{total_pages}...")
            page = pdf_doc[page_num]
            
            # Convert page to image using PyMuPDF
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)  # 2x zoom for better OCR
            img_data = pix.tobytes("ppm")
            img = Image.open(io.BytesIO(img_data))
            
            # Run OCR
            text = pytesseract.image_to_string(img, lang='ara+eng')
            
            if text.strip():
                extracted_text += f"\n--- Page {page_num + 1} ---\n{text}"
                print(f"✅ Extracted {len(text)} characters from page {page_num + 1}")
            else:
                print(f"⚠️ No text found on page {page_num + 1}")
        
        pdf_doc.close()
        
        result = extracted_text.strip()
        print(f"📊 Total OCR text: {len(result)} characters")
        return result
        
    except Exception as e:
        print(f"❌ PyMuPDF OCR failed: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""


def extract_text_with_ocr_pdf2image(file_path: str) -> str:
    """
    استخدم pdf2image مع poppler كبديل للـ PyMuPDF
    """
    if not OCR_AVAILABLE:
        return ""
    
    try:
        from pdf2image import convert_from_path

        print(f"🔍 Trying OCR with pdf2image from {file_path}...")
        # Pass explicit poppler_path on Windows so pdfinfo/pdftoppm can be found
        if os.path.exists(poppler_path):
            images = convert_from_path(file_path, dpi=200, poppler_path=poppler_path)
        else:
            images = convert_from_path(file_path, dpi=200)
        print(f"✅ Converted PDF to {len(images)} image(s)")
        
        extracted_text = ""
        for idx, image in enumerate(images):
            print(f"📄 Processing page {idx + 1}/{len(images)} with Tesseract...")
            text = pytesseract.image_to_string(image, lang='ara+eng')
            if text.strip():
                extracted_text += f"\n--- Page {idx + 1} ---\n{text}"
                print(f"✅ Extracted {len(text)} characters from page {idx + 1}")
            else:
                print(f"⚠️ No text found on page {idx + 1}")
        
        result = extracted_text.strip()
        print(f"📊 Total pdf2image OCR text: {len(result)} characters")
        return result
        
    except Exception as e:
        print(f"❌ pdf2image OCR failed: {type(e).__name__}: {str(e)}")
        return ""


def process_pdf_to_vectorstore(file_path):
    """
    هذه الوظيفة تأخذ ملف PDF وتحوله إلى قاعدة بيانات متجهة (Vector Store)
    تحاول أولاً استخراج النص المنسق، ثم تحاول OCR إذا فشل الاستخراج الأول
    """
    documents = []
    total_text = ""
    
    # 1. محاولة أولى: استخدام pdfplumber (أفضل من PyPDFLoader)
    try:
        with pdfplumber.open(file_path) as pdf:
            for idx, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": file_path, "page": idx}
                    ))
                    total_text += text
    except Exception as e:
        print(f"⚠️ pdfplumber extraction failed: {str(e)}")
    
    # 2. إذا لم نجد نص، نحاول PyPDFLoader كخيار بديل
    if not total_text:
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            total_text = "".join([doc.page_content for doc in documents])
        except Exception as e:
            print(f"⚠️ PyPDFLoader extraction failed: {str(e)}")
    
    # 3. إذا لم نجد نص، نحاول OCR مع PyMuPDF أولاً، ثم pdf2image
    if not total_text.strip() and OCR_AVAILABLE:
        print("🔍 No selectable text found, attempting OCR...")
        
        # Try PyMuPDF first (more stable)
        ocr_text = extract_text_with_ocr_pymupdf(file_path)
        
        # If PyMuPDF fails, try pdf2image with poppler
        if not ocr_text:
            print("🔄 PyMuPDF failed, trying pdf2image with poppler...")
            ocr_text = extract_text_with_ocr_pdf2image(file_path)
        
        if ocr_text:
            documents = [Document(
                page_content=ocr_text,
                metadata={"source": file_path, "method": "OCR"}
            )]
            total_text = ocr_text
    
    # التحقق من أن الـ PDF يحتوي على نص
    if not documents:
        raise ValueError("❌ Failed to extract text from PDF. File may be empty or corrupted.")
    
    total_text = total_text.strip()
    if not total_text:
        if OCR_AVAILABLE:
            raise ValueError("❌ No readable text found in PDF. Tried text extraction and OCR. File may be password-protected or corrupted.")
        else:
            raise ValueError("❌ No readable text found in PDF. Install OCR support: pip install pytesseract pdf2image")
    
    # 2. تقسيم النص لقطع صغيرة (Chunks) لضمان دقة البحث
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)
    
    # التحقق من أن التقسيم أنتج chunks
    if not docs:
        raise ValueError("❌ Failed to split documents into chunks. Text may be too short or malformed.")
    
    # 3. اختيار موديل Embeddings مجاني (HuggingFace)
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        raise ValueError(f"❌ Failed to load embeddings model: {str(e)}")
    
    # 4. بناء قاعدة البيانات المتجهة (في الذاكرة حالياً)
    try:
        vectorstore = Chroma.from_documents(docs, embeddings)
    except ValueError as e:
        raise ValueError(f"❌ Failed to create vector store: {str(e)}. Try a different PDF.")
    
    return vectorstore
