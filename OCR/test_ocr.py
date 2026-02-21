import pytesseract
import fitz
from PIL import Image
import io
import os
import sys

# Add Tesseract to PATH
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR'
os.environ['PATH'] = TESSERACT_PATH + os.pathsep + os.environ.get('PATH', '')

# Also set pytesseract path
pytesseract.pytesseract.pytesseract_cmd = os.path.join(TESSERACT_PATH, 'tesseract.exe')

print("Testing OCR on temp.pdf...")
print("=" * 50)
print(f"Tesseract path: {pytesseract.pytesseract.pytesseract_cmd}")
print(f"Tesseract in PATH: {os.path.exists(pytesseract.pytesseract.pytesseract_cmd)}")

try:
    # Verify tesseract can run
    import subprocess
    result = subprocess.run([pytesseract.pytesseract.pytesseract_cmd, '--version'], 
                           capture_output=True, text=True, timeout=5)
    print(f"Tesseract version check: {result.returncode == 0}")
    
    pdf = fitz.open('temp.pdf')
    print(f"\n✅ Opened: {len(pdf)} pages")
    
    print("\nTesting first 3 pages with OCR...")
    total_text = ""
    
    for page_num in range(min(3, len(pdf))):
        print(f"\nPage {page_num + 1}:")
        page = pdf[page_num]
        
        # Convert to image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_data = pix.tobytes("ppm")
        img = Image.open(io.BytesIO(img_data))
        print(f"  - Image size: {pix.width}x{pix.height}")
        
        # Run OCR
        text = pytesseract.image_to_string(img, lang='ara+eng')
        print(f"  - Extracted: {len(text)} chars")
        
        if text.strip():
            total_text += text
            print(f"  - Sample: {text[:50] if len(text) > 50 else text}")
        else:
            print(f"  - No text extracted")
    
    pdf.close()
    
    print("\n" + "=" * 50)
    if total_text.strip():
        print(f"✅ SUCCESS: Extracted {len(total_text)} chars total")
        print(f"✅ OCR IS WORKING!")
    else:
        print(f"❌ PROBLEM: OCR extracted 0 chars from first 3 pages")
        print(f"Your PDF pages might be blank or low quality")
        
except Exception as e:
    print(f"❌ ERROR: {type(e).__name__}")
    print(f"   {e}")
    import traceback
    traceback.print_exc()
