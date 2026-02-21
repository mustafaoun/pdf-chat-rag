import pytesseract
import fitz
from PIL import Image
import io

pytesseract.pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

print("Testing OCR on temp.pdf...")
print("=" * 50)

try:
    pdf = fitz.open('temp.pdf')
    print(f"✅ Opened: {len(pdf)} pages")
    
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
            print(f"  - First 50 chars: {text[:50]}")
        else:
            print(f"  - No text extracted (blank page or unreadable image)")
    
    pdf.close()
    
    print("\n" + "=" * 50)
    if total_text.strip():
        print(f"✅ SUCCESS: Extracted {len(total_text)} chars from PDF")
        print(f"📄 OCR IS WORKING!")
    else:
        print(f"❌ PROBLEM: OCR extracted 0 chars")
        print(f"The PDF pages might be:")
        print(f"  - Completely blank/white pages")
        print(f"  - Very poor quality/corrupted scans")
        print(f"  - Text in a language tesseract can't recognize")
        
except Exception as e:
    print(f"❌ ERROR: {type(e).__name__}")
    print(f"   {e}")
    import traceback
    traceback.print_exc()
