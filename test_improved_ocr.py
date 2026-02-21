import os
os.environ['PATH'] = r'C:\Program Files\Tesseract-OCR' + os.pathsep + os.environ.get('PATH', '')

import pytesseract
pytesseract.pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

import fitz
from PIL import Image, ImageEnhance
import io

print("Testing IMPROVED OCR on temp.pdf...")
print("=" * 60)
print("Improvements:")
print("  - 3x zoom (was 2x)")
print("  - 1.8x contrast enhancement")
print("  - English-only (Arabic data not available)")
print("  - Better PSM settings")
print("=" * 60 + "\n")

try:
    pdf = fitz.open('temp.pdf')
    print(f"PDF: {len(pdf)} pages\n")
    
    total_text = ""
    
    # Test first 3pages
    for page_num in range(min(3, len(pdf))):
        print(f"Page {page_num + 1}:")
        page = pdf[page_num]
        
        # 3x zoom
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
        img_data = pix.tobytes("ppm")
        img = Image.open(io.BytesIO(img_data))
        print(f"  - Image: {pix.width}x{pix.height}")
        
        # Enhance contrast
        contrast = ImageEnhance.Contrast(img)
        img = contrast.enhance(1.8)
        print(f"  - Contrast enhanced 1.8x")
        
        # OCR with better settings
        text = pytesseract.image_to_string(img, lang='eng', config='--psm 6')
        
        char_count = len(text.strip())
        total_text += text
        print(f"  - Extracted: {char_count} chars")
        
        if char_count > 0:
            sample = text[:80].replace('\n', ' ')
            print(f"  - Sample: {sample}...")
        print()
    
    pdf.close()
    
    print("=" * 60)
    total = len(total_text.strip())
    if total > 0:
        print(f"✅ IMPROVED OCR: {total} chars extracted")
        print(f"\n📄 First 200 chars:")
        print(total_text[:200])
    else:
        print(f"❌ Still 0 chars - check if pages are blank or unreadable")
        
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
