import os
import fitz
from paddleocr import PaddleOCR
from PIL import Image
import io

def preprocess_pdf(pdf_path, output_root):
    doc_id = os.path.basename(pdf_path).replace('.pdf', '')
    save_dir = os.path.join(output_root, doc_id)
    os.makedirs(save_dir, exist_ok=True)
    
    doc = fitz.open(pdf_path)
    ocr = PaddleOCR(use_angle_cls=True, lang='en')

    # 1. Extract Text
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    with open(os.path.join(save_dir, "text.txt"), "w", encoding="utf-8") as f:
        f.write(full_text)

    # 2. Extract Figures
    for page_index in range(len(doc)):
        for img_index, img in enumerate(doc.get_page_images(page_index)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            image.save(os.path.join(save_dir, f"fig_{page_index}_{img_index}.png"))

    print(f"Finished processing {doc_id}")