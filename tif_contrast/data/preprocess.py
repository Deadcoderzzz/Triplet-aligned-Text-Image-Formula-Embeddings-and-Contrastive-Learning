import os
import argparse
from pathlib import Path
import fitz  
from PIL import Image
from paddleocr import PaddleOCR
from tqdm import tqdm
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess academic PDFs')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing PDF files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for processed documents')
    parser.add_argument('--use_paddleocr', action='store_true',
                       help='Use PaddleOCR for formula extraction')
    parser.add_argument('--max_docs', type=int, default=None,
                       help='Maximum number of documents to process')
    
    return parser.parse_args()


class PDFProcessor:
    """Process PDF documents to extract multimodal content"""
    
    def __init__(self, use_ocr=True):
        self.use_ocr = use_ocr
        
        if use_ocr:
            # Initialize PaddleOCR for formula extraction
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=True,
                show_log=False
            )
    
    def extract_text(self, pdf_path):
        """Extract text from PDF"""
        doc = fitz.open(pdf_path)
        
        full_text = []
        
        for page in doc:
            text = page.get_text()
            full_text.append(text)
        
        doc.close()
        
        return "\n\n".join(full_text)
    
    def extract_images(self, pdf_path, output_dir):
        """Extract images from PDF"""
        doc = fitz.open(pdf_path)
        
        image_paths = []
        
        for page_num, page in enumerate(doc):
            # Get images on page
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Save image
                img_filename = f"page_{page_num}_img_{img_index}.png"
                img_path = os.path.join(output_dir, img_filename)
                
                with open(img_path, "wb") as f:
                    f.write(image_bytes)
                
                image_paths.append(img_path)
        
        doc.close()
        
        return image_paths
    
    def extract_formulas(self, pdf_path, output_dir):
        """Extract mathematical formulas using OCR"""
        if not self.use_ocr:
            return []
        
        doc = fitz.open(pdf_path)
        
        formula_data = []
        
        for page_num, page in enumerate(doc):
            # Render page as image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_path = os.path.join(output_dir, f"page_{page_num}_temp.png")
            pix.save(img_path)
            
            # Run OCR
            result = self.ocr.ocr(img_path, cls=True)
            
            if result and result[0]:
                for line in result[0]:
                    text = line[1][0]
                    confidence = line[1][1]
                    
                    # Heuristic: check if it looks like a formula
                    if self.is_formula(text):
                        formula_data.append({
                            'text': text,
                            'confidence': confidence,
                            'page': page_num
                        })
            
            # Clean up temp image
            os.remove(img_path)
        
        doc.close()
        
        return formula_data
    
    @staticmethod
    def is_formula(text):
        """Simple heuristic to detect if text is a mathematical formula"""
        math_symbols = ['=', '+', '-', '*', '/', '^', '∫', '∑', '√', 'α', 'β', 'γ']
        
        # Check for math symbols
        has_math = any(symbol in text for symbol in math_symbols)
        
        # Check for LaTeX-like patterns
        has_latex = '\\' in text or '$' in text
        
        return has_math or has_latex


def process_pdf(pdf_path, output_dir, processor):
    """
    Process a single PDF file
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save extracted content
        processor: PDFProcessor instance
    """
    # Create output directory for this document
    doc_name = Path(pdf_path).stem
    doc_output_dir = os.path.join(output_dir, 'documents', doc_name)
    os.makedirs(doc_output_dir, exist_ok=True)
    
    try:
        # Extract text
        text = processor.extract_text(pdf_path)
        text_path = os.path.join(doc_output_dir, 'text.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Extract images
        images = processor.extract_images(pdf_path, doc_output_dir)
        
        # Select the first/largest image as the main figure
        if images:
            # For simplicity, use the first image
            main_image = images[0]
            os.rename(main_image, os.path.join(doc_output_dir, 'figure.png'))
        else:
            # Create a placeholder if no images
            placeholder = Image.new('RGB', (224, 224), color='white')
            placeholder.save(os.path.join(doc_output_dir, 'figure.png'))
        
        # Extract formulas
        formulas = processor.extract_formulas(pdf_path, doc_output_dir)
        
        # Save formulas
        if formulas:
            # Combine all formulas
            formula_text = "\n".join([f['text'] for f in formulas])
            formula_path = os.path.join(doc_output_dir, 'formula.txt')
            with open(formula_path, 'w', encoding='utf-8') as f:
                f.write(formula_text)
        else:
            # Create empty formula file
            with open(os.path.join(doc_output_dir, 'formula.txt'), 'w') as f:
                f.write("")
        
        return {
            'doc_id': doc_name,
            'status': 'success',
            'text_length': len(text),
            'num_formulas': len(formulas)
        }
    
    except Exception as e:
        return {
            'doc_id': doc_name,
            'status': 'failed',
            'error': str(e)
        }


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'documents'), exist_ok=True)
    
    # Initialize processor
    processor = PDFProcessor(use_ocr=args.use_paddleocr)
    
    # Get all PDF files
    pdf_files = list(Path(args.input_dir).glob('*.pdf'))
    
    if args.max_docs:
        pdf_files = pdf_files[:args.max_docs]
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF
    results = []
    
    for pdf_path in tqdm(pdf_files, desc='Processing PDFs'):
        result = process_pdf(str(pdf_path), args.output_dir, processor)
        results.append(result)
    
    # Save processing summary
    summary = {
        'total_docs': len(pdf_files),
        'successful': sum(1 for r in results if r['status'] == 'success'),
        'failed': sum(1 for r in results if r['status'] == 'failed'),
        'results': results
    }
    
    summary_path = os.path.join(args.output_dir, 'processing_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nProcessing complete!")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Summary saved to {summary_path}")


if __name__ == '__main__':
    main()
