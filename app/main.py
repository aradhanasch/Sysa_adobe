import os
import json
from outline_extractor import extract_outline

INPUT_DIR = '/app/input'
OUTPUT_DIR = '/app/output'


def process_all_pdfs(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
    """Process all PDFs in the input directory"""
    pdfs = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    if not pdfs:
        print("No PDF files found in input directory")
        return
    
    print(f"Found {len(pdfs)} PDF files to process")
    
    for filename in pdfs:
        pdf_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename[:-4] + '.json')
        
        try:
            print(f"Processing {filename}...")
            outline = extract_outline(pdf_path, debug=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(outline, f, ensure_ascii=False, indent=2)
            
            print(f"✓ Successfully processed {filename} -> {output_path}")
            
        except Exception as e:
            print(f"✗ Failed to process {filename}: {e}")
            # Create a basic error output
            error_output = {
                "title": "",
                "outline": [],
                "error": str(e)
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(error_output, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    process_all_pdfs() 