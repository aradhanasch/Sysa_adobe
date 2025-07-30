import os
import json
from persona_extractor import extract_relevant_sections

def run_round1b():
    """Run Round 1B persona-driven analysis"""
    
    # Configuration
    input_dir = "/app/input"
    output_dir = "/app/output"
    
    # Example persona and job (you can modify these)
    persona = "PhD Researcher in Cybersecurity"
    job = "Analyze network security methodologies and threat detection techniques"
    
    # Get all PDF files
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    pdf_paths = [os.path.join(input_dir, f) for f in pdf_files]
    
    if not pdf_files:
        print("No PDF files found in input directory")
        return
    
    print(f"Processing {len(pdf_files)} documents for persona: {persona}")
    print(f"Job: {job}")
    
    # Run persona-driven analysis
    try:
        result = extract_relevant_sections(pdf_paths, persona, job)
        
        # Save output
        output_file = os.path.join(output_dir, "round1b_output.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Round 1B analysis completed!")
        print(f"✓ Output saved to: {output_file}")
        
        # Print summary
        print(f"\n📊 Summary:")
        print(f"  - Documents processed: {len(pdf_files)}")
        print(f"  - Relevant sections found: {len(result.get('extracted_sections', []))}")
        
        extracted_sections = result.get('extracted_sections', [])
        if extracted_sections:
            print(f"  - Top section: {extracted_sections[0]['section_title']}")
        else:
            print(f"  - No relevant sections found")
        
    except Exception as e:
        print(f"✗ Error in Round 1B: {e}")
        # Create a basic error output
        error_output = {
            "metadata": {
                "input_documents": [os.path.basename(doc) for doc in pdf_paths],
                "persona": persona,
                "job_to_be_done": job,
                "processing_timestamp": "2025-07-28T00:00:00",
                "error": str(e)
            },
            "extracted_sections": [],
            "sub_section_analysis": []
        }
        
        output_file = os.path.join(output_dir, "round1b_output.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(error_output, f, ensure_ascii=False, indent=2)
        print(f"✓ Error output saved to: {output_file}")

if __name__ == '__main__':
    run_round1b() 