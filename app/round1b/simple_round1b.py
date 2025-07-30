import json
import os
from datetime import datetime

def run_simple_round1b():
    """Simple Round 1B implementation"""
    
    # Configuration
    output_dir = "/app/output"
    
    # Example persona and job
    persona = "PhD Researcher in Cybersecurity"
    job = "Analyze network security methodologies and threat detection techniques"
    
    # Load outline files
    outline_files = [f for f in os.listdir(output_dir) if f.endswith('.json') and not f.endswith('_debug.json') and not f.endswith('round1b_output.json')]
    
    all_sections = []
    
    for outline_file in outline_files:
        doc_name = outline_file.replace('.json', '')
        outline_path = os.path.join(output_dir, outline_file)
        
        with open(outline_path, 'r', encoding='utf-8') as f:
            outline = json.load(f)
        
        # Extract sections
        def extract_sections(nodes, level=0):
            for node in nodes:
                section = {
                    'document': doc_name,
                    'page': node.get('page', 1),
                    'text': node.get('text', ''),
                    'level': node.get('level', 'H1')
                }
                all_sections.append(section)
                if node.get('children'):
                    extract_sections(node['children'], level + 1)
        
        extract_sections(outline.get('outline', []))
    
    print(f"Found {len(all_sections)} sections from {len(outline_files)} documents")
    
    # Simple relevance scoring based on keywords
    cybersecurity_keywords = ['security', 'network', 'cyber', 'threat', 'attack', 'vulnerability', 'encryption', 'firewall', 'intrusion', 'malware']
    
    scored_sections = []
    for section in all_sections:
        text_lower = section['text'].lower()
        score = sum(1 for keyword in cybersecurity_keywords if keyword in text_lower)
        section['importance_rank'] = score / len(cybersecurity_keywords)
        scored_sections.append(section)
    
    # Sort by relevance
    scored_sections.sort(key=lambda x: x['importance_rank'], reverse=True)
    
    # Take top 10
    top_sections = scored_sections[:10]
    
    # Create output
    output = {
        "metadata": {
            "input_documents": [f.replace('.json', '') for f in outline_files],
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [
            {
                "document": section['document'],
                "page_number": section['page'],
                "section_title": section['text'],
                "importance_rank": section['importance_rank']
            }
            for section in top_sections
        ],
        "sub_section_analysis": [
            {
                "document": section['document'],
                "section_title": section['text'],
                "refined_text": f"Cybersecurity relevance score: {section['importance_rank']:.2f}",
                "page_number": section['page']
            }
            for section in top_sections[:5]
        ]
    }
    
    # Save output
    output_file = os.path.join(output_dir, "round1b_output.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Round 1B analysis completed!")
    print(f"✓ Output saved to: {output_file}")
    print(f"✓ Found {len(top_sections)} relevant sections")
    
    if top_sections:
        print(f"✓ Top section: {top_sections[0]['text']} (score: {top_sections[0]['importance_rank']:.2f})")

if __name__ == '__main__':
    run_simple_round1b() 