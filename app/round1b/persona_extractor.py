import json
import os
import re
from collections import defaultdict
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class PersonaExtractor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
    def extract_relevant_sections(self, documents: List[str], persona: str, job: str) -> Dict[str, Any]:
        """
        Extract relevant sections from documents based on persona and job-to-be-done.
        
        Args:
            documents: List of PDF file paths
            persona: Description of the persona (e.g., "PhD Researcher in Computational Biology")
            job: Job-to-be-done (e.g., "Prepare literature review")
            
        Returns:
            Dictionary with metadata, extracted sections, and sub-section analysis
        """
        
        # Load outlines from Round 1A
        outlines = []
        output_dir = "/app/output"
        
        for doc_path in documents:
            doc_name = os.path.basename(doc_path)
            outline_path = os.path.join(output_dir, doc_name.replace('.pdf', '.json'))
            print(f"Looking for outline: {outline_path}")
            
            if os.path.exists(outline_path):
                print(f"Found outline: {outline_path}")
                with open(outline_path, 'r', encoding='utf-8') as f:
                    outlines.append(json.load(f))
            else:
                print(f"Outline not found: {outline_path}")
        
        # Extract all sections from outlines
        all_sections = []
        for i, outline in enumerate(outlines):
            doc_name = os.path.basename(documents[i])
            sections = self._extract_sections_from_outline(outline, doc_name)
            all_sections.extend(sections)
        
        print(f"Total sections extracted: {len(all_sections)}")
        
        if not all_sections:
            print("No sections found in outlines. Creating basic output.")
            return self._create_basic_output(documents, persona, job)
        
        # Calculate relevance scores
        relevance_scores = self._calculate_relevance(all_sections, persona, job)
        
        # Rank sections by relevance
        ranked_sections = sorted(relevance_scores, key=lambda x: x['importance_rank'], reverse=True)
        
        # Get top relevant sections
        top_sections = ranked_sections[:10]  # Top 10 most relevant
        
        # Generate sub-section analysis
        sub_sections = self._analyze_sub_sections(top_sections, persona, job)
        
        # Create output
        output = {
            "metadata": {
                "input_documents": [os.path.basename(doc) for doc in documents],
                "persona": persona,
                "job_to_be_done": job,
                "processing_timestamp": self._get_timestamp()
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
            "sub_section_analysis": sub_sections
        }
        
        return output
    
    def _create_basic_output(self, documents: List[str], persona: str, job: str) -> Dict[str, Any]:
        """Create basic output when no sections are found"""
        return {
            "metadata": {
                "input_documents": [os.path.basename(doc) for doc in documents],
                "persona": persona,
                "job_to_be_done": job,
                "processing_timestamp": self._get_timestamp(),
                "note": "No sections found in outlines"
            },
            "extracted_sections": [],
            "sub_section_analysis": []
        }
    
    def _extract_sections_from_outline(self, outline: Dict, doc_name: str) -> List[Dict]:
        """Extract all sections from an outline"""
        sections = []
        
        def extract_recursive(nodes, level=0):
            for node in nodes:
                section = {
                    'document': doc_name,
                    'page': node.get('page', 1),
                    'text': node.get('text', ''),
                    'level': node.get('level', 'H1'),
                    'children': node.get('children', [])
                }
                sections.append(section)
                if node.get('children'):
                    extract_recursive(node['children'], level + 1)
        
        extract_recursive(outline.get('outline', []))
        return sections
    
    def _calculate_relevance(self, sections: List[Dict], persona: str, job: str) -> List[Dict]:
        """Calculate relevance scores for sections"""
        
        # Combine persona and job for query
        query = f"{persona} {job}"
        
        # Extract text from sections
        texts = [section['text'] for section in sections if section['text'].strip()]
        
        if not texts:
            # If no text, use simple scoring
            for section in sections:
                section['importance_rank'] = 0.1
            return sections
        
        # Add query to texts for vectorization
        texts.append(query)
        
        # Vectorize texts
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            query_vector = tfidf_matrix[-1]  # Last vector is the query
            section_vectors = tfidf_matrix[:-1]  # All other vectors are sections
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, section_vectors).flatten()
            
            # Add relevance scores to sections
            for i, section in enumerate(sections):
                if i < len(similarities):
                    section['importance_rank'] = float(similarities[i])
                else:
                    section['importance_rank'] = 0.1
                
        except Exception as e:
            # Fallback: simple keyword matching
            print(f"TF-IDF failed, using keyword matching: {e}")
            for section in sections:
                section['importance_rank'] = self._simple_keyword_score(section['text'], query)
        
        return sections
    
    def _simple_keyword_score(self, text: str, query: str) -> float:
        """Simple keyword-based relevance scoring"""
        text_lower = text.lower()
        query_lower = query.lower()
        
        # Count matching words
        query_words = set(query_lower.split())
        text_words = set(text_lower.split())
        
        matches = len(query_words.intersection(text_words))
        return matches / max(len(query_words), 1)
    
    def _analyze_sub_sections(self, sections: List[Dict], persona: str, job: str) -> List[Dict]:
        """Analyze sub-sections for detailed insights"""
        sub_analysis = []
        
        for section in sections[:5]:  # Top 5 sections
            # Extract key insights from section text
            insights = self._extract_insights(section['text'], persona, job)
            
            sub_analysis.append({
                "document": section['document'],
                "section_title": section['text'],
                "refined_text": insights,
                "page_number": section['page']
            })
        
        return sub_analysis
    
    def _extract_insights(self, text: str, persona: str, job: str) -> str:
        """Extract key insights from section text"""
        # Simple insight extraction based on keywords
        insights = []
        
        # Look for key phrases
        key_phrases = [
            "method", "approach", "technique", "algorithm",
            "result", "finding", "conclusion", "summary",
            "analysis", "evaluation", "comparison", "review"
        ]
        
        text_lower = text.lower()
        for phrase in key_phrases:
            if phrase in text_lower:
                insights.append(f"Contains {phrase}")
        
        if not insights:
            insights.append("General content section")
        
        return "; ".join(insights)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

def extract_relevant_sections(documents, persona, job):
    """Main function for Round 1B"""
    extractor = PersonaExtractor()
    return extractor.extract_relevant_sections(documents, persona, job) 