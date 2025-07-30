import os
import json
from outline_extractor import extract_outline

def test_outline_extractor():
    sample_pdf = os.path.join(os.path.dirname(__file__), '../input/sample.pdf')
    if not os.path.exists(sample_pdf):
        print('Sample PDF not found, skipping test.')
        return
    outline = extract_outline(sample_pdf)
    assert 'title' in outline
    assert 'outline' in outline
    assert isinstance(outline['outline'], list)
    print('Test passed: Outline structure is valid.')

if __name__ == '__main__':
    test_outline_extractor() 