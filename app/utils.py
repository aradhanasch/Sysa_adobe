import re

def normalize_text(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def is_heading_numbering(text):
    # Matches patterns like 1., 1.1, I., A., etc.
    return bool(re.match(r'^(\d+\.|[A-Z]\.|[IVX]+\.)', text.strip()))

def is_title_case(text):
    words = text.split()
    if not words:
        return False
    return sum(w[0].isupper() for w in words if w[0].isalpha()) / len(words) > 0.7

def is_all_caps(text):
    return text.isupper() and len(text) > 2

def get_language(text):
    # Stub for future language detection
    return 'unknown' 