import fitz  # PyMuPDF
import re
import os
from collections import Counter, defaultdict
from utils import normalize_text, is_heading_numbering, is_title_case, is_all_caps, get_language
import json

# Optional: ML classifier (if present)
try:
    import joblib
    clf = joblib.load('/app/models/heading_classifier.joblib')
    ML_AVAILABLE = True
except Exception:
    clf = None
    ML_AVAILABLE = False

def extract_outline(pdf_path, debug=False):
    doc = fitz.open(pdf_path)
    outline = []
    title = None
    heading_candidates = []
    font_stats = Counter()
    position_stats = Counter()
    repeated_texts = Counter()
    debug_info = []

    # Pass 1: Collect text blocks and font sizes, positions, etc.
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")['blocks']
        for block in blocks:
            if block['type'] != 0:
                continue
            for line in block['lines']:
                for span in line['spans']:
                    text = normalize_text(span['text'])
                    if not text.strip():
                        continue
                    size = round(span['size'], 1)
                    font = span['font']
                    flags = span['flags']
                    bbox = span['bbox']
                    x0, y0, x1, y1 = bbox
                    width = x1 - x0
                    height = y1 - y0
                    repeated_texts[text] += 1
                    font_stats[(size, font, flags)] += 1
                    position_stats[(round(x0, 1), round(y0, 1))] += 1
                    heading_candidates.append({
                        'text': text,
                        'size': size,
                        'font': font,
                        'flags': flags,
                        'page': page_num + 1,
                        'block': block,
                        'span': span,
                        'x0': x0,
                        'y0': y0,
                        'width': width,
                        'height': height
                    })

    # Filter out repeated elements (headers/footers/page numbers)
    repeated = {t for t, c in repeated_texts.items() if c > len(doc) * 0.7 and len(t) < 40}
    heading_candidates = [c for c in heading_candidates if c['text'] not in repeated]

    # Heuristic: Largest font on first page is likely the title
    first_page_candidates = [c for c in heading_candidates if c['page'] == 1]
    if first_page_candidates:
        title_candidate = max(first_page_candidates, key=lambda c: c['size'])
        title = title_candidate['text']

    # Cluster font sizes for heading levels
    sizes = sorted(set(c['size'] for c in heading_candidates))
    size_counts = Counter(c['size'] for c in heading_candidates)
    top_sizes = [s for s, _ in size_counts.most_common(6)]
    if len(top_sizes) < 3:
        top_sizes += [top_sizes[-1]] * (3 - len(top_sizes))
    h1_size, h2_size, h3_size = top_sizes[:3]

    # Build heading candidates with features
    enriched = []
    for c in heading_candidates:
        features = {
            'size': c['size'],
            'is_bold': bool(c['flags'] & 2),
            'is_all_caps': is_all_caps(c['text']),
            'is_title_case': is_title_case(c['text']),
            'is_numbered': is_heading_numbering(c['text']),
            'x0': c['x0'],
            'y0': c['y0'],
            'font': c['font'],
            'page': c['page'],
            'text_len': len(c['text'])
        }
        c['features'] = features
        enriched.append(c)

    # ML classifier or advanced heuristics
    headings = []
    for c in enriched:
        level = None
        if ML_AVAILABLE:
            # Use ML classifier
            X = [[
                c['features']['size'],
                int(c['features']['is_bold']),
                int(c['features']['is_all_caps']),
                int(c['features']['is_title_case']),
                int(c['features']['is_numbered']),
                c['features']['x0'],
                c['features']['y0'],
                c['features']['text_len']
            ]]
            pred = clf.predict(X)[0]
            if pred in ('H1', 'H2', 'H3'):
                level = pred
        else:
            # Heuristics
            if c['size'] == h1_size and (c['features']['is_bold'] or c['features']['is_all_caps'] or c['features']['is_numbered']):
                level = 'H1'
            elif c['size'] == h2_size and (c['features']['is_bold'] or c['features']['is_title_case'] or c['features']['is_numbered']):
                level = 'H2'
            elif c['size'] == h3_size and (c['features']['is_bold'] or c['features']['is_numbered']):
                level = 'H3'
        if level:
            headings.append({
                'level': level,
                'text': c['text'],
                'page': c['page'],
                'features': c['features']
            })
        if debug:
            debug_info.append({**c, 'assigned_level': level})

    # Build nested hierarchy
    def nest_headings(headings):
        result = []
        stack = []
        for h in headings:
            node = {k: h[k] for k in ('level', 'text', 'page')}
            node['children'] = []
            while stack and stack[-1]['level'] >= h['level']:
                stack.pop()
            if stack:
                stack[-1]['children'].append(node)
            else:
                result.append(node)
            stack.append({'level': h['level'], 'children': node['children']})
        return result

    # Convert level to int for nesting
    level_map = {'H1': 1, 'H2': 2, 'H3': 3}
    for h in headings:
        h['level'] = level_map.get(h['level'], 4)
    headings.sort(key=lambda x: (x['page'], x['level']))
    nested = nest_headings(headings)

    # Convert back to string levels
    def restore_levels(nodes):
        for n in nodes:
            n['level'] = {1: 'H1', 2: 'H2', 3: 'H3', 4: 'OTHER'}.get(n['level'], 'OTHER')
            if n['children']:
                restore_levels(n['children'])
    restore_levels(nested)

    # Write debug info
    if debug:
        debug_path = os.path.join('/app/output', os.path.basename(pdf_path)[:-4] + '_debug.json')
        with open(debug_path, 'w', encoding='utf-8') as f:
            json.dump(debug_info, f, ensure_ascii=False, indent=2)

    return {
        'title': title or '',
        'outline': nested
    } 