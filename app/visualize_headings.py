import fitz
import os
import base64
from outline_extractor import extract_outline

COLORS = {'H1': '#FF5733', 'H2': '#33C1FF', 'H3': '#75FF33', 'OTHER': '#CCCCCC'}


def render_page_with_headings(pdf_path, outline, output_dir):
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=120)
        img_data = pix.tobytes('png')
        img_b64 = base64.b64encode(img_data).decode('utf-8')
        # Find headings on this page
        page_headings = []
        def collect_headings(nodes):
            for n in nodes:
                if n['page'] == page_num + 1:
                    page_headings.append(n)
                if 'children' in n:
                    collect_headings(n['children'])
        collect_headings(outline['outline'])
        # HTML for overlay
        overlay_html = ''
        for h in page_headings:
            color = COLORS.get(h['level'], '#CCCCCC')
            overlay_html += f'<div style="position:absolute;left:0;top:0;color:{color};font-weight:bold;font-size:18px;">{h["level"]}: {h["text"]}</div>'
        images.append(f'<div style="position:relative;display:inline-block;margin:10px;">'
                      f'<img src="data:image/png;base64,{img_b64}" style="display:block;">{overlay_html}</div>')
    return images


def visualize_pdf(pdf_path, output_dir='/app/output'):
    outline = extract_outline(pdf_path)
    images = render_page_with_headings(pdf_path, outline, output_dir)
    html = '<html><head><title>PDF Headings Visualization</title></head><body>'
    html += '<h2>Detected Headings Visualization</h2>'
    html += ''.join(images)
    html += '</body></html>'
    out_html = os.path.join(output_dir, os.path.basename(pdf_path)[:-4] + '_visualization.html')
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Visualization saved to {out_html}')

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python visualize_headings.py /path/to/pdf')
    else:
        visualize_pdf(sys.argv[1]) 