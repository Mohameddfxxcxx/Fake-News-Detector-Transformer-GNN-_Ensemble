"""Render each page of the paper PDF to a PNG image so we can view it visually."""
import os
import fitz  # PyMuPDF

PDF_PATH = "s41598-025-31653-3.pdf"
OUT_DIR = "paper_pages"
os.makedirs(OUT_DIR, exist_ok=True)

doc = fitz.open(PDF_PATH)
print(f"Pages: {doc.page_count}")

zoom = 2.0  # 144 dpi
mat = fitz.Matrix(zoom, zoom)
for i, page in enumerate(doc):
    pix = page.get_pixmap(matrix=mat)
    out = os.path.join(OUT_DIR, f"page_{i+1:02d}.png")
    pix.save(out)
    print(out, pix.width, "x", pix.height)
