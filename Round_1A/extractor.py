# extractor.py

import fitz  # PyMuPDF
from PyPDF2 import PdfReader
import re

def is_noise(text):
    text = text.strip()
    if len(text) < 4:
        return True
    if re.match(r"^(Page|Fig|Table|Date|April|January|Monday|Tuesday|June|July)", text, re.IGNORECASE):
        return True
    if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]?\d{2,4}?\b", text):  # e.g., 12/12/2003
        return True
    if re.search(r"^\d{4}$", text):  # e.g., "2003"
        return True
    if all(c in "-–•◦. " for c in text):
        return True
    if text.startswith("◦") or text.count("◦") > 1:
        return True
    if len(text.split()) <= 1 and not text.isupper():  # likely not a heading
        return True
    return False

def extract_title_and_outline(pdf_path):
    title = extract_title_pymupdf(pdf_path)
    outline = extract_outline_hybrid(pdf_path)

    # Remove duplicate of title in outline
    outline = [
        item for item in outline
        if item["text"].strip().lower() != title.strip().lower()
    ]
    return title.strip(), outline

def extract_title_pymupdf(pdf_path):
    doc = fitz.open(pdf_path)
    page1 = doc[0]
    title_spans = []

    for block in page1.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if "Bold" in span["font"] and span["size"] > 10:
                    text = span["text"].strip()
                    if text and len(text.split()) > 2:
                        title_spans.append(text)

    title = " ".join(title_spans).strip()

    if not title:
        try:
            reader = PdfReader(pdf_path)
            metadata = reader.metadata
            title = metadata.title or "Untitled Document"
        except:
            title = "Untitled Document"

    return title

def extract_outline_hybrid(pdf_path):
    doc = fitz.open(pdf_path)
    outlines = []

    # Phase 1: TOC
    toc = doc.get_toc(simple=True)
    if toc:
        for level, text, page in toc:
            if not is_noise(text):
                outlines.append({
                    "level": f"H{level}",
                    "text": text.strip(),
                    "page": page
                })
        return clean_outline(outlines)

    # Phase 2: Visual-based
    spans = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                text_line = ""
                size = 0
                bold = False
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if not text:
                        continue
                    text_line += " " + text
                    size = max(size, round(span["size"], 1))
                    if "Bold" in span["font"]:
                        bold = True
                clean_text = text_line.strip()
                if clean_text and not is_noise(clean_text):
                    spans.append({
                        "text": clean_text,
                        "size": size,
                        "bold": bold,
                        "page": i + 1
                    })

    # Map font size to H1, H2, H3
    font_sizes = sorted(set(s["size"] for s in spans if s["bold"]), reverse=True)
    level_map = {}
    for idx, size in enumerate(font_sizes[:3]):
        level_map[size] = f"H{idx + 1}"

    for s in spans:
        if s["bold"] and s["size"] in level_map:
            outlines.append({
                "level": level_map[s["size"]],
                "text": s["text"],
                "page": s["page"]
            })

    return clean_outline(outlines)

def clean_outline(outlines):
    seen = set()
    final = []
    for item in outlines:
        key = (item["text"].lower().strip(), item["page"])
        if key not in seen and len(item["text"].strip()) > 4:
            seen.add(key)
            final.append(item)
    return final
