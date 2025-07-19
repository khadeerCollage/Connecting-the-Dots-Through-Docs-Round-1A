# extractor.py

import fitz  # PyMuPDF
from PyPDF2 import PdfReader
import re

# Try to import the smart ML-enhanced extractor
try:
    from hackathon_extractor import extract_title_and_outline as smart_extract
    SMART_EXTRACTOR_AVAILABLE = True
except ImportError:
    SMART_EXTRACTOR_AVAILABLE = False

def is_noise(text):
    text = text.strip()
    if len(text) < 2:
        return True
    if re.match(r"^(Page|Fig|Table|Date|April|January|Monday|Tuesday|June|July)\s*\d*$", text, re.IGNORECASE):
        return True
    if re.search(r"^\d{1,3}$", text):  # Just page numbers
        return True
    if re.search(r"^[.,;:!?]+$", text):  # Just punctuation
        return True
    if all(c in "-–•◦._=| " for c in text):  # Separators/decorations
        return True
    # Don't filter out URLs, emails, or important formatting
    return False

def is_likely_heading(text, size, bold, is_uppercase, position_y, avg_font_size):
    """Determine if text is likely a heading based on multiple factors"""
    text = text.strip()
    
    # Basic filters
    if len(text) < 2:
        return False
    if is_noise(text):
        return False
    
    score = 0
    
    # Font size factor (most important)
    if size > avg_font_size * 1.2:
        score += 3
    elif size > avg_font_size * 1.1:
        score += 2
    elif size > avg_font_size:
        score += 1
    
    # Bold text
    if bold:
        score += 2
    
    # Uppercase text
    if is_uppercase and len(text) > 3:
        score += 1
    
    # Position factors (top of page likely headings)
    if position_y < 100:  # Near top of page
        score += 1
    
    # Length factors
    word_count = len(text.split())
    if 2 <= word_count <= 15:  # Good heading length
        score += 1
    elif word_count > 20:  # Too long to be heading
        score -= 2
    
    # Pattern matching for common heading patterns
    if re.match(r'^(Round|Chapter|Section|Part|Step|Phase)\s+\d+', text, re.IGNORECASE):
        score += 2
    if re.match(r'^\d+\.', text):  # Numbered sections
        score += 1
    if text.endswith(':'):  # Often indicates sections
        score += 1
    
    # Content-based scoring
    heading_keywords = ['introduction', 'conclusion', 'overview', 'summary', 'challenge', 'round', 'phase', 'step', 'requirements', 'guidelines', 'instructions', 'background', 'objective', 'goal', 'welcome', 'appendix', 'submission', 'evaluation', 'deadline', 'deliverable']
    if any(keyword in text.lower() for keyword in heading_keywords):
        score += 2
    
    # Check for numbered sections or bullet points
    if re.match(r'^\d+\.\d+', text) or re.match(r'^[•▪▫◦]\s*', text):
        score += 1
    
    # Lower threshold for better detection
    return score >= 2

def extract_title_and_outline(pdf_path):
    """
    Main extraction function with ML enhancement for Adobe Hackathon
    Uses smart ML-inspired techniques when available, falls back to rule-based
    """
    if SMART_EXTRACTOR_AVAILABLE:
        try:
            # Use the smart ML-enhanced extractor
            title, outline = smart_extract(pdf_path)
            return title, outline
        except Exception as e:
            print(f"Smart extractor failed, using fallback: {e}")
    
    # Fallback to original rule-based approach
    title = extract_title_pymupdf(pdf_path)
    outline = extract_outline_hybrid(pdf_path)

    # Enhanced title filtering - remove title parts from outline
    outline = filter_title_from_outline(title, outline)
    return title.strip(), outline

def filter_title_from_outline(title, outline):
    """Remove title content from outline with better matching"""
    if not title or title == "Untitled Document":
        return outline
    
    # Clean and normalize title
    title_clean = title.strip().lower()
    title_words = set(word.strip() for word in title_clean.split() if len(word.strip()) > 2)
    
    filtered_outline = []
    
    for item in outline:
        text_clean = item["text"].strip().lower()
        text_words = set(word.strip() for word in text_clean.split() if len(word.strip()) > 2)
        
        should_skip = False
        
        # Skip if exact match
        if text_clean == title_clean:
            should_skip = True
            
        # Skip if text is just part of title (like "TRANSFORMERS" from "transformers for vision")
        elif text_words and title_words and text_words.issubset(title_words) and len(text_words) > 0:
            should_skip = True
            
        # Skip if it's a single word that's prominently in the title
        elif len(text_words) == 1 and text_words.issubset(title_words):
            should_skip = True
            
        # Skip if more than 80% of words match title words
        elif text_words and title_words:
            overlap = len(text_words.intersection(title_words))
            if overlap / len(text_words) > 0.8:
                should_skip = True
        
        # Special handling for page 1 - be more aggressive in filtering title components
        if item["page"] == 1 and not should_skip:
            # More strict filtering for page 1
            if title_words and any(word in text_clean for word in title_words if len(word) > 3):
                # Check if it's likely a title component
                if len(text_clean.split()) <= 2:  # Very short phrases on page 1 that match title
                    should_skip = True
                elif text_clean.isupper() and len(text_clean.split()) == 1:  # Single uppercase word matching title
                    should_skip = True
        
        if not should_skip:
            filtered_outline.append(item)
    
    return filtered_outline

def extract_title_pymupdf(pdf_path):
    doc = fitz.open(pdf_path)
    page1 = doc[0]
    title_candidates = []

    # Get all text from first page with formatting info
    page_dict = page1.get_text("dict")
    
    for block in page_dict["blocks"]:
        if "lines" not in block:
            continue
            
        for line in block["lines"]:
            if "spans" not in line:
                continue
                
            # Check if this line could be a title
            line_text = ""
            max_size = 0
            has_bold = False
            
            for span in line["spans"]:
                text = span["text"].strip()
                if text:
                    line_text += " " + text
                    size = span["size"]
                    max_size = max(max_size, size)
                    if "Bold" in span.get("font", "") or "bold" in span.get("font", "").lower():
                        has_bold = True
            
            line_text = line_text.strip()
            if line_text and len(line_text) > 3:
                # Get line position (higher on page = lower y value)
                line_bbox = line.get("bbox", [0, 0, 0, 0])
                y_position = line_bbox[1]
                
                title_candidates.append({
                    "text": line_text,
                    "size": max_size,
                    "bold": has_bold,
                    "y_position": y_position,
                    "word_count": len(line_text.split())
                })

    if title_candidates:
        # Sort by position (top first), then by size
        title_candidates.sort(key=lambda x: (x["y_position"], -x["size"]))
        
        # Look for the best title candidate
        for candidate in title_candidates[:5]:  # Check top 5 candidates
            text = candidate["text"]
            # Good title criteria - be more flexible
            if (candidate["size"] >= 10 and 
                candidate["word_count"] >= 2 and 
                candidate["word_count"] <= 20 and
                not is_noise(text) and
                candidate["y_position"] < 200):  # Must be in top portion
                return text

    # Fallback: try metadata
    try:
        reader = PdfReader(pdf_path)
        metadata = reader.metadata
        if metadata and metadata.title and metadata.title.strip() and metadata.title != "Untitled":
            return metadata.title.strip()
    except:
        pass

    # Last resort: try to combine potential title parts from top of page
    if title_candidates:
        # Take first few candidates that might be title parts
        title_parts = []
        for candidate in title_candidates[:4]:
            if candidate["y_position"] < 150 and candidate["size"] >= 10:  # Near top of page and decent size
                title_parts.append(candidate["text"])
        
        if title_parts:
            combined_title = " ".join(title_parts)
            # Don't return overly long combined titles
            if len(combined_title.split()) <= 15:
                return combined_title

    return "Untitled Document"

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
        if outlines:  # Only return TOC if we found something good
            return clean_outline(outlines)

    # Phase 2: Enhanced Visual-based extraction
    all_spans = []
    font_sizes = []
    
    # First pass: collect all text spans with their properties
    for i in range(min(len(doc), 50)):  # Limit to first 50 pages for performance
        page = doc.load_page(i)
        page_dict = page.get_text("dict")
        
        for block in page_dict["blocks"]:
            if "lines" not in block:
                continue
                
            for line in block["lines"]:
                if "spans" not in line:
                    continue
                    
                # Get line bounding box for position
                line_bbox = line.get("bbox", [0, 0, 0, 0])
                
                text_parts = []
                max_size = 0
                has_bold = False
                
                for span in line["spans"]:
                    text = span["text"].strip()
                    if text:
                        text_parts.append(text)
                        size = round(span["size"], 1)
                        max_size = max(max_size, size)
                        font_sizes.append(size)
                        
                        if "Bold" in span.get("font", "") or "bold" in span.get("font", "").lower():
                            has_bold = True
                
                if text_parts:
                    full_text = " ".join(text_parts).strip()
                    if full_text and len(full_text) > 1:
                        all_spans.append({
                            "text": full_text,
                            "size": max_size,
                            "bold": has_bold,
                            "page": i + 1,
                            "position_y": line_bbox[1],
                            "is_uppercase": full_text.isupper()
                        })

    if not all_spans:
        return []
    
    # Calculate average font size for reference
    avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
    
    # Filter spans that are likely headings
    heading_candidates = []
    for span in all_spans:
        if is_likely_heading(
            span["text"], 
            span["size"], 
            span["bold"], 
            span["is_uppercase"],
            span["position_y"],
            avg_font_size
        ):
            heading_candidates.append(span)
    
    # If we don't have many candidates, lower the threshold
    if len(heading_candidates) < 5:
        for span in all_spans:
            # More lenient criteria for documents with few headings
            text = span["text"].strip()
            if (span["size"] > avg_font_size * 0.95 and 
                (span["bold"] or text.isupper() or 
                 any(keyword in text.lower() for keyword in ['round', 'phase', 'challenge', 'step', 'objective', 'requirement', 'submission', 'evaluation', 'deadline', 'persona', 'document', 'intelligence', 'appendix']) or
                 re.match(r'^\d+\.', text) or text.endswith(':')) and
                not is_noise(text) and
                len(text.split()) >= 1 and len(text.split()) <= 20):
                if span not in heading_candidates:
                    heading_candidates.append(span)
    
    # Group by font size and boldness to determine hierarchy
    size_groups = {}
    for candidate in heading_candidates:
        key = (round(candidate["size"], 1), candidate["bold"])
        if key not in size_groups:
            size_groups[key] = []
        size_groups[key].append(candidate)
    
    # Sort by importance (size first, then bold)
    sorted_groups = sorted(size_groups.keys(), key=lambda x: (x[0], x[1]), reverse=True)
    
    # Assign heading levels more intelligently
    level_map = {}
    level = 1
    for group_key in sorted_groups[:6]:  # Limit to 6 levels (H1-H6)
        level_map[group_key] = f"H{level}"
        level += 1
    
    # Create outline entries
    for candidate in heading_candidates:
        key = (round(candidate["size"], 1), candidate["bold"])
        if key in level_map:
            outlines.append({
                "level": level_map[key],
                "text": candidate["text"],
                "page": candidate["page"]
            })
    
    # Sort by page order
    outlines.sort(key=lambda x: (x["page"], x["text"]))
    
    return clean_outline(outlines)

def clean_outline(outlines):
    seen = set()
    final = []
    for item in outlines:
        # More lenient cleaning - allow shorter headings
        text = item["text"].strip()
        key = (text.lower(), item["page"])
        if key not in seen and len(text) >= 2:
            # Additional filtering for obviously bad entries
            if not (text.isdigit() or text in ['•', '-', '–', '—', '|']):
                seen.add(key)
                final.append(item)
    return final
