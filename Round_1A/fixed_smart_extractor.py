# fixed_smart_extractor.py - Fixed PDF Extractor that properly combines text spans

import fitz  # PyMuPDF
from PyPDF2 import PdfReader
import re
import os
import json
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FixedPDFExtractor:
    """Fixed PDF extractor that properly combines text spans to avoid fragmentation"""
    
    def __init__(self):
        pass
    
    def extract_title_and_outline(self, pdf_path: str) -> Tuple[str, List[Dict]]:
        """Extract title and outline with proper text combination"""
        try:
            title = self._extract_title(pdf_path)
            outline = self._extract_outline(pdf_path)
            
            # Filter out title components from outline
            outline = self._filter_title_from_outline(title, outline)
            
            # Final cleanup
            outline = self._final_cleanup(outline)
            
            logger.info(f"Extracted title: {title}")
            logger.info(f"Extracted {len(outline)} outline items")
            
            return title, outline
            
        except Exception as e:
            logger.error(f"Extraction failed for {pdf_path}: {e}")
            return "Untitled Document", []
    
    def _extract_title(self, pdf_path: str) -> str:
        """Extract title from first page with proper text combination"""
        try:
            doc = fitz.open(pdf_path)
            if len(doc) == 0:
                return "Untitled Document"
            
            page1 = doc[0]
            page_dict = page1.get_text("dict")
            
            title_candidates = []
            
            # Process blocks to combine spans properly
            for block in page_dict.get("blocks", []):
                if "lines" not in block:
                    continue
                
                for line in block["lines"]:
                    if "spans" not in line:
                        continue
                    
                    # Combine all spans in the line
                    combined_text = ""
                    max_size = 0
                    is_bold = False
                    
                    for span in line["spans"]:
                        text = span.get("text", "").strip()
                        if text:
                            # Add space if needed
                            if combined_text and not combined_text.endswith(" ") and not text.startswith(" "):
                                combined_text += " "
                            combined_text += text
                            
                            max_size = max(max_size, span.get("size", 12))
                            if span.get("flags", 0) & 16:  # Bold flag
                                is_bold = True
                    
                    combined_text = self._clean_text(combined_text)
                    
                    if combined_text and len(combined_text) > 3:
                        y_pos = line.get("bbox", [0, 0, 0, 0])[1]
                        
                        title_candidates.append({
                            "text": combined_text,
                            "size": max_size,
                            "bold": is_bold,
                            "y_position": y_pos,
                            "word_count": len(combined_text.split())
                        })
            
            # Find best title candidate
            if title_candidates:
                # Sort by y-position (top first), then by font size (largest first)
                title_candidates.sort(key=lambda x: (x["y_position"], -x["size"]))
                
                for candidate in title_candidates[:5]:  # Check top 5 candidates
                    if (candidate["size"] >= 12 and 
                        2 <= candidate["word_count"] <= 20 and
                        candidate["y_position"] < 300 and  # Must be in upper part of page
                        not self._is_noise(candidate["text"])):
                        return candidate["text"]
            
            # Fallback to metadata
            try:
                reader = PdfReader(pdf_path)
                if reader.metadata and reader.metadata.title:
                    return reader.metadata.title.strip()
            except:
                pass
            
            doc.close()
            return "Untitled Document"
            
        except Exception as e:
            logger.error(f"Title extraction failed: {e}")
            return "Untitled Document"
    
    def _extract_outline(self, pdf_path: str) -> List[Dict]:
        """Extract outline with proper text combination"""
        try:
            doc = fitz.open(pdf_path)
            outline = []
            
            # Try TOC first
            toc = doc.get_toc(simple=True)
            if toc:
                for level, text, page in toc:
                    clean_text = self._clean_text(text)
                    if clean_text and not self._is_noise(clean_text):
                        outline.append({
                            "level": f"H{min(level, 6)}",  # Cap at H6
                            "text": clean_text,
                            "page": page
                        })
                if outline:
                    doc.close()
                    return outline
            
            # Visual extraction with proper text combination
            font_sizes = []
            text_lines = []
            
            # Analyze pages (limit to first 30 pages for performance)
            for page_num in range(min(30, len(doc))):
                page = doc[page_num]
                page_dict = page.get_text("dict")
                
                for block in page_dict.get("blocks", []):
                    if "lines" not in block:
                        continue
                    
                    for line in block["lines"]:
                        if "spans" not in line:
                            continue
                        
                        # Combine all spans in the line properly
                        combined_text = ""
                        max_size = 0
                        is_bold = False
                        
                        for span in line["spans"]:
                            text = span.get("text", "").strip()
                            if text:
                                # Add space if needed for proper word separation
                                if combined_text and not combined_text.endswith(" ") and not text.startswith(" "):
                                    combined_text += " "
                                combined_text += text
                                
                                size = span.get("size", 12)
                                max_size = max(max_size, size)
                                font_sizes.append(size)
                                
                                if span.get("flags", 0) & 16:  # Bold flag
                                    is_bold = True
                        
                        combined_text = self._clean_text(combined_text)
                        
                        if combined_text and len(combined_text) > 1:
                            y_pos = line.get("bbox", [0, 0, 0, 0])[1]
                            
                            text_lines.append({
                                "text": combined_text,
                                "size": max_size,
                                "bold": is_bold,
                                "page": page_num + 1,
                                "y_position": y_pos
                            })
            
            if not text_lines:
                doc.close()
                return []
            
            # Calculate average font size
            avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
            logger.info(f"Average font size: {avg_font_size:.1f}")
            
            # Find headings
            for text_line in text_lines:
                if self._is_likely_heading(text_line, avg_font_size):
                    level = self._determine_heading_level(text_line, avg_font_size)
                    outline.append({
                        "level": level,
                        "text": text_line["text"],
                        "page": text_line["page"]
                    })
            
            doc.close()
            return self._deduplicate_outline(outline)
            
        except Exception as e:
            logger.error(f"Outline extraction failed: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """Clean text while preserving meaningful content"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common PDF artifacts
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)  # Control characters
        
        return text
    
    def _is_noise(self, text: str) -> bool:
        """Check if text is noise/artifact"""
        if not text or len(text) < 2:
            return True
        
        # Common noise patterns
        noise_patterns = [
            r'^(Page|Fig|Figure|Table|Appendix)\s*\d*$',
            r'^\d+$',  # Just numbers
            r'^[.,;:!?]+$',  # Just punctuation
            r'^[A-Z]$',  # Single letters
            r'^\s*$',  # Just whitespace
            r'^©.*$',  # Copyright
            r'^www\.',  # URLs
            r'^http',  # URLs
            r'^\d{1,3}$',  # Page numbers
            r'^\.+$',  # Just dots
            r'^[.\s]+$',  # Dots and spaces
            r'^Version\s+\d+(\.\d+)*$',  # Version numbers like "Version 1.0"
            r'^Revision\s+History\s*\d*$',  # Revision History
            r'^Table\s+of\s+Contents\s*\d*$',  # Table of Contents
            r'^References\s*\d*$',  # References
            r'^\d+\.\s*$',  # Just numbered items without text
            r'^[.\-_=]+$',  # Lines made of dots, dashes, underscores, equals
        ]
        
        for pattern in noise_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        
        # Check for lines that are mostly dots or other repetitive characters
        if len(text) > 10:
            # Count dots, dashes, spaces
            special_chars = text.count('.') + text.count('-') + text.count('_') + text.count('=') + text.count(' ')
            if special_chars / len(text) > 0.8:  # If 80% or more are special chars
                return True
        
        return False
    
    def _is_likely_heading(self, text_line: Dict, avg_font_size: float) -> bool:
        """Determine if text line is likely a heading"""
        text = text_line["text"].strip()
        
        # Basic filters
        if len(text) < 3 or len(text) > 300:
            return False
        
        if self._is_noise(text):
            return False
        
        # Skip very common words/phrases that aren't headings
        common_non_headings = [
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'end', 'few', 'man', 'run', 'say', 'she', 'try', 'way', 'too'
        ]
        if text.lower() in common_non_headings:
            return False
        
        # Additional filters for common document artifacts
        document_artifacts = [
            'revision history', 'table of contents', 'list of figures', 'list of tables',
            'references', 'bibliography', 'appendix', 'glossary', 'index',
            'acknowledgments', 'preface', 'foreword'
        ]
        if any(artifact in text.lower() for artifact in document_artifacts) and len(text) < 30:
            return False
        
        # Filter out version numbers and dates
        if re.match(r'^(version|v\.?)\s*\d+(\.\d+)*$', text, re.IGNORECASE):
            return False
        
        # Filter out table/figure references
        if re.match(r'^(table|figure|fig)\s*\d+', text, re.IGNORECASE):
            return False
        
        # Filter out lines that are mostly version/date information
        if re.search(r'\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)', text, re.IGNORECASE):
            return False
        
        score = 0
        
        # Font size factor (most important)
        size_ratio = text_line["size"] / avg_font_size
        if size_ratio > 1.4:
            score += 4
        elif size_ratio > 1.2:
            score += 3
        elif size_ratio > 1.1:
            score += 2
        elif size_ratio > 1.0:
            score += 1
        
        # Bold text
        if text_line["bold"]:
            score += 2
        
        # Length factor - headings are usually not too long
        word_count = len(text.split())
        if 2 <= word_count <= 15:
            score += 2
        elif word_count <= 25:
            score += 1
        elif word_count > 50:
            score -= 3
        
        # Pattern matching for common heading structures
        if re.match(r'^(Chapter|Section|Part|Article)\s+\d+', text, re.IGNORECASE):
            score += 3
        elif re.match(r'^\d+\.\s+[A-Za-z]', text):  # Numbered headings with actual text
            score += 2
        elif re.match(r'^[A-Z][a-z]+\s+[A-Z]', text):  # Title case
            score += 1
        
        # Keyword matching for common heading words
        heading_keywords = [
            'introduction', 'conclusion', 'overview', 'summary', 'background',
            'methodology', 'results', 'discussion', 'abstract',
            'purpose', 'objectives', 'scope', 'requirements', 'specifications',
            'proposal', 'request', 'analysis', 'implementation', 'evaluation'
        ]
        if any(keyword in text.lower() for keyword in heading_keywords):
            score += 2
        
        # Penalize very short fragments that look like broken text
        if len(text) < 5 and not re.match(r'^\d+\.', text):
            score -= 2
        
        # Penalize lines with too many numbers/dates (likely metadata)
        number_count = len(re.findall(r'\d+', text))
        if number_count > 3:
            score -= 2
        
        return score >= 4  # Increased threshold for better filtering
    
    def _determine_heading_level(self, text_line: Dict, avg_font_size: float) -> str:
        """Determine heading level based on formatting"""
        size_ratio = text_line["size"] / avg_font_size
        text = text_line["text"]
        
        # H1: Largest fonts, often bold, important sections
        if size_ratio > 1.6 or (size_ratio > 1.4 and text_line["bold"]):
            return "H1"
        # H2: Large fonts, section headers
        elif size_ratio > 1.3 or (size_ratio > 1.2 and text_line["bold"]):
            return "H2"
        # H3: Medium-large fonts, subsections
        elif size_ratio > 1.15 or (size_ratio > 1.1 and text_line["bold"]):
            return "H3"
        # H4: Smaller headings
        elif size_ratio > 1.05 or text_line["bold"]:
            return "H4"
        else:
            return "H5"
    
    def _deduplicate_outline(self, outline: List[Dict]) -> List[Dict]:
        """Remove duplicates and very similar entries"""
        if not outline:
            return []
        
        seen = set()
        deduplicated = []
        
        for item in outline:
            text = item["text"].strip()
            
            # Create a key for deduplication (case-insensitive)
            key = (text.lower(), item["page"])
            
            if key not in seen and len(text) >= 3:
                seen.add(key)
                deduplicated.append(item)
        
        return deduplicated
    
    def _filter_title_from_outline(self, title: str, outline: List[Dict]) -> List[Dict]:
        """Remove title components from outline"""
        if not title or title == "Untitled Document":
            return outline
        
        title_words = set(word.lower().strip() for word in title.split() if len(word.strip()) > 2)
        filtered = []
        
        for item in outline:
            text_words = set(word.lower().strip() for word in item["text"].split() if len(word.strip()) > 2)
            
            # Skip if too similar to title
            if text_words and title_words:
                overlap = len(text_words.intersection(title_words))
                similarity = overlap / max(len(text_words), len(title_words))
                if similarity < 0.7:  # Keep if less than 70% similarity
                    filtered.append(item)
            else:
                filtered.append(item)
        
        return filtered
    
    def _final_cleanup(self, outline: List[Dict]) -> List[Dict]:
        """Final cleanup to remove remaining noise"""
        cleaned = []
        
        for item in outline:
            text = item["text"].strip()
            
            # Skip very short or very long items
            if len(text) < 3 or len(text) > 200:
                continue
            
            # Skip items that are clearly not headings
            if self._is_noise(text):
                continue
            
            # Skip incomplete words/fragments
            if len(text) < 10 and not re.match(r'^\d+\.', text) and ' ' not in text:
                continue
            
            cleaned.append(item)
        
        return cleaned

def extract_pdf_smart_fixed(pdf_path: str, output_path: str = None) -> Dict[str, Any]:
    """Main function to extract PDF with fixed text combination"""
    try:
        extractor = FixedPDFExtractor()
        title, outline = extractor.extract_title_and_outline(pdf_path)
        
        result = {
            "title": title,
            "outline": outline
        }
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_path}")
        
        return result
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return {"title": "Untitled Document", "outline": []}

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python fixed_smart_extractor.py <pdf_path> [output_path]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    result = extract_pdf_smart_fixed(pdf_path, output_path)
    
    print(f"\nTitle: {result['title']}")
    print(f"Outline items: {len(result['outline'])}")
    
    for item in result['outline'][:10]:  # Show first 10 items
        print(f"  {item['level']}: {item['text']} (page {item['page']})")
    
    if len(result['outline']) > 10:
        print(f"  ... and {len(result['outline']) - 10} more items")
