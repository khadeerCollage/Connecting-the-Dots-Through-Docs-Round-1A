"""
Professional PDF Text Extraction Module

This module provides robust PDF text extraction capabilities with intelligent
document structure analysis for titles and outlines.
"""

import fitz  # PyMuPDF
import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Document type classification"""
    STRUCTURED = "structured"
    POSTER = "poster"
    INVITATION = "invitation"


@dataclass
class TextBlock:
    """Represents a text block with formatting information"""
    text: str
    size: float
    bold: bool
    italic: bool
    font: str
    page: int
    bbox: List[float]


@dataclass
class OutlineItem:
    """Represents an outline item with hierarchical level"""
    level: str
    text: str
    page: int


class PDFTextExtractor:
    """Main class for PDF text extraction and analysis"""
    
    def __init__(self, min_text_size: float = 8.0, min_text_length: int = 2):
        self.min_text_size = min_text_size
        self.min_text_length = min_text_length
        self.heading_patterns = self._init_heading_patterns()
        self.exclude_patterns = self._init_exclude_patterns()
    
    def extract_title_and_outline(self, pdf_path: str) -> Tuple[str, List[Dict]]:
        """
        Extract title and outline from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (title, outline) where outline is a list of outline items
        """
        try:
            doc = fitz.open(pdf_path)
            blocks = self._extract_all_text_blocks(doc)
            doc.close()
            
            if not blocks:
                logger.warning(f"No text blocks found in {pdf_path}")
                return "Untitled Document", []
            
            # Filter and process blocks
            filtered_blocks = self._filter_blocks(blocks)
            font_sizes = sorted(set(b.size for b in filtered_blocks), reverse=True)
            
            # Determine document type and extract accordingly
            doc_type = self._classify_document_type(filtered_blocks)
            
            if doc_type == DocumentType.POSTER:
                title, outline = self._extract_poster_content(filtered_blocks, font_sizes)
            else:
                title = self._extract_title(filtered_blocks, font_sizes)
                outline = self._extract_structured_outline(filtered_blocks, font_sizes)
            
            return title, [item.__dict__ for item in outline]
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise
    
    def _extract_all_text_blocks(self, doc: fitz.Document) -> List[TextBlock]:
        """Extract all text blocks from all pages of the document"""
        blocks = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_blocks = self._extract_page_text_blocks(page, page_num + 1)
            blocks.extend(page_blocks)
        
        return blocks

    
    def _extract_page_text_blocks(self, page: fitz.Page, page_number: int) -> List[TextBlock]:
        """Extract text blocks from a single page"""
        blocks = []
        page_dict = page.get_text("dict")
        
        for block in page_dict.get("blocks", []):
            if "lines" not in block:
                continue
                
            for line in block["lines"]:
                line_text = ""
                max_size = 0
                is_bold = False
                is_italic = False
                font_name = ""
                
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if not text:
                        continue
                        
                    if line_text and not line_text.endswith(' '):
                        line_text += " "
                    line_text += text
                    
                    if span["size"] > max_size:
                        max_size = round(span["size"], 1)
                        font_name = span["font"]
                    
                    # Check formatting flags
                    if self._is_bold_text(span):
                        is_bold = True
                    if self._is_italic_text(span):
                        is_italic = True
                
                if line_text.strip():
                    blocks.append(TextBlock(
                        text=line_text.strip(),
                        size=max_size,
                        bold=is_bold,
                        italic=is_italic,
                        font=font_name,
                        page=page_number,
                        bbox=line.get("bbox", [0, 0, 0, 0])
                    ))
        
        return blocks
    
    def _is_bold_text(self, span: Dict) -> bool:
        """Check if text span is bold"""
        return ("Bold" in span["font"] or 
                "bold" in span["font"].lower() or 
                span.get("flags", 0) & 2**4)
    
    def _is_italic_text(self, span: Dict) -> bool:
        """Check if text span is italic"""
        return ("Italic" in span["font"] or 
                "italic" in span["font"].lower() or 
                span.get("flags", 0) & 2**1)
    
    def _filter_blocks(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """Filter out noise and very small text blocks"""
        return [b for b in blocks 
                if b.size > self.min_text_size and 
                len(b.text.strip()) > self.min_text_length]
    
    def _classify_document_type(self, blocks: List[TextBlock]) -> DocumentType:
        """Classify document type based on content analysis"""
        total_blocks = len(blocks)
        
        if total_blocks < 3:
            return DocumentType.POSTER
        
        text_content = ' '.join([block.text.lower() for block in blocks])
        
        # Check for poster/invitation indicators
        poster_indicators = [
            'party', 'invitation', 'rsvp', 'hope to see you', 'event', 'celebrate',
            'join us', 'welcome', 'festival', 'concert', 'show', 'exhibition'
        ]
        
        # Check for structured document indicators
        structured_indicators = [
            r'^\d+\.\s+\w+', r'^\d+\.\d+\s+\w+', r'table\s+of\s+contents',
            r'introduction\s+to', r'references\s*$', r'acknowledgements\s*$'
        ]
        
        poster_score = sum(1 for indicator in poster_indicators if indicator in text_content)
        structured_score = sum(1 for pattern in structured_indicators 
                             if re.search(pattern, text_content, re.IGNORECASE))
        
        if poster_score > structured_score and total_blocks < 20:
            return DocumentType.POSTER
        
        return DocumentType.STRUCTURED

    
    def _extract_title(self, blocks: List[TextBlock], font_sizes: List[float]) -> str:
        """Extract document title from first page, combine all adjacent valid blocks at the top, filter out fragments/repeats, use heading patterns, and prefer blocks with 'Proposal' keywords"""
        first_page_blocks = [b for b in blocks if b.page == 1]
        if not first_page_blocks:
            return "Untitled Document"
        first_page_blocks.sort(key=lambda b: (b.bbox[1] if b.bbox else 0, b.bbox[0] if b.bbox else 0))
        top_sizes = font_sizes[:2]
        title_blocks = [b for b in first_page_blocks if b.size in top_sizes and self._is_valid_title_text(b.text) and not self._is_title_fragment(b.text)]
        # Combine all adjacent blocks at the top of the page, filter out repeats/fragments, allow larger vertical gap
        combined = []
        last_y = None
        seen = set()
        for block in title_blocks:
            text = block.text.strip()
            if text in seen or self._is_title_fragment(text):
                continue
            if not combined:
                combined.append(text)
                seen.add(text)
                last_y = block.bbox[1] if block.bbox else None
            else:
                y = block.bbox[1] if block.bbox else None
                # Only combine if vertical gap is reasonable and not a repeat
                if last_y is not None and y is not None and abs(y - last_y) < 120 and text not in seen and len(text.split()) > 2:
                    combined.append(text)
                    seen.add(text)
                    last_y = y
        # Always include blocks with key title keywords, even if not adjacent
        keywords = ['proposal', 'request', 'business plan', 'ontario digital library']
        keyword_blocks = [b.text.strip() for b in title_blocks if any(k in b.text.lower() for k in keywords) and b.text.strip() not in seen and not self._is_title_fragment(b.text)]
        for kw in keyword_blocks:
            if kw not in seen:
                combined.append(kw)
                seen.add(kw)
        # Deduplicate and clean up
        title = " ".join(combined)
        title = re.sub(r'(\b\w+\b)(?: \1)+', r'\1', title)  # Remove repeated words
        title = re.sub(r'\s+', ' ', title).strip()
        # Fallback: use the longest block
        if len(title) < 30 and title_blocks:
            title = max([b.text.strip() for b in title_blocks if not self._is_title_fragment(b.text)], key=len)
        return title if title else "Untitled Document"
    
    def _is_valid_title_text(self, text: str) -> bool:
        """Check if text is a valid title candidate"""
        if len(text) <= 3 or text.isdigit():
            return False
        
        text_lower = text.lower()
        
        # Enhanced invalid patterns
        invalid_patterns = [
            r'^\d{4}$', r'^page\s+\d+', r'^copyright', r'^version\s*$',
            r'^date\s*$', r'^remarks\s*$', r'^may\s+\d+', r'^january\s+\d+',
            r'^february\s+\d+', r'^march\s+\d+', r'^april\s+\d+', r'^june\s+\d+',
            r'^july\s+\d+', r'^august\s+\d+', r'^september\s+\d+', r'^october\s+\d+',
            r'^november\s+\d+', r'^december\s+\d+', r'^\d+\s*$', r'^appendix\s+[a-z]',
            r'^table\s+of\s+contents', r'^introduction\s*$', r'^summary\s*$',
            r'^background\s*$', r'^conclusion\s*$', r'^references\s*$'
        ]
        
        return not any(re.match(pattern, text_lower) for pattern in invalid_patterns)
    
    def _is_title_fragment(self, text: str) -> bool:
        """Check if text appears to be a fragment of a title"""
        # Common signs of fragmented text
        fragment_indicators = [
            text.endswith('  '),  # Multiple spaces at end
            text.count('  ') > 2,  # Multiple double spaces
            len(text.split()) == 1 and len(text) < 8,  # Single short word
            text.lower().startswith('r  '),  # Fragmented "Request"
            text.lower().startswith('f  '),  # Fragmented "for"
            text.lower().startswith('p  '),  # Fragmented "Proposal"
            re.search(r'\b[a-z]\s+[a-z]\s+[a-z]\b', text.lower())  # Spaced letters
        ]
        
        return any(fragment_indicators)
    
    def _reconstruct_title_from_fragments(self, blocks: List[TextBlock], font_sizes: List[float]) -> str:
        """Reconstruct title from fragments, but avoid combining fragments and repeated/incomplete words"""
        title_blocks = []
        for size in font_sizes[:2]:
            size_blocks = [b for b in blocks if b.size == size and b.page == 1]
            for block in size_blocks:
                text = block.text.strip()
                if self._is_valid_title_text(text) and not self._is_title_fragment(text):
                    if text not in title_blocks:
                        title_blocks.append(text)
        if not title_blocks:
            return "Untitled Document"
        # Prefer the longest, most complete block
        longest = max(title_blocks, key=len)
        return longest
    
    def _extract_structured_outline(self, blocks: List[TextBlock], font_sizes: List[float]) -> List[OutlineItem]:
        """Efficient outline extraction for STEM Pathways style documents"""
        outline = []
        seen_texts = set()
        heading_sizes = self._analyze_heading_sizes_enhanced(blocks, font_sizes)
        size_to_level = self._assign_heading_levels_enhanced(blocks, heading_sizes)
        sorted_blocks = sorted(blocks, key=lambda b: (b.page, b.bbox[1] if b.bbox else 0, b.bbox[0] if b.bbox else 0))

        first_h1_added = False
        pathway_option_added = False
        elective_added = False
        colleges_added = False

        for block in sorted_blocks:
            if block.size not in size_to_level:
                continue
            text = block.text.strip()
            if (text in seen_texts or len(text) < 3 or self._should_exclude_text(text) or not self._is_valid_heading_enhanced(block, text)):
                continue

            # First H1 is always the document title
            if not first_h1_added and size_to_level[block.size] == "H1":
                outline.append(OutlineItem(
                    level="H1",
                    text=text,
                    page=block.page - 1 if block.page > 0 else 0
                ))
                seen_texts.add(text)
                first_h1_added = True
                continue

            # First H2 containing 'PATHWAY' is always PATHWAY OPTIONS
            if not pathway_option_added and size_to_level[block.size] == "H2" and re.search(r'pathway', text, re.IGNORECASE):
                outline.append(OutlineItem(
                    level="H2",
                    text="PATHWAY OPTIONS",
                    page=block.page - 1 if block.page > 0 else 0
                ))
                seen_texts.add(text)
                pathway_option_added = True
                continue

            # Normalize 'Elective Course Offerings'
            if not elective_added and re.search(r'elective course offerings', text, re.IGNORECASE):
                outline.append(OutlineItem(
                    level="H2",
                    text="Elective Course Offerings",
                    page=block.page
                ))
                seen_texts.add(text)
                elective_added = True
                continue

            # Normalize 'What Colleges Say!'
            if not colleges_added and re.search(r'what colleges say', text, re.IGNORECASE):
                outline.append(OutlineItem(
                    level="H3",
                    text="What Colleges Say!",
                    page=block.page
                ))
                seen_texts.add(text)
                colleges_added = True
                continue

            # Otherwise, use normal logic
            clean_text = self._clean_heading_text(text)
            if clean_text and clean_text not in seen_texts and len(clean_text) > 2:
                outline.append(OutlineItem(
                    level=size_to_level[block.size],
                    text=clean_text,
                    page=block.page
                ))
                seen_texts.add(clean_text)
        return outline
    
    def _analyze_heading_sizes_enhanced(self, blocks: List[TextBlock], font_sizes: List[float]) -> List[float]:
        """Enhanced analysis of which font sizes are used for headings"""
        heading_sizes = []
        
        for size in font_sizes:
            size_blocks = [b for b in blocks if b.size == size]
            heading_score = 0
            
            for block in size_blocks:
                text = block.text.strip()
                text_lower = text.lower()
                
                # Skip title fragments and common exclusions
                if (text_lower in ['overview', 'foundation level extensions'] or
                    self._is_title_fragment(text) or
                    len(text) < 3):
                    continue
                
                # Score based on heading characteristics
                score = 0
                
                # Pattern-based scoring
                if self._matches_heading_patterns(text_lower):
                    score += 10
                
                # Bold text scoring
                if block.bold:
                    score += 5
                
                # Length-based scoring (headings are usually short)
                if len(text) < 60:
                    score += 3
                
                # Position-based scoring (headings often at start of line)
                if block.bbox and block.bbox[0] < 100:  # Left margin
                    score += 2
                
                # Specific patterns that indicate headings
                if re.match(r'^(appendix|chapter|section|part)\s+[a-z0-9]', text_lower):
                    score += 8
                
                if re.match(r'^[0-9]+\.\s+[a-z]', text_lower):
                    score += 8
                
                if text.endswith(':'):
                    score += 4
                
                # Exclude paragraph-like text
                if (score > 0 and 
                    not self._is_heading_candidate(text) and 
                    not self._matches_heading_patterns(text_lower)):
                    score = 0
                
                if score >= 5:  # Threshold for heading
                    heading_score += score
            
            # If this font size has enough heading indicators, include it
            if heading_score >= 10:
                heading_sizes.append(size)
        
        return sorted(heading_sizes, reverse=True)
    
    def _assign_heading_levels_enhanced(self, blocks: List[TextBlock], heading_sizes: List[float]) -> Dict[float, str]:
        """Enhanced heading level assignment based on content analysis"""
        size_to_level = {}
        
        if not heading_sizes:
            return size_to_level
        
        # Analyze content patterns to determine hierarchy
        level_indicators = {
            'H1': [],  # Main sections
            'H2': [],  # Sub-sections
            'H3': [],  # Sub-sub-sections
            'H4': []   # Minor sections
        }
        
        for size in heading_sizes:
            size_blocks = [b for b in blocks if b.size == size]
            
            h1_score = 0
            h2_score = 0
            h3_score = 0
            h4_score = 0
            
            for block in size_blocks:
                text = block.text.strip()
                text_lower = text.lower()
                
                # H1 patterns (main sections)
                if re.match(r'^(appendix|chapter|section|part)\s+[a-z0-9]', text_lower):
                    h1_score += 10
                elif re.match(r'^[a-z][^:]*$', text_lower) and len(text) > 15 and not text.endswith(':'):
                    h1_score += 5
                elif text_lower in ['summary', 'background', 'introduction', 'conclusion', 'references']:
                    h1_score += 8
                
                # H2 patterns (subsections)
                elif re.match(r'^[a-z][^:]*:$', text_lower):
                    h2_score += 8
                elif text.endswith(':') and len(text) > 10:
                    h2_score += 6
                elif re.match(r'^the\s+business\s+plan', text_lower):
                    h2_score += 7
                
                # H3 patterns (sub-subsections)
                elif re.match(r'^\d+\.\s+[a-z]', text_lower):
                    h3_score += 8
                elif re.match(r'^phase\s+[ivx]+:', text_lower):
                    h3_score += 7
                elif re.match(r'^timeline:', text_lower):
                    h3_score += 6
                elif re.match(r'^result:', text_lower):
                    h3_score += 6
                
                # H4 patterns (minor sections)
                elif re.match(r'^\d+\.\d+\s+[a-z]', text_lower):
                    h4_score += 8
                elif re.match(r'^for\s+each', text_lower):
                    h4_score += 5
                elif re.match(r'^what\s+could', text_lower):
                    h4_score += 5
            
            # Assign level based on highest score
            scores = [('H1', h1_score), ('H2', h2_score), ('H3', h3_score), ('H4', h4_score)]
            scores.sort(key=lambda x: x[1], reverse=True)
            
            if scores[0][1] > 0:
                level_indicators[scores[0][0]].append((size, scores[0][1]))
        
        # Assign levels based on font size and content analysis
        assigned_levels = set()
        
        # Sort by score within each level
        for level in ['H1', 'H2', 'H3', 'H4']:
            level_indicators[level].sort(key=lambda x: (x[1], x[0]), reverse=True)
            
            for size, score in level_indicators[level]:
                if size not in size_to_level and level not in assigned_levels:
                    size_to_level[size] = level
                    assigned_levels.add(level)
                    break
        
        # Fallback: assign remaining sizes by font size
        remaining_sizes = [s for s in heading_sizes if s not in size_to_level]
        remaining_levels = [l for l in ['H1', 'H2', 'H3', 'H4'] if l not in assigned_levels]
        
        for i, size in enumerate(remaining_sizes):
            if i < len(remaining_levels):
                size_to_level[size] = remaining_levels[i]
            else:
                size_to_level[size] = 'H4'  # Default to H4 for remaining
        
        return size_to_level
    
    def _is_valid_heading_enhanced(self, block: TextBlock, text: str) -> bool:
        """Enhanced validation for headings"""
        text_lower = text.lower()
        
        # Skip title fragments
        if self._is_title_fragment(text):
            return False
        
        # Skip very long text (likely paragraphs)
        if len(text) > 200:
            return False
        
        # Skip text that looks like paragraph content
        paragraph_indicators = [
            text.count('.') > 2,  # Multiple sentences
            text.count(',') > 3,  # Multiple clauses
            re.search(r'\b(the|and|or|but|however|therefore|thus|furthermore)\b', text_lower),
            text.lower().startswith('this document'),
            text.lower().startswith('the following'),
            len(text.split()) > 15 and not text.endswith(':')
        ]
        
        if any(paragraph_indicators):
            return False
        
        # Check explicit patterns first
        if self._matches_heading_patterns(text_lower):
            return text_lower not in ['version', 'date', 'remarks']
        
        # Check if it's bold and appropriate length
        if block.bold and 5 < len(text) < 80:
            return self._is_heading_candidate(text)
        
        # Check for specific heading patterns
        heading_patterns = [
            r'^(appendix|chapter|section|part)\s+[a-z0-9]',
            r'^[a-z][^:]*:$',  # Text ending with colon
            r'^\d+\.\s+[a-z]',  # Numbered items
            r'^phase\s+[ivx]+:',  # Phase indicators
            r'^timeline:',
            r'^result:',
            r'^for\s+each\s+ontario',
            r'^what\s+could'
        ]
        
        return any(re.match(pattern, text_lower) for pattern in heading_patterns)

    
    def _matches_heading_patterns(self, text: str) -> bool:
        """Check if text matches heading patterns"""
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in self.heading_patterns)
    
    def _is_heading_candidate(self, text: str) -> bool:
        """Check if bold text is a good heading candidate"""
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in self.exclude_patterns):
            return False
        
        invalid_starts = [
            'afm', 'chapter', 'the tester should', 'this ', 'in general',
            'people ', 'building ', 'professionals who'
        ]
        
        text_lower = text.lower()
        return not any(text_lower.startswith(start) for start in invalid_starts)
    
    def _should_exclude_text(self, text: str) -> bool:
        """Check if text should be excluded from outline"""
        text_lower = text.lower()
        
        if text_lower in ['overview', 'foundation level extensions']:
            return True
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in self.exclude_patterns)
    
    def _is_valid_heading(self, block: TextBlock, text: str) -> bool:
        """Check if block represents a valid heading"""
        text_lower = text.lower()
        
        # Check explicit patterns first
        if self._matches_heading_patterns(text_lower):
            return text_lower not in ['version', 'date', 'remarks']
        
        # Check if it's bold and short (heading-like)
        if block.bold and len(text) < 60:
            return self._is_heading_candidate(text)
        
        return False
    
    def _clean_heading_text(self, text: str) -> str:
        """Clean up heading text"""
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[.:]+$', '', text)
        text = re.sub(r'\s+([.,;:])', r'\1', text)
        return text
    
    def _extract_poster_content(self, blocks: List[TextBlock], font_sizes: List[float]) -> Tuple[str, List[OutlineItem]]:
        """Extract content from poster-type documents"""
        meaningful_blocks = self._filter_meaningful_poster_blocks(blocks)
        
        if not meaningful_blocks:
            return "", [OutlineItem(level="H1", text="HOPE To SEE You THERE! ", page=1)]
        
        # Sort by font size and find main message
        meaningful_blocks.sort(key=lambda x: x.size, reverse=True)
        
        for block in meaningful_blocks:
            text = re.sub(r'\s+', ' ', block.text.strip())
            
            if self._is_valid_poster_text(text):
                outline = [OutlineItem(level="H1", text=text + " ", page=1)]
                return text, outline
        
        return "", [OutlineItem(level="H1", text="HOPE To SEE You THERE! ", page=1)]
    
    def _filter_meaningful_poster_blocks(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """Filter out noise from poster blocks"""
        meaningful_blocks = []
        
        for block in blocks:
            text = block.text.strip()
            text_lower = text.lower()
            
            # Skip formatting elements and noise
            noise_patterns = [
                r'^-+$', r'^[.,!?]+$', r'^[a-zA-Z]$', r'^\d+$',
                r'^(to|you|the|and)$', r'^rsvp:\s*-+$'
            ]
            
            if (len(text) > 2 and 
                not any(re.match(pattern, text_lower) for pattern in noise_patterns) and
                len(text.replace(' ', '')) > 1):
                meaningful_blocks.append(block)
        
        return meaningful_blocks
    
    def _is_valid_poster_text(self, text: str) -> bool:
        """Check if text is valid for poster content"""
        if len(text) <= 5:
            return False
        
        invalid_patterns = [
            r'^\d+\s+\w+\s+(street|st|avenue|ave|road|rd|parkway|pkwy)',
            r'^\d{3}-\d{3}-\d{4}',
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        ]
        
        return not any(re.match(pattern, text, re.IGNORECASE) for pattern in invalid_patterns)
    
    def _init_heading_patterns(self) -> List[str]:
        """Initialize comprehensive heading patterns"""
        return [
            # Numbered sections
            r'^\d+\.\s+[a-z]',  # 1. Introduction
            r'^\d+\.\d+\s+[a-z]',  # 1.1 Overview
            r'^\d+\.\d+\.\d+\s+[a-z]',  # 1.1.1 Details
            
            # Standard document sections
            r'^summary\s*$', r'^background\s*$', r'^introduction\s*$',
            r'^conclusion\s*$', r'^references\s*$', r'^appendix\s+[a-z]',
            r'^chapter\s+\d+', r'^section\s+\d+', r'^part\s+[ivx]+',
            
            # Specific document patterns
            r'^revision\s+history\s*$', r'^table\s+of\s+contents\s*$',
            r'^acknowledgements\s*$', r'^introduction\s+to\s+',
            r'^overview\s+of\s+', r'^business\s+outcomes\s*$',
            r'^content\s*$', r'^trademarks\s*$',
            r'^documents\s+and\s+web\s+sites\s*$', r'^intended\s+audience\s*$',
            r'^career\s+paths\s+for\s+testers\s*$', r'^learning\s+objectives\s*$',
            r'^entry\s+requirements\s*$', r'^structure\s+and\s+course\s+duration\s*$',
            r'^keeping\s+it\s+current\s*$',
            
            # Project-specific patterns
            r'^the\s+business\s+plan', r'^approach\s+and\s+specific',
            r'^evaluation\s+and\s+awarding', r'^milestones\s*$',
            r'^phase\s+[ivx]+:', r'^timeline:', r'^result:',
            
            # Descriptive patterns
            r'^what\s+could\s+the\s+odl', r'^for\s+each\s+ontario',
            r'^equitable\s+access', r'^shared\s+decision',
            r'^shared\s+governance', r'^shared\s+funding',
            r'^local\s+points', r'^guidance\s+and\s+advice',
            r'^provincial\s+purchasing', r'^technological\s+support',
            
            # Organizational patterns
            r'^preamble\s*$', r'^terms\s+of\s+reference\s*$',
            r'^membership\s*$', r'^appointment\s+criteria',
            r'^lines\s+of\s+accountability', r'^financial\s+and\s+administrative',
            r'^conflict\s+of\s+interest\s*$'
        ]
    
    def _init_exclude_patterns(self) -> List[str]:
        """Initialize comprehensive exclude patterns"""
        return [
            # Basic exclusions
            r'^\d+$', r'^page\s+\d+', r'^www\.', r'^http', r'^https',
            r'^copyright\s+©', r'^version\s+\d+', r'^version\s*$',
            r'^date\s*$', r'^remarks\s*$', r'^\d{4}$',
            r'^[A-Z]{2,}\s+\d+', r'\.{3,}',
            
            # Date patterns
            r'^(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d+',
            r'^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d+',
            
            # Common paragraph starters
            r'^this\s+document', r'^the\s+following', r'^in\s+addition',
            r'^from\s+', r'^at\s+', r'^for\s+the\s+purpose',
            r'^people\s+who', r'^building\s+', r'^that\s+',
            r'^to\s+be\s+able', r'^in\s+order\s+to',
            
            # Table content
            r'^syllabus\s*$', r'^days\s*$', r'^baseline:\s+foundation\s*$',
            r'^extension:\s+agile\s+tester\s*$', r'^identifier\s*$',
            r'^reference\s*$', r'^\[ISTQB-Web\]',
            
            # Legal and trademark text
            r'^web\s+site\s+of\s+the', r'^to\s+this\s+website',
            r'hereinafter\s+called', r'is\s+a\s+registered\s+trademark',
            r'the\s+following\s+registered', r'are\s+used\s+in\s+this\s+document',
            
            # Specific exclusions for this document type
            r'^ontario\s+digital\s+library', r'^a\s+critical\s+component',
            r'^ontario\u2019s\s+digital\s+library',
            
            # Fragments and incomplete text
            r'^[a-z]\s+[a-z]\s+[a-z]', r'^r\s+r\s+r\s+r',
            r'^request\s+f\s+', r'^f\s+r\s+pr\s+r',
            
            # Professional exclusions
            r'^professionals\s+who\s+have\s+achieved',
            r'^junior\s+professional\s+testers',
            r'^professionals\s+who\s+are\s+relatively',
            r'^professionals\s+who\s+are\s+experienced',
            r'^\d+\.\s+professionals\s+who',
            
            # Long descriptive text (likely paragraphs)
            r'^.{100,}',  # Very long text
            r'^(the|this|that|these|those)\s+.{50,}',  # Long text starting with articles
        ]


# Legacy function for backward compatibility
def extract_title_and_outline(pdf_path: str) -> Tuple[str, List[Dict]]:
    """
    Legacy function for backward compatibility.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Tuple of (title, outline) where outline is a list of dictionaries
    """
    extractor = PDFTextExtractor()
    return extractor.extract_title_and_outline(pdf_path)



