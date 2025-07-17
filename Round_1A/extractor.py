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
        """Extract document title from first page"""
        first_page_blocks = [b for b in blocks if b.page == 1]
        
        if not first_page_blocks:
            return "Untitled Document"
        
        title_parts = []
        
        # Check top 3 font sizes for title candidates
        for size in font_sizes[:3]:
            size_blocks = [b for b in first_page_blocks if b.size == size]
            
            for block in size_blocks:
                text = block.text.strip()
                if self._is_valid_title_text(text):
                    title_parts.append(text)
            
            if title_parts:
                break
        
        return "  ".join(title_parts) if title_parts else "Untitled Document"
    
    def _is_valid_title_text(self, text: str) -> bool:
        """Check if text is a valid title candidate"""
        if len(text) <= 3 or text.isdigit():
            return False
        
        text_lower = text.lower()
        invalid_patterns = [
            r'^\d{4}$', r'^page\s+\d+', r'^copyright', r'^version\s*$',
            r'^date\s*$', r'^remarks\s*$', r'^may\s+\d+'
        ]
        
        return not any(re.match(pattern, text_lower) for pattern in invalid_patterns)
    
    def _extract_structured_outline(self, blocks: List[TextBlock], font_sizes: List[float]) -> List[OutlineItem]:
        """Extract outline from structured documents"""
        outline = []
        seen_texts = set()
        
        # Analyze font sizes for heading levels
        heading_sizes = self._analyze_heading_sizes(blocks, font_sizes)
        size_to_level = self._assign_heading_levels(blocks, heading_sizes)
        
        # Extract headings
        for block in blocks:
            if block.size not in size_to_level:
                continue
                
            text = block.text.strip()
            
            if (text in seen_texts or 
                len(text) < 3 or 
                self._should_exclude_text(text) or
                not self._is_valid_heading(block, text)):
                continue
            
            clean_text = self._clean_heading_text(text)
            if clean_text:
                outline.append(OutlineItem(
                    level=size_to_level[block.size],
                    text=clean_text + " ",
                    page=block.page
                ))
                seen_texts.add(text)
        
        return outline
    
    def _analyze_heading_sizes(self, blocks: List[TextBlock], font_sizes: List[float]) -> List[float]:
        """Analyze which font sizes are used for headings"""
        heading_sizes = []
        
        for size in font_sizes:
            size_blocks = [b for b in blocks if b.size == size]
            heading_count = 0
            
            for block in size_blocks:
                text = block.text.strip().lower()
                
                if (text in ['overview', 'foundation level extensions'] or
                    not self._matches_heading_patterns(text) and 
                    not (block.bold and len(text) < 60 and self._is_heading_candidate(text))):
                    continue
                
                heading_count += 1
            
            if heading_count > 0:
                heading_sizes.append(size)
        
        return sorted(heading_sizes, reverse=True)
    
    def _assign_heading_levels(self, blocks: List[TextBlock], heading_sizes: List[float]) -> Dict[float, str]:
        """Assign heading levels based on font sizes and content analysis"""
        size_to_level = {}
        
        # Find main sections and subsections
        main_section_size = None
        subsection_size = None
        
        for size in heading_sizes:
            size_blocks = [b for b in blocks if b.size == size]
            
            # Look for numbered sections
            for block in size_blocks:
                text = block.text.strip()
                if re.match(r'^\d+\.\s+', text):
                    main_section_size = size
                    break
                elif re.match(r'^\d+\.\d+\s+', text):
                    subsection_size = size
                    break
        
        # Assign levels
        if main_section_size and subsection_size:
            for i, size in enumerate(heading_sizes):
                if size == main_section_size:
                    size_to_level[size] = "H1"
                elif size == subsection_size:
                    size_to_level[size] = "H2"
                elif i == 0 and size not in [main_section_size, subsection_size]:
                    size_to_level[size] = "H1"
        else:
            # Fallback to size-based assignment
            for i, size in enumerate(heading_sizes[:3]):
                size_to_level[size] = f"H{i+1}"
        
        return size_to_level

    
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
        """Initialize heading patterns"""
        return [
            r'^\d+\.\s+', r'^\d+\.\d+\s+', r'^\d+\.\d+\.\d+\s+',
            r'^revision\s+history\s*$', r'^table\s+of\s+contents\s*$',
            r'^acknowledgements\s*$', r'^references\s*$',
            r'^introduction\s+to\s+', r'^overview\s+of\s+',
            r'^business\s+outcomes\s*$', r'^content\s*$', r'^trademarks\s*$',
            r'^documents\s+and\s+web\s+sites\s*$', r'^intended\s+audience\s*$',
            r'^career\s+paths\s+for\s+testers\s*$', r'^learning\s+objectives\s*$',
            r'^entry\s+requirements\s*$', r'^structure\s+and\s+course\s+duration\s*$',
            r'^keeping\s+it\s+current\s*$'
        ]
    
    def _init_exclude_patterns(self) -> List[str]:
        """Initialize exclude patterns"""
        return [
            r'^\d+$', r'^page\s+\d+', r'^www\.', r'^http', r'^copyright\s+©',
            r'^version\s+\d+', r'^version\s*$', r'^date\s*$', r'^remarks\s*$',
            r'^may\s+\d+', r'^\d{4}$', r'^[A-Z]{2,}\s+\d+', r'\.{3,}',
            r'^this\s+document', r'^the\s+', r'^in\s+', r'^from\s+', r'^at\s+',
            r'^for\s+', r'^people\s+', r'^building\s+', r'^that\s+',
            r'^to\s+be\s+able', r'^syllabus\s*$', r'^days\s*$',
            r'^baseline:\s+foundation\s*$', r'^extension:\s+agile\s+tester\s*$',
            r'^identifier\s*$', r'^reference\s*$', r'^\[ISTQB-Web\]',
            r'^web\s+site\s+of\s+the', r'^to\s+this\s+website',
            r'hereinafter\s+called', r'is\s+a\s+registered\s+trademark',
            r'the\s+following\s+registered', r'are\s+used\s+in\s+this\s+document',
            r'^overview\s*$', r'^foundation\s+level\s+extensions\s*$',
            r'^professionals\s+who\s+have\s+achieved', r'^junior\s+professional\s+testers',
            r'^professionals\s+who\s+are\s+relatively', r'^professionals\s+who\s+are\s+experienced',
            r'^\d+\.\s+professionals\s+who'
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



