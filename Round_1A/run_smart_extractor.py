# # # run_smart_extractor.py - Simple runner for the Smart PDF Extractor

# # import os
# # import sys
# # from pathlib import Path

# # # Add current directory to path
# # sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# # from smart_pdf_extractor import process_all_pdfs, process_single_pdf

# # def main():
# #     """Main runner function"""
# #     print("🚀 Smart PDF Extractor")
# #     print("=" * 50)
    
# #     # Check if input directory exists
# #     input_dir = "app/inputs"
# #     output_dir = "app/outputs"
    
# #     if not os.path.exists(input_dir):
# #         print(f"❌ Input directory not found: {input_dir}")
# #         print("Please create the directory and add your PDF files.")
# #         return
    
# #     # Create output directory
# #     os.makedirs(output_dir, exist_ok=True)
    
# #     # Check for PDF files
# #     pdf_files = list(Path(input_dir).glob("*.pdf"))
    
# #     if not pdf_files:
# #         print(f"⚠️ No PDF files found in {input_dir}")
# #         print("Please add PDF files to the input directory.")
# #         return
    
# #     print(f"📂 Input Directory: {input_dir}")
# #     print(f"📁 Output Directory: {output_dir}")
# #     print(f"📄 Found {len(pdf_files)} PDF files:")
    
# #     for pdf_file in pdf_files:
# #         print(f"   - {pdf_file.name}")
    
# #     print("\n🔧 Available extraction methods:")
# #     print("   - Basic Rule-Based (always available)")
# #     print("   - NLP-Enhanced (requires spaCy)")
# #     print("   - ML-Enhanced (requires transformers)")
    
# #     # Check library availability
# #     try:
# #         import spacy
# #         print("   ✅ SpaCy available")
# #     except ImportError:
# #         print("   ❌ SpaCy not available")
    
# #     try:
# #         import transformers
# #         print("   ✅ Transformers available")
# #     except ImportError:
# #         print("   ❌ Transformers not available")
    
# #     print("\n🚀 Starting extraction process...")
# #     print("The system will automatically select the best method for each PDF.\n")
    
# #     # Process all PDFs
# #     process_all_pdfs(input_dir, output_dir)
    
# #     print(f"\n📁 Check the results in: {output_dir}")

# # if __name__ == "__main__":
# #     main()


# # smart_pdf_extractor.py - Intelligent PDF Extractor with Automatic Method Selection (v3)

# import fitz  # PyMuPDF
# from PyPDF2 import PdfReader
# import re
# import os
# import json
# from typing import List, Dict, Any, Optional, Tuple
# import logging
# from pathlib import Path
# import time

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Try to import advanced libraries
# try:
#     from transformers import pipeline
#     TRANSFORMERS_AVAILABLE = True
#     logger.info("✅ Transformers library available")
# except ImportError:
#     TRANSFORMERS_AVAILABLE = False
#     logger.info("❌ Transformers not available")

# try:
#     import spacy
#     SPACY_AVAILABLE = True
#     logger.info("✅ SpaCy library available")
# except ImportError:
#     SPACY_AVAILABLE = False
#     logger.info("❌ SpaCy not available")

# class PDFAnalyzer:
#     """Analyzes PDF characteristics to determine the best extraction method"""

#     def __init__(self):
#         self.method_thresholds = {
#             'simple': 0.3,      # Use basic extractor
#             'nlp': 0.6,         # Use NLP-enhanced extractor
#             'ml': 0.8           # Use ML-based extractor
#         }

#     def analyze_pdf(self, pdf_path: str) -> Dict[str, Any]:
#         """Comprehensive PDF analysis to determine extraction strategy"""
#         try:
#             doc = fitz.open(pdf_path)

#             analysis = {
#                 'total_pages': len(doc),
#                 'has_toc': bool(doc.get_toc()),
#                 'complexity_score': 0.0,
#                 'structure_type': 'simple',
#                 'recommended_method': 'basic',
#                 'characteristics': {
#                     'font_variations': 0,
#                     'text_density': 0,
#                     'formatting_complexity': 0,
#                     'is_academic': False,
#                     'is_presentation': False,
#                     'is_technical': False,
#                     'has_complex_structure': False
#                 }
#             }

#             # Analyze structure complexity
#             font_variations = set()
#             total_text_length = 0
#             heading_patterns = 0
#             technical_terms = 0

#             # Sample first 5 pages for analysis
#             pages_to_analyze = min(5, len(doc))

#             for i in range(pages_to_analyze):
#                 page = doc[i]
#                 page_dict = page.get_text("dict")
#                 page_text = page.get_text()

#                 # Count text and fonts
#                 for block in page_dict.get("blocks", []):
#                     if "lines" not in block:
#                         continue

#                     for line in block["lines"]:
#                         if "spans" not in line:
#                             continue

#                         for span in line["spans"]:
#                             text = span.get("text", "").strip()
#                             if text:
#                                 total_text_length += len(text)
#                                 font_key = (
#                                     round(span.get("size", 12), 1),
#                                     span.get("font", ""),
#                                     span.get("flags", 0) & 16 > 0  # Bold flag
#                                 )
#                                 font_variations.add(font_key)

#                 # Detect heading patterns
#                 heading_patterns += len(re.findall(r'^(Chapter|Section|Part|\d+\.)', page_text, re.MULTILINE | re.IGNORECASE))

#                 # Detect technical content
#                 technical_keywords = ['algorithm', 'method', 'approach', 'model', 'framework', 'analysis', 'evaluation', 'implementation']
#                 technical_terms += sum(1 for keyword in technical_keywords if keyword.lower() in page_text.lower())

#             # Calculate complexity metrics
#             analysis['characteristics']['font_variations'] = len(font_variations)
#             analysis['characteristics']['text_density'] = total_text_length / pages_to_analyze if pages_to_analyze > 0 else 0
#             analysis['characteristics']['formatting_complexity'] = len(font_variations) / max(1, pages_to_analyze)

#             # Determine document type
#             if technical_terms > 5:
#                 analysis['characteristics']['is_technical'] = True

#             if 'presentation' in pdf_path.lower() or analysis['characteristics']['text_density'] < 500:
#                 analysis['characteristics']['is_presentation'] = True

#             if any(keyword in pdf_path.lower() for keyword in ['research', 'paper', 'journal', 'academic']):
#                 analysis['characteristics']['is_academic'] = True

#             if heading_patterns > 3 or analysis['characteristics']['font_variations'] > 8:
#                 analysis['characteristics']['has_complex_structure'] = True

#             # Calculate overall complexity score
#             complexity_score = 0.0

#             # Page count factor (0-0.2)
#             if analysis['total_pages'] > 50:
#                 complexity_score += 0.2
#             elif analysis['total_pages'] > 20:
#                 complexity_score += 0.15
#             elif analysis['total_pages'] > 10:
#                 complexity_score += 0.1

#             # TOC factor (0-0.15)
#             if analysis['has_toc']:
#                 complexity_score += 0.15

#             # Font variation factor (0-0.25)
#             if analysis['characteristics']['font_variations'] > 10:
#                 complexity_score += 0.25
#             elif analysis['characteristics']['font_variations'] > 5:
#                 complexity_score += 0.15
#             elif analysis['characteristics']['font_variations'] > 3:
#                 complexity_score += 0.1

#             # Structure complexity factor (0-0.2)
#             if analysis['characteristics']['has_complex_structure']:
#                 complexity_score += 0.2

#             # Content type factor (0-0.2)
#             if analysis['characteristics']['is_technical'] or analysis['characteristics']['is_academic']:
#                 complexity_score += 0.15

#             if analysis['characteristics']['is_presentation']:
#                 complexity_score += 0.1

#             analysis['complexity_score'] = min(complexity_score, 1.0)

#             # Determine recommended method
#             if analysis['complexity_score'] >= self.method_thresholds['ml'] and (TRANSFORMERS_AVAILABLE or SPACY_AVAILABLE):
#                 analysis['recommended_method'] = 'ml'
#                 analysis['structure_type'] = 'complex'
#             elif analysis['complexity_score'] >= self.method_thresholds['nlp']:
#                 analysis['recommended_method'] = 'nlp'
#                 analysis['structure_type'] = 'moderate'
#             else:
#                 analysis['recommended_method'] = 'basic'
#                 analysis['structure_type'] = 'simple'

#             doc.close()
#             return analysis

#         except Exception as e:
#             logger.error(f"Error analyzing PDF {pdf_path}: {e}")
#             return {
#                 'total_pages': 0,
#                 'has_toc': False,
#                 'complexity_score': 0.0,
#                 'structure_type': 'simple',
#                 'recommended_method': 'basic',
#                 'characteristics': {}
#             }

# class BasicPDFExtractor:
#     """Basic rule-based PDF extractor for simple documents"""

#     def extract_title_and_outline(self, pdf_path: str) -> Tuple[str, List[Dict]]:
#         """Extract title and outline using basic rules"""
#         try:
#             title = self._extract_title(pdf_path)
#             outline = self._extract_outline(pdf_path)

#             # Remove title from outline
#             outline = self._filter_title_from_outline(title, outline)

#             return title, outline

#         except Exception as e:
#             logger.error(f"Basic extraction failed for {pdf_path}: {e}")
#             return "Untitled Document", []

#     def _extract_title(self, pdf_path: str) -> str:
#         """Extract title from first page"""
#         try:
#             doc = fitz.open(pdf_path)
#             if len(doc) == 0:
#                 return "Untitled Document"

#             page1 = doc[0]
#             # --- NEW: Use get_text("blocks") to get more detailed structure ---
#             blocks = page1.get_text("blocks")
            
#             title_candidates = []
            
#             # Combine nearby text blocks to form title candidates
#             for i, block in enumerate(blocks):
#                 text = block[4].strip().replace('\n', ' ')
#                 if not text:
#                     continue
                
#                 # Simple heuristics for titles
#                 if block[1] < 200: # y-position check
#                     title_candidates.append({
#                         "text": text,
#                         "y_position": block[1],
#                         "word_count": len(text.split())
#                     })

#             # Find best title candidate
#             if title_candidates:
#                 # Often the first significant piece of text is the title
#                 title_candidates.sort(key=lambda x: x["y_position"])
#                 for candidate in title_candidates:
#                     if 2 <= candidate["word_count"] <= 20:
#                         return candidate["text"]
            
#             # Fallback to metadata
#             try:
#                 reader = PdfReader(pdf_path)
#                 if reader.metadata and reader.metadata.title:
#                     return reader.metadata.title.strip()
#             except:
#                 pass
            
#             doc.close()
#             # If no good candidate found, return a default
#             return title_candidates[0]['text'] if title_candidates else "Untitled Document"

#         except Exception as e:
#             logger.error(f"Title extraction failed: {e}")
#             return "Untitled Document"

#     def _extract_outline(self, pdf_path: str) -> List[Dict]:
#         """Extract outline using a robust line grouping method"""
#         try:
#             doc = fitz.open(pdf_path)
#             outline = []

#             # Try TOC first - this is the most reliable method
#             toc = doc.get_toc(simple=False)
#             if toc:
#                 for level, text, page, _ in toc:
#                     outline.append({
#                         "level": f"H{level}",
#                         "text": text.strip(),
#                         "page": page
#                     })
#                 if outline:
#                     doc.close()
#                     return self._clean_outline(outline)

#             # --- RE-ENGINEERED VISUAL EXTRACTION ---
#             font_sizes = []
            
#             # 1. Get all text blocks with detailed info
#             all_blocks = []
#             for i in range(len(doc)):
#                 page_blocks = doc[i].get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_LIGATURES)["blocks"]
#                 for b in page_blocks:
#                     for l in b.get("lines", []):
#                         for s in l.get("spans", []):
#                             font_sizes.append(s['size'])
#                             all_blocks.append({
#                                 "text": s['text'].strip(),
#                                 "size": s['size'],
#                                 "font": s['font'],
#                                 "bold": "bold" in s['font'].lower(),
#                                 "bbox": s['bbox'],
#                                 "page": i + 1
#                             })
            
#             if not all_blocks:
#                 doc.close()
#                 return []

#             # 2. Calculate average font size for context
#             avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12

#             # 3. Group adjacent blocks into coherent lines
#             grouped_lines = self._group_text_blocks(all_blocks)

#             # 4. Identify headings from the clean, grouped lines
#             for line_info in grouped_lines:
#                 if self._is_likely_heading(line_info, avg_font_size):
#                     level = self._determine_heading_level(line_info, avg_font_size)
#                     outline.append({
#                         "level": level,
#                         "text": line_info["text"],
#                         "page": line_info["page"]
#                     })
            
#             doc.close()
#             return self._clean_outline(outline)

#         except Exception as e:
#             logger.error(f"Outline extraction failed: {e}")
#             return []

#     def _group_text_blocks(self, blocks: List[Dict], y_threshold: float = 5.0) -> List[Dict]:
#         """Groups nearby text blocks into single logical lines."""
#         if not blocks:
#             return []

#         grouped = []
#         current_line = blocks[0]

#         for i in range(1, len(blocks)):
#             prev = blocks[i - 1]
#             curr = blocks[i]

#             # Check if blocks are on the same line and from the same page
#             same_line = abs(prev['bbox'][1] - curr['bbox'][1]) < y_threshold
#             same_page = prev['page'] == curr['page']

#             if same_line and same_page:
#                 # Merge current block with the previous one
#                 current_line['text'] += " " + curr['text']
#                 # Update bbox to cover both
#                 current_line['bbox'] = (
#                     min(current_line['bbox'][0], curr['bbox'][0]),
#                     min(current_line['bbox'][1], curr['bbox'][1]),
#                     max(current_line['bbox'][2], curr['bbox'][2]),
#                     max(current_line['bbox'][3], curr['bbox'][3]),
#                 )
#                 current_line['size'] = max(current_line['size'], curr['size'])
#             else:
#                 # End of the current line, start a new one
#                 grouped.append(current_line)
#                 current_line = curr
        
#         grouped.append(current_line) # Add the last line
        
#         # Second pass to merge multi-line headings
#         final_lines = []
#         if not grouped:
#             return []
            
#         multi_line_heading = grouped[0]
#         for i in range(1, len(grouped)):
#             prev = grouped[i-1]
#             curr = grouped[i]
#             # If current line starts with lowercase, likely part of previous
#             if curr['text'].strip() and curr['text'].strip()[0].islower() and prev['page'] == curr['page']:
#                  multi_line_heading['text'] += " " + curr['text']
#             else:
#                 final_lines.append(multi_line_heading)
#                 multi_line_heading = curr
#         final_lines.append(multi_line_heading)


#         return final_lines


#     def _is_likely_heading(self, line: Dict, avg_font_size: float) -> bool:
#         """Determine if a line of text is likely a heading"""
#         text = line["text"].strip()
#         lower_text = text.lower()

#         # --- Filter out common non-informative table headers and metadata ---
#         NON_INFORMATIVE_KEYWORDS = {
#             'version', 'date', 'remarks', 'author', 'status', 'document id',
#             'confidential', 'proprietary', 'page', 'revision history'
#         }

#         if lower_text in NON_INFORMATIVE_KEYWORDS:
#             return False

#         # --- Filter out lines that are just version numbers, dates, or noise ---
#         if re.fullmatch(r'v?\d+(\.\d+)*', lower_text):
#             return False
#         if re.fullmatch(r'(\d{1,2}\s+)?(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[\w\s,]*\d{4}', lower_text, re.IGNORECASE):
#             return False
#         if re.fullmatch(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', lower_text):
#             return False

#         if len(text) < 2 or len(text) > 200:
#             return False

#         # Filter out lines that are likely part of a paragraph
#         if text.endswith('.') or text.endswith(','):
#              if len(text.split()) > 15: # Long sentence ending with punctuation
#                 return False

#         score = 0

#         # Font size factor (more selective)
#         size_ratio = line["size"] / avg_font_size
#         if size_ratio > 1.15:
#             score += 3
#         elif size_ratio > 1.05:
#             score += 2

#         # Bold text
#         if line["bold"]:
#             score += 2
        
#         # All caps text
#         if text.isupper() and len(text.split()) > 1:
#             score += 2

#         # Length factor (short lines are better)
#         word_count = len(text.split())
#         if 1 <= word_count <= 15:
#             score += 1
#         elif word_count > 25:
#             score -= 2

#         # Pattern matching (starts with number, chapter, etc.)
#         if re.match(r'^(Chapter|Section|Part|Appendix)\s*\d*|(\d+(\.\d+)*)\s+[A-Z]', text, re.IGNORECASE):
#             score += 3

#         # Colon at the end is a good indicator
#         if text.endswith(':'):
#             score += 2
            
#         # --- Increased threshold to be more selective ---
#         return score >= 5

#     def _determine_heading_level(self, line: Dict, avg_font_size: float) -> str:
#         """Determine heading level based on formatting"""
#         size_ratio = line["size"] / avg_font_size

#         if size_ratio > 1.4 or (size_ratio > 1.2 and line["bold"]):
#             return "H1"
#         elif size_ratio > 1.2 or (size_ratio > 1.1 and line["bold"]):
#             return "H2"
#         elif size_ratio > 1.05 or line["bold"]:
#             return "H3"
#         else:
#             return "H4"

#     def _clean_outline(self, outline: List[Dict]) -> List[Dict]:
#         """Clean and deduplicate the final outline."""
#         seen = set()
#         cleaned = []
        
#         for item in outline:
#             text = item["text"].strip()
#             # Clean up extra spaces that might result from merging
#             text = re.sub(r'\s+', ' ', text)
            
#             key = (text.lower(), item["page"])

#             if key not in seen and len(text.split()) > 1:
#                 seen.add(key)
#                 item['text'] = text
#                 cleaned.append(item)

#         return cleaned

#     def _filter_title_from_outline(self, title: str, outline: List[Dict]) -> List[Dict]:
#         """Remove title components from outline"""
#         if not title or title == "Untitled Document":
#             return outline

#         title_words = set(word.lower().strip() for word in title.split() if len(word.strip()) > 2)
#         filtered = []

#         for item in outline:
#             text_words = set(word.lower().strip() for word in item["text"].split() if len(word.strip()) > 2)

#             # Skip if too similar to title and on the first page
#             if text_words and title_words and item['page'] <= 2:
#                 overlap = len(text_words.intersection(title_words))
#                 # Check for high overlap with title
#                 if overlap / max(len(text_words), 1) > 0.85:
#                     continue # Skip this item as it's too similar to title
            
#             filtered.append(item)

#         return filtered


# class NLPEnhancedExtractor(BasicPDFExtractor):
#     """NLP-enhanced extractor for moderate complexity documents"""

#     def __init__(self):
#         super().__init__()
#         self.nlp = None
#         if SPACY_AVAILABLE:
#             try:
#                 import spacy
#                 self.nlp = spacy.load("en_core_web_sm")
#                 logger.info("✅ SpaCy model loaded successfully")
#             except:
#                 try:
#                     self.nlp = spacy.blank("en")
#                     logger.info("✅ SpaCy blank model loaded")
#                 except:
#                     logger.warning("❌ Failed to load SpaCy models")

#     def extract_title_and_outline(self, pdf_path: str) -> Tuple[str, List[Dict]]:
#         """Extract using NLP-enhanced techniques"""
#         try:
#             title = self._extract_title_nlp(pdf_path)
#             outline = self._extract_outline_nlp(pdf_path)

#             # Enhanced filtering
#             outline = self._filter_title_from_outline(title, outline)
#             outline = self._enhance_outline_with_nlp(outline)

#             return title, outline

#         except Exception as e:
#             logger.error(f"NLP extraction failed for {pdf_path}: {e}")
#             # Fallback to basic method
#             return super().extract_title_and_outline(pdf_path)

#     def _extract_title_nlp(self, pdf_path: str) -> str:
#         """Enhanced title extraction with NLP"""
#         # Start with basic extraction
#         title = self._extract_title(pdf_path)

#         if self.nlp and title != "Untitled Document":
#             # Use NLP to refine title
#             doc = self.nlp(title)

#             # Remove common non-title elements
#             filtered_tokens = []
#             for token in doc:
#                 if not token.is_stop and not token.is_punct and token.pos_ not in ['DET']:
#                     filtered_tokens.append(token.text)

#             if filtered_tokens:
#                 refined_title = " ".join(filtered_tokens)
#                 if len(refined_title) > 3:
#                     return refined_title

#         return title

#     def _extract_outline_nlp(self, pdf_path: str) -> List[Dict]:
#         """Enhanced outline extraction with NLP analysis"""
#         # Get basic outline
#         outline = self._extract_outline(pdf_path)

#         if not self.nlp or not outline:
#             return outline

#         # Enhance with NLP
#         enhanced_outline = []

#         for item in outline:
#             text = item["text"]

#             # Analyze with spaCy
#             doc = self.nlp(text)

#             # Calculate NLP-based confidence
#             confidence = self._calculate_nlp_confidence(doc, text)

#             # Update item - but don't add method/confidence to final output
#             if confidence > 0.3:  # Threshold for keeping items
#                 enhanced_outline.append(item)

#         return enhanced_outline

#     def _calculate_nlp_confidence(self, doc, text: str) -> float:
#         """Calculate confidence using NLP features"""
#         confidence = 0.5  # Base confidence

#         # Named entity recognition
#         if doc.ents:
#             confidence += 0.1

#         # POS tag analysis
#         noun_count = sum(1 for token in doc if token.pos_ in ['NOUN', 'PROPN'])
#         verb_count = sum(1 for token in doc if token.pos_ == 'VERB')

#         if noun_count > verb_count:  # Headings are usually noun-heavy
#             confidence += 0.2

#         # Length factor
#         if 2 <= len(doc) <= 10:
#             confidence += 0.2
#         elif len(doc) > 15:
#             confidence -= 0.3

#         # Capitalization
#         if text.istitle():
#             confidence += 0.1

#         return min(max(confidence, 0.0), 1.0)

#     def _enhance_outline_with_nlp(self, outline: List[Dict]) -> List[Dict]:
#         """Further enhance outline using NLP techniques"""
#         if not self.nlp:
#             return outline

#         # Group similar headings
#         grouped_outline = []
#         seen_concepts = set()

#         for item in outline:
#             text = item["text"]
#             doc = self.nlp(text)

#             # Extract key concepts
#             key_concepts = set()
#             for token in doc:
#                 if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop:
#                     key_concepts.add(token.lemma_.lower())

#             # Check if similar concept already exists
#             is_duplicate = False
#             for seen in seen_concepts:
#                 if len(key_concepts.intersection(seen)) / max(len(key_concepts), len(seen), 1) > 0.7:
#                     is_duplicate = True
#                     break

#             if not is_duplicate:
#                 seen_concepts.add(frozenset(key_concepts))
#                 grouped_outline.append(item)

#         return grouped_outline

# class MLEnhancedExtractor(NLPEnhancedExtractor):
#     """ML-enhanced extractor for complex documents"""

#     def __init__(self):
#         super().__init__()
#         self.classifier = None

#         if TRANSFORMERS_AVAILABLE:
#             try:
#                 # Use a lightweight model for heading classification
#                 self.classifier = pipeline(
#                     "text-classification",
#                     model="distilbert-base-uncased-finetuned-sst-2-english", # A better choice for classification
#                     device=-1  # Use CPU
#                 )
#                 logger.info("✅ ML classifier loaded")
#             except:
#                 logger.warning("❌ Failed to load ML classifier, ML features will be simulated.")

#     def extract_title_and_outline(self, pdf_path: str) -> Tuple[str, List[Dict]]:
#         """Extract using ML-enhanced techniques"""
#         try:
#             title = self._extract_title_ml(pdf_path)
#             outline = self._extract_outline_ml(pdf_path)

#             # ML-enhanced filtering
#             outline = self._filter_title_from_outline(title, outline)
#             # NLP enhancement is already good, no need for extra ML enhancement step
#             # outline = self._enhance_outline_with_ml(outline)

#             return title, outline

#         except Exception as e:
#             logger.error(f"ML extraction failed for {pdf_path}: {e}")
#             # Fallback to NLP method
#             return super().extract_title_and_outline(pdf_path)

#     def _extract_title_ml(self, pdf_path: str) -> str:
#         """ML-enhanced title extraction"""
#         # Start with NLP extraction
#         title = self._extract_title_nlp(pdf_path)

#         if self.classifier and title != "Untitled Document":
#             try:
#                 # Use ML to validate title quality
#                 # This is a proxy task. A real model would be trained on title quality.
#                 result = self.classifier(title)
#                 if result and len(result) > 0:
#                     # Simple validation based on classification confidence
#                     confidence = result[0].get('score', 0.5)
#                     if confidence < 0.6:  # Low confidence, might not be a good title
#                         pass # Could implement title refinement here
#             except:
#                 pass  # Fallback to original title

#         return title

#     def _extract_outline_ml(self, pdf_path: str) -> List[Dict]:
#         """ML-enhanced outline extraction"""
#         # Get NLP-enhanced outline
#         outline = self._extract_outline_nlp(pdf_path)

#         if not self.classifier:
#             return outline

#         # Enhance with ML classification
#         ml_enhanced_outline = []

#         for item in outline:
#             text = item["text"]

#             try:
#                 # Classify text as heading vs non-heading
#                 ml_confidence = self._classify_heading_ml(text)

#                 # Combine confidence scores (heuristic)
#                 nlp_conf = item.get("confidence", 0.5) if hasattr(item, 'get') and 'confidence' in str(item) else 0.5
#                 combined_confidence = 0.7 * ml_confidence + 0.3 * nlp_conf # Weight ML higher

#                 if combined_confidence > 0.5:
#                     ml_enhanced_outline.append(item)

#             except Exception as e:
#                 # Fallback - just add the item
#                 ml_enhanced_outline.append(item)

#         return ml_enhanced_outline

#     def _classify_heading_ml(self, text: str) -> float:
#         """Use ML to classify if text is a heading"""
#         if not self.classifier:
#             return 0.5 # Default confidence if no classifier

#         try:
#             # A real model would be trained on (text, is_heading) data.
#             # We simulate this with the sentiment classifier: positive sentiment as a proxy for a confident heading.
#             result = self.classifier(text)[0]
            
#             if result['label'] == 'POSITIVE':
#                 return result['score']
#             else: # NEGATIVE
#                 return 1.0 - result['score']

#         except Exception as e:
#             logger.error(f"ML classification failed: {e}")
#             return 0.5

# class SmartPDFExtractor:
#     """Main extractor that automatically selects the best method"""

#     def __init__(self):
#         self.analyzer = PDFAnalyzer()
#         self.basic_extractor = BasicPDFExtractor()
#         self.nlp_extractor = NLPEnhancedExtractor()
#         self.ml_extractor = MLEnhancedExtractor()

#     def extract(self, pdf_path: str) -> Dict[str, Any]:
#         """Extract title and outline using the most suitable method"""
#         start_time = time.time()

#         # Analyze PDF characteristics
#         analysis = self.analyzer.analyze_pdf(pdf_path)
#         method = analysis['recommended_method']

#         logger.info(f"📄 Processing: {os.path.basename(pdf_path)}")
#         logger.info(f"📊 Complexity Score: {analysis['complexity_score']:.2f}")
#         logger.info(f"🔧 Selected Method: {method.upper()}")

#         # Select and run extractor
#         try:
#             if method == 'ml' and self.ml_extractor.classifier and SPACY_AVAILABLE:
#                 title, outline = self.ml_extractor.extract_title_and_outline(pdf_path)
#                 method_used = "ML-Enhanced"
#             elif method == 'nlp' and SPACY_AVAILABLE:
#                 title, outline = self.nlp_extractor.extract_title_and_outline(pdf_path)
#                 method_used = "NLP-Enhanced"
#             else:
#                 title, outline = self.basic_extractor.extract_title_and_outline(pdf_path)
#                 method_used = "Basic Rule-Based"

#             extraction_time = time.time() - start_time

#             # Clean outline - remove any temporary fields
#             clean_outline = []
#             for item in outline:
#                 clean_item = {
#                     "level": item.get("level"),
#                     "text": item.get("text"),
#                     "page": item.get("page")
#                 }
#                 clean_outline.append(clean_item)

#             # Prepare result - simple format without analysis
#             result = {
#                 "title": title,
#                 "outline": clean_outline
#             }

#             logger.info(f"✅ Extraction completed in {extraction_time:.2f}s using {method_used}")
#             logger.info(f"📝 Found {len(outline)} outline items")

#             return result

#         except Exception as e:
#             logger.error(f"❌ Extraction failed: {e}")
#             # Fallback to basic method
#             try:
#                 title, outline = self.basic_extractor.extract_title_and_outline(pdf_path)
#                 clean_outline = []
#                 for item in outline:
#                     clean_item = {
#                         "level": item.get("level"),
#                         "text": item.get("text"),
#                         "page": item.get("page")
#                     }
#                     clean_outline.append(clean_item)

#                 return {
#                     "title": title,
#                     "outline": clean_outline
#                 }
#             except Exception as fallback_error:
#                 logger.error(f"❌ Fallback extraction also failed: {fallback_error}")
#                 return {
#                     "title": "Extraction Failed",
#                     "outline": []
#                 }

# def process_single_pdf(input_path: str, output_path: str) -> bool:
#     """Process a single PDF file"""
#     try:
#         extractor = SmartPDFExtractor()
#         result = extractor.extract(input_path)

#         # Save result
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(result, f, indent=2, ensure_ascii=False)

#         return True

#     except Exception as e:
#         logger.error(f"❌ Failed to process {input_path}: {e}")
#         return False

# def process_all_pdfs(input_dir: str, output_dir: str) -> None:
#     """Process all PDFs in the input directory"""
#     input_path = Path(input_dir)
#     output_path = Path(output_dir)

#     # Create output directory if it doesn't exist
#     output_path.mkdir(parents=True, exist_ok=True)

#     # Find all PDF files
#     pdf_files = list(input_path.glob("*.pdf"))

#     if not pdf_files:
#         logger.warning(f"⚠️ No PDF files found in {input_dir}")
#         return

#     logger.info(f"🚀 Starting batch processing of {len(pdf_files)} PDFs")

#     successful = 0
#     failed = 0

#     for pdf_file in pdf_files:
#         output_file = output_path / f"{pdf_file.stem}_smart.json"

#         logger.info(f"\n📁 Processing: {pdf_file.name}")

#         if process_single_pdf(str(pdf_file), str(output_file)):
#             successful += 1
#             logger.info(f"✅ Saved: {output_file.name}")
#         else:
#             failed += 1

#     logger.info(f"\n🎯 Batch processing completed!")
#     logger.info(f"✅ Successful: {successful}")
#     logger.info(f"❌ Failed: {failed}")

# if __name__ == "__main__":
#     import sys

#     # Default paths
#     INPUT_DIR = "./app/inputs"
#     OUTPUT_DIR = "./app/outputs"

#     if len(sys.argv) > 1:
#         # Single file mode
#         pdf_path = sys.argv[1]
#         if not os.path.exists(pdf_path):
#             print(f"❌ File not found: {pdf_path}")
#             sys.exit(1)

#         output_path = f"{os.path.splitext(pdf_path)[0]}_smart.json"
#         if len(sys.argv) > 2:
#             output_path = sys.argv[2]

#         print(f"🔍 Processing single file: {pdf_path}")
#         if process_single_pdf(pdf_path, output_path):
#             print(f"✅ Results saved to: {output_path}")
#         else:
#             print("❌ Processing failed")
#     else:
#         # Batch mode
#         print("🚀 Smart PDF Extractor - Batch Mode")
#         print(f"📂 Input Directory: {INPUT_DIR}")
#         print(f"📁 Output Directory: {OUTPUT_DIR}")
#         process_all_pdfs(INPUT_DIR, OUTPUT_DIR)
















# smart_pdf_extractor.py - Optimized for Constrained Environments (v8 - With Line Merging)
# Meets: <=10s/50pg, <=200MB, CPU-only, No Network

import fitz  # PyMuPDF
import re
import os
import json
from typing import List, Dict, Any, Tuple
import logging
from pathlib import Path
import time
from statistics import mean, median

# Configure logging for production environment
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConstrainedPDFExtractor:
    """
    A highly optimized, rule-based extractor designed for performance and human-like accuracy
    in resource-constrained, offline environments. It uses layout-aware logic, including
    intelligent line merging, to understand document structure without heavy AI models.
    """

    def _get_document_statistics(self, doc: fitz.Document) -> Dict[str, float]:
        """
        Calculates statistics about font sizes across the document. This is a fast
        and crucial step for adapting rules to each specific document's styling.
        """
        sizes = []
        page_count = doc.page_count
        sample_pages = range(min(page_count, 50))

        for i in sample_pages:
            page = doc[i]
            blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT).get("blocks", [])
            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        sizes.append(span['size'])
        
        if not sizes:
            return {'mean': 12.0, 'median': 12.0, 'max': 14.0}
            
        return {
            'mean': mean(sizes),
            'median': median(sorted(sizes)),
            'max': max(sizes)
        }

    def _is_likely_heading(self, text: str, size: float, is_bold: bool, stats: Dict) -> bool:
        """
        Determines if a line of text is a true heading using a strict, multi-layered
        filtering and scoring system to achieve human-level precision.
        """
        # Layer 1: Aggressive Noise & Junk Rejection
        if re.fullmatch(r'[\.\-_*#=]{5,}', text): return False
        if '...' in text and len(text.replace('.', '').strip()) < 10: return False
        if re.search(r'(version|date|remarks|author|page\s+\d+)', text, re.IGNORECASE): return False
        if re.match(r'^\d+(\.\d+)*$', text): return False

        # Layer 2: Content and Structure Validation
        word_count = len(text.split())
        # Allow slightly longer headings now that we merge lines.
        if not (1 <= word_count <= 25): return False
        if text.endswith('.') or text.endswith(','): return False
        if re.match(r'^\s*[●•*–-]\s*', text): return False

        # Layer 3: Scoring Based on Positive Indicators
        score = 0
        if size > stats['median'] * 1.20: score += 3
        elif size > stats['median'] * 1.10: score += 2
        
        if is_bold: score += 2
        
        if text.isupper() and word_count > 1: score += 2

        if text.istitle() and word_count > 1: score += 1

        if re.match(r'^\d+(\.\d+)*\s+[A-Z]', text): score += 3
        
        if text.endswith(':'): score += 1

        return score >= 4

    def _determine_heading_level(self, size: float, stats: Dict) -> str:
        """Assigns H1-H4 levels based on font size relative to document stats."""
        size_ratio = size / stats['median']
        if size_ratio > 1.4: return "H1"
        if size_ratio > 1.25: return "H2"
        if size_ratio > 1.10: return "H3"
        return "H4"

    def _merge_lines(self, lines: List[Dict]) -> List[Dict]:
        """
        NEW: Merges consecutive lines that are likely part of the same multi-line heading.
        This is the key to fixing split headings.
        """
        if not lines:
            return []

        merged_lines = []
        current_line_info = lines[0]

        for i in range(1, len(lines)):
            prev_line_info = lines[i-1]
            next_line_info = lines[i]

            # Heuristics to merge lines:
            vertical_distance = next_line_info['bbox'][1] - prev_line_info['bbox'][3]
            font_size_similar = abs(next_line_info['size'] - current_line_info['size']) < 1
            style_similar = next_line_info['is_bold'] == current_line_info['is_bold']
            same_page = next_line_info['page'] == current_line_info['page']

            # Condition to merge: must be on the same page, vertically close, and have similar styling.
            if same_page and vertical_distance < (current_line_info['size'] * 0.5) and font_size_similar and style_similar:
                 # Merge text
                 current_line_info['text'] += " " + next_line_info['text']
                 # Update bbox to encompass both lines
                 current_line_info['bbox'] = (
                     min(current_line_info['bbox'][0], next_line_info['bbox'][0]),
                     current_line_info['bbox'][1],
                     max(current_line_info['bbox'][2], next_line_info['bbox'][2]),
                     next_line_info['bbox'][3]
                 )
            else:
                # Not a continuation, so finalize the current line and start a new one.
                merged_lines.append(current_line_info)
                current_line_info = next_line_info
            
        merged_lines.append(current_line_info) # Add the last processed line
        return merged_lines

    def extract(self, pdf_path: str) -> Dict[str, Any]:
        """
        Main extraction function. It reads the PDF, analyzes its structure,
        groups text blocks, and classifies them to build a clean outline.
        """
        try:
            doc = fitz.open(pdf_path)
            stats = self._get_document_statistics(doc)
            
            document_title = os.path.basename(pdf_path).replace('.pdf', '') # Default title
            outline = []
            
            # --- Smarter Title Detection ---
            title_found = False
            page = doc[0]
            blocks = sorted(page.get_text("blocks"), key=lambda b: b[1])
            for i, b in enumerate(blocks):
                if i > 5: break
                block_text = " ".join(b[4].strip().split())
                if not block_text: continue
                
                lines = page.get_text("dict", clip=b[:4])['blocks'][0].get('lines', [])
                if lines and lines[0]['spans']:
                    size = lines[0]['spans'][0]['size']
                    if size >= stats['max'] * 0.85 and 1 < len(block_text.split()) < 20:
                        document_title = block_text
                        title_found = True
                        break
            
            # --- REVISED Heading Extraction with Line Merging ---
            # 1. First, extract all individual lines from the document.
            all_lines = []
            for page_num, page in enumerate(doc, 1):
                page_blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT, sort=True).get("blocks", [])
                for block in page_blocks:
                    for line in block.get("lines", []):
                        spans = line.get("spans", [])
                        if not spans: continue
                        
                        line_text = " ".join([s['text'] for s in spans]).strip()
                        if not line_text: continue
                        
                        avg_size = mean([s['size'] for s in spans])
                        is_bold = any("bold" in s['font'].lower() for s in spans)
                        
                        all_lines.append({
                            "text": line_text, "size": avg_size, "is_bold": is_bold,
                            "page": page_num, "bbox": line['bbox']
                        })

            # 2. Now, merge consecutive lines that are likely split headings.
            merged_lines = self._merge_lines(all_lines)

            # 3. Finally, classify the clean, merged lines.
            for line_info in merged_lines:
                page_height = doc[line_info['page']-1].rect.height
                y_pos = line_info['bbox'][1]
                # Filter out headers/footers
                if not (page_height * 0.08 < y_pos < page_height * 0.92):
                    continue

                if self._is_likely_heading(line_info['text'], line_info['size'], line_info['is_bold'], stats):
                    level = self._determine_heading_level(line_info['size'], stats)
                    outline.append({
                        "level": level, "text": line_info['text'], "page": line_info['page']
                    })
            
            # --- Final Cleanup for Human-Like Output ---
            seen = set()
            cleaned_outline = []
            if document_title.lower() in [item['text'].lower() for item in outline]:
                 outline = [item for item in outline if item['text'].lower() != document_title.lower()]

            for item in outline:
                key = (item['text'].lower(), item['page'])
                if key not in seen:
                    cleaned_outline.append(item)
                    seen.add(key)

            return {"title": document_title, "outline": cleaned_outline}

        except Exception as e:
            logger.error(f"Critical failure during extraction for {pdf_path}: {e}", exc_info=True)
            return {"title": "Extraction Failed", "outline": []}

def process_single_pdf(input_path: str, output_path: str) -> bool:
    """Wrapper function to process a single PDF file."""
    try:
        logger.info(f"Processing: {input_path}")
        start_time = time.time()
        
        extractor = ConstrainedPDFExtractor()
        result = extractor.extract(input_path)
        
        extraction_time = time.time() - start_time
        logger.info(f"Extraction completed in {extraction_time:.2f}s")
        logger.info(f"Title: '{result['title']}', Found {len(result['outline'])} outline items.")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to process {input_path}: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    # Adhering to the specified input/output paths for the environment
    INPUT_DIR = Path("./app/inputs")
    OUTPUT_DIR = Path("./app/outputs")
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in the input directory: {INPUT_DIR}")
        (OUTPUT_DIR / "no_files_found.txt").touch()
    else:
        logger.info(f"Found {len(pdf_files)} PDF(s) to process.")
        for pdf_file in pdf_files:
            output_file = OUTPUT_DIR / f"{pdf_file.stem}_outline.json"
            process_single_pdf(str(pdf_file), str(output_file))
    
    logger.info("Batch processing complete.")
