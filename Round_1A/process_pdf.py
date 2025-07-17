"""
Professional PDF Processing Module

This module provides a robust PDF processing pipeline for extracting titles 
and outlines from PDF documents with comprehensive error handling and logging.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from extractor import PDFTextExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for PDF processing"""
    pdf_dir: str = "app/inputs"
    output_dir: str = "app/outputs"
    supported_extensions: List[str] = None
    
    def __post_init__(self):
        if self.supported_extensions is None:
            self.supported_extensions = ['.pdf']


class PDFProcessor:
    """Professional PDF processing pipeline"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.extractor = PDFTextExtractor()
        self.results = []
    
    def process_all_pdfs(self) -> Dict[str, any]:
        """
        Process all PDF files in the input directory.
        
        Returns:
            Dictionary containing processing results and statistics
        """
        logger.info("Starting PDF processing pipeline")
        
        # Validate directories
        if not self._validate_directories():
            return self._create_error_result("Directory validation failed")
        
        # Find PDF files
        pdf_files = self._find_pdf_files()
        if not pdf_files:
            logger.warning(f"No PDF files found in '{self.config.pdf_dir}'")
            return self._create_result(success=True, message="No PDF files to process")
        
        logger.info(f"Found {len(pdf_files)} PDF file(s) to process")
        
        # Process each PDF
        results = []
        successful = 0
        failed = 0
        
        for pdf_file in pdf_files:
            try:
                result = self._process_single_pdf(pdf_file)
                results.append(result)
                
                if result['success']:
                    successful += 1
                    logger.info(f"✅ Successfully processed: {pdf_file.name}")
                else:
                    failed += 1
                    logger.error(f"❌ Failed to process: {pdf_file.name}")
                    
            except Exception as e:
                failed += 1
                error_msg = f"Unexpected error processing {pdf_file.name}: {str(e)}"
                logger.error(error_msg)
                results.append({
                    'filename': pdf_file.name,
                    'success': False,
                    'error': error_msg
                })
        
        # Create final result
        return {
            'success': True,
            'total_files': len(pdf_files),
            'successful': successful,
            'failed': failed,
            'results': results,
            'message': f"Processed {successful}/{len(pdf_files)} files successfully"
        }
    
    def _validate_directories(self) -> bool:
        """Validate input and output directories"""
        input_path = Path(self.config.pdf_dir)
        output_path = Path(self.config.output_dir)
        
        if not input_path.exists():
            logger.error(f"Input directory '{self.config.pdf_dir}' does not exist")
            return False
        
        # Create output directory if it doesn't exist
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory ready: {self.config.output_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to create output directory: {str(e)}")
            return False
    
    def _find_pdf_files(self) -> List[Path]:
        """Find all PDF files in the input directory"""
        input_path = Path(self.config.pdf_dir)
        pdf_files = []
        
        for ext in self.config.supported_extensions:
            pdf_files.extend(input_path.glob(f"*{ext}"))
            pdf_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        return sorted(pdf_files)
    
    def _process_single_pdf(self, pdf_file: Path) -> Dict[str, any]:
        """Process a single PDF file"""
        try:
            logger.info(f"Processing: {pdf_file.name}")
            
            # Extract title and outline
            title, outline = self.extractor.extract_title_and_outline(str(pdf_file))
            
            # Create output data structure
            output_data = {
                "title": title,
                "outline": outline
            }
            
            # Save to JSON file
            output_path = Path(self.config.output_dir) / f"{pdf_file.stem}.json"
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)
            
            # Log processing details
            logger.info(f"   📝 Title: {title}")
            logger.info(f"   📋 Extracted {len(outline)} headings")
            logger.info(f"   💾 Saved to: {output_path}")
            
            return {
                'filename': pdf_file.name,
                'success': True,
                'title': title,
                'outline_count': len(outline),
                'output_path': str(output_path)
            }
            
        except Exception as e:
            error_msg = f"Error processing {pdf_file.name}: {str(e)}"
            logger.error(error_msg)
            return {
                'filename': pdf_file.name,
                'success': False,
                'error': error_msg
            }
    
    def _create_result(self, success: bool, message: str, **kwargs) -> Dict[str, any]:
        """Create a standardized result dictionary"""
        result = {
            'success': success,
            'message': message,
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'results': []
        }
        result.update(kwargs)
        return result
    
    def _create_error_result(self, message: str) -> Dict[str, any]:
        """Create an error result dictionary"""
        return self._create_result(success=False, message=message)


def process_all_pdfs() -> Dict[str, any]:
    """
    Legacy function for backward compatibility.
    Process all PDF files in the input directory.
    """
    processor = PDFProcessor()
    return processor.process_all_pdfs()


def main():
    """Main entry point for the PDF processing pipeline"""
    try:
        processor = PDFProcessor()
        result = processor.process_all_pdfs()
        
        if result['success']:
            print(f"\n🎉 Processing completed successfully!")
            print(f"📊 Results: {result['successful']}/{result['total_files']} files processed")
        else:
            print(f"\n❌ Processing failed: {result['message']}")
            
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error in main: {str(e)}")
        print(f"\n💥 Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()
