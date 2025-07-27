# Connecting-the-Dots-Through-Docs

A robust PDF document processor that intelligently extracts outlines and titles from a wide variety of documents using advanced AI techniques. This project is containerized with Docker for easy deployment and reproducibility.
The system automatically detects document types (academic papers, resumes, presentations, posters) and applies specialized extraction algorithms for optimal results. Built with PyMuPDF for high-performance PDF parsing and featuring smart boilerplate removal to ensure clean, structured output.
## 📁 Project Structure

```
Connecting-the-Dots-Through-Docs-Round-1a/
│
├── app/
│   ├── pdf_process.py         # Main PDF processing script   
│   ├── inputs/                # Place your input PDF files here
│   └── outputs/               # Extracted outlines and results will be saved here
│
├── dockerfile                 # Dockerfile for building the container
├── requirements.txt           # Python dependencies
└── LICENSE
```

##  Features

- Advanced, position-aware boilerplate detection
- Robust title and outline extraction for academic, resume, and presentation PDFs
- Clean, organized output in JSON format
- Easy to run anywhere with Docker


##  Function Documentation

### Core Extraction Class: `FinalExtractor`

**`_get_document_statistics(doc)`**
Analyzes font sizes, text density, and page count to classify documents as academic, resume, poster, or presentation types.
This classification helps optimize the extraction strategy and parameters for different document layouts.

**`_clean_text(text)`**
Removes repetitive patterns, normalizes whitespace, and fixes common OCR issues like fragmented words.
Also handles Unicode ligature replacement (ﬁ→fi, ﬂ→fl) and eliminates duplicate tokens from text blocks.

**`_is_meaningful_text(text, is_title=False)`**
Applies intelligent filters to distinguish between actual headings and noise like bullet points or page numbers.
Uses length constraints, alphabetic character requirements, and pattern matching to validate text quality.

**`_detect_boilerplate(doc)`**
Identifies repeating headers and footers by tracking text positions across multiple pages using y-coordinate bucketing.
Creates normalized signatures of text blocks and flags content that appears in similar positions on 2+ pages.

**`_find_document_title(doc)`**
Locates the document title by finding the largest font text in the upper half of the first page.
Combines multiple title candidates that meet size criteria and sorts them by position for accurate title reconstruction.

**`_determine_heading_level(score)`**
Maps numerical heading confidence scores to hierarchical levels (H1, H2, H3) based on formatting strength.
Uses threshold-based classification where higher scores indicate more prominent headings.

**`_extract_academic_outline(doc, stats, boilerplate)`**
Main extraction engine that scores text blocks based on font size, bold formatting, spacing, and numbering patterns.
Processes each page sequentially, calculating confidence scores and filtering out boilerplate to build the document outline.

**`extract(pdf_path)`**
Orchestrates the complete extraction pipeline from document analysis to final outline generation.
Coordinates all helper methods and handles error recovery to ensure robust processing of various PDF formats.

### Utility Functions

**`process_single_pdf(input_path, output_path)`**
Wrapper function that handles individual PDF processing with timing and logging functionality.
Creates FinalExtractor instance, processes the file, and saves results as formatted JSON output.

**`__main__` execution block**
Batch processing controller that discovers PDF files in the inputs directory and processes them sequentially.
Sets up directory structure, handles file discovery, and coordinates the processing of multiple documents.

###""


## 🐳 Running with Docker

### 1. Build the Docker Image

Open a terminal in the root of your project and run:

```sh
docker build --platform linux/amd64 -t round_1a:latest .
```

### 2. Run the Docker Container

Make sure your PDFs are in `app/inputs/`. To process them and get results in `app/outputs/`, run:

```sh
docker run --rm -v ${PWD}/app/inputs:/app/inputs -v ${PWD}/app/outputs:/app/outputs --network none round_1a:latest
```

- `${PWD}` ensures your current directory is used for mounting.
- The `--network none` flag disables networking for extra security.

## 📝 How It Works

- Place your PDF files in the `app/inputs/` directory.
- When you run the Docker container, it processes all PDFs in `inputs/` and writes the extracted outlines as JSON files to `outputs/`.
- The main logic is in `app/pdf_process.py`.


## 📦 Requirements

- [Docker](https://www.docker.com/) installed on your system.

## 🤝 License

This project is licensed under the terms of the LICENSE file.

