import os
import json
from extractor import extract_title_and_outline

PDF_DIR = "./app/inputs"
OUTPUT_DIR = "./app/outputs"

def process_all_pdfs():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for filename in os.listdir(PDF_DIR):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(PDF_DIR, filename)
            print(f"📄 Processing: {filename}")
            try:
                title, outline = extract_title_and_outline(pdf_path)

                output_data = {
                    "title": title,
                    "outline": outline
                }

                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(OUTPUT_DIR, base_name + ".json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)

                print(f"✅ Saved: {output_path}")

            except Exception as e:
                print(f"❌ Failed to process {filename}: {str(e)}")

if __name__ == "__main__":
    process_all_pdfs()
