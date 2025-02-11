import os
from pathlib import Path
from unstructured.partition.pdf import partition_pdf

# Configuration
INPUT_DIR = Path("/Docs/")  # Replace with your PDF folder path
OUTPUT_DIR = Path("/src/Unstructured")  # Replace with your output folder path

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def process_pdfs():
    # Process each PDF in the input directory
    for pdf_file in INPUT_DIR.glob("*.pdf"):
        try:
            print(f"Processing {pdf_file.name}...")

            # Extract elements from PDF
            elements = partition_pdf(
                filename=str(pdf_file),
                strategy="auto",  # Choose processing strategy (auto, fast, hi_res, ocr_only)
                infer_table_structure=True,  # Maintain table structures
            )

            # Combine element texts
            text = "\n\n".join([str(el) for el in elements])

            # Create output file path
            output_path = OUTPUT_DIR / f"{pdf_file.stem}.txt"

            # Save to text file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"Successfully processed {pdf_file.name}")

        except Exception as e:
            print(f"Error processing {pdf_file.name}: {str(e)}")


if __name__ == "__main__":
    process_pdfs()
    print("PDF processing complete!")