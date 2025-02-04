import os
import json
import logging
import fitz  # PyMuPDF for text extraction
from pdf_table_extraction.pdf_table_extractor import PDFTableExtractor

# Set up logging
logging.basicConfig(filename="contriever_process.log", level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def ensure_output_directory(directory):
    """Ensure the output directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def extract_text_from_pdf(pdf_path):
    """Extract plain text from a PDF file using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        text_data = []

        for page in doc:
            text = page.get_text("text")
            if text.strip():  # Avoid empty text
                text_data.append({
                    "page": page.number + 1,
                    "text": text.strip()
                })

        return text_data
    except Exception as e:
        logging.error(f"Failed to extract text from {pdf_path}: {e}")
        return None


def process_pdf(pdf_path, output_dir):
    """Processes a single PDF file, extracting both text and tables."""
    if not os.path.exists(pdf_path):
        logging.error(f"PDF file not found: {pdf_path}")
        return None

    try:
        pdf_name = os.path.basename(pdf_path)
        extractor = PDFTableExtractor(os.path.dirname(pdf_path))  # Initialize extractor

        # Extract structured tables
        extracted_tables = extractor.extract_tables_from_pdf(pdf_path)

        # Extract unstructured text
        extracted_text = extract_text_from_pdf(pdf_path)

        if extracted_text is None:
            logging.warning(f"No text extracted from {pdf_name}")

        # Convert extracted tables to JSON
        table_json_path = None
        if extracted_tables:
            table_json_path = extractor.csv_to_json(extracted_tables)

        # Ensure the output directory exists
        ensure_output_directory(output_dir)

        # Store combined output
        combined_output = {
            "pdf_name": pdf_name,
            "text_data": extracted_text if extracted_text else "No text found",
            "table_data": table_json_path if table_json_path else "No tables found"
        }

        output_json = os.path.join(output_dir, f"{os.path.splitext(pdf_name)[0]}_combined.json")
        with open(output_json, "w") as f:
            json.dump(combined_output, f, indent=4)

        logging.info(f"Processed PDF successfully: {pdf_name}")
        return output_json
    except Exception as e:
        logging.error(f"Failed to process {pdf_path}: {e}")
        return None


def process_pdfs(pdf_folder, output_dir):
    """Processes all PDFs in the given folder and saves output to the specified directory."""
    if not os.path.exists(pdf_folder):
        logging.error(f"PDF folder not found: {pdf_folder}")
        return

    pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

    if not pdf_files:
        logging.warning(f"No PDFs found in folder: {pdf_folder}")
        return

    for pdf in pdf_files:
        process_pdf(pdf, output_dir)


if __name__ == "__main__":
    pdf_folder = "./Docs/"  # Input folder for PDFs
    output_dir = "src/json"  # Directory to save JSON outputs
    process_pdfs(pdf_folder, output_dir)
    logging.info("PDF processing completed.")
