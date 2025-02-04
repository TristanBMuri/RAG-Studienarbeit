import os
import pandas as pd
import pdfplumber
import camelot
import json
import logging
import random
from multiprocessing import Pool


class PDFTableExtractor:
    def __init__(self, pdf_folder, log_file="table_extraction.log"):
        self.pdf_folder = pdf_folder
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def ensure_output_directory(self, directory):
        """Ensure the output directory exists."""
        if not os.path.exists(directory):
            os.makedirs(directory)

    def detect_table(self, pdf_path):
        try:
            tables = camelot.read_pdf(pdf_path, pages="1")
            return len(tables) > 0
        except Exception as e:
            logging.error(f"Error detecting tables in {pdf_path} with Camelot: {e}")
            return False

    def detect_table_plumber(self, pdf_path):
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables()
                    if tables:
                        return True
        except Exception as e:
            logging.error(f"Error detecting tables in {pdf_path} with pdfplumber: {e}")
        return False

    def extract_tables_from_pdf(self, pdf_path):
        tables_data = []
        output_csv = pdf_path.replace(".pdf", "_tables.csv")

        try:
            if self.detect_table(pdf_path):  # Try Camelot first
                tables = camelot.read_pdf(pdf_path, pages="all")
                if tables.n > 0:
                    df = pd.concat([table.df for table in tables])
                    df.to_csv(output_csv, index=False)
            else:  # Use PDFPlumber as a fallback
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        tables = page.extract_tables()
                        for table in tables:
                            tables_data.extend(table)
                if tables_data:
                    df = pd.DataFrame(tables_data)
                    df.to_csv(output_csv, index=False)

            if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
                logging.info(f"Extracted tables successfully: {pdf_path}")
                return output_csv
            else:
                logging.warning(f"No valid tables extracted: {pdf_path}")
                return None
        except Exception as e:
            logging.error(f"Error extracting tables from {pdf_path}: {e}")
            return None

    def csv_to_json(self, csv_file):
        """Convert a CSV file to JSON and save it in src/json."""
        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                logging.warning(f"Skipping empty CSV file: {csv_file}")
                return None

            # Ensure the output directory exists
            output_dir = os.path.join("src", "json")
            self.ensure_output_directory(output_dir)

            # Create JSON filename using secure path joining
            base_name = os.path.splitext(os.path.basename(csv_file))[0]
            json_file = os.path.join(output_dir, f"{base_name}.json")

            # Convert to JSON and save
            json_output = df.to_json(orient="records", indent=2)  # Added indent for readability
            with open(json_file, "w", encoding="utf-8") as f:
                f.write(json_output)

            return json_file
        except Exception as e:
            logging.error(f"Error converting CSV to JSON: {csv_file}: {e}")
            return None

    def process_pdfs_in_parallel(self):
        pdf_files = [os.path.join(self.pdf_folder, f) for f in os.listdir(self.pdf_folder) if f.endswith(".pdf")]

        if not pdf_files:
            logging.warning("No PDF files found for table extraction.")
            return []

        with Pool(processes=4) as pool:
            table_files = pool.map(self.extract_tables_from_pdf, pdf_files)

        return [file for file in table_files if file]

    def run(self):
        extracted_tables = self.process_pdfs_in_parallel()
        extracted_tables = [file for file in extracted_tables if file]  # Filter out None values
        json_files = [self.csv_to_json(file) for file in extracted_tables if file]
        logging.info(f"Successfully processed {len(json_files)} tables into JSON.")

if __name__ == "__main__":
    pdf_folder = input("Enter the PDF folder path: ")  # Allow user input
    extractor = PDFTableExtractor(pdf_folder)
    extractor.run()
