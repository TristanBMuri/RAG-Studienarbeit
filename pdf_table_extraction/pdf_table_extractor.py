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

    def process_pdfs_in_parallel(self):
        pdf_files = [os.path.join(self.pdf_folder, f) for f in os.listdir(self.pdf_folder) if f.endswith(".pdf")]

        if not pdf_files:
            logging.warning("No PDF files found for table extraction.")
            return []

        with Pool(processes=4) as pool:
            table_files = pool.map(self.extract_tables_from_pdf, pdf_files)

        return [file for file in table_files if file]

    def validate_table_output(self, csv_file):
        try:
            df = pd.read_csv(csv_file)
            if df.empty or df.shape[1] < 2:
                logging.warning(f"Potential issue with extracted table: {csv_file}")
                return False
            return True
        except Exception as e:
            logging.error(f"Error reading {csv_file}: {e}")
            return False

    def sample_and_inspect(self, extracted_tables, sample_fraction=0.05):
        sample_size = max(1, int(len(extracted_tables) * sample_fraction))
        sample_files = random.sample(extracted_tables, sample_size)
        for file in sample_files:
            try:
                print(pd.read_csv(file, nrows=5))  # Load only a few rows for efficiency
            except Exception as e:
                logging.error(f"Error sampling {file}: {e}")

    def csv_to_json(self, csv_file):
        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                logging.warning(f"Skipping empty CSV file: {csv_file}")
                return None

            json_output = df.to_json(orient="records")
            json_file = csv_file.replace(".csv", ".json")
            with open(json_file, "w") as f:
                f.write(json_output)

            return json_file
        except Exception as e:
            logging.error(f"Error converting CSV to JSON: {csv_file}: {e}")
            return None

    def run(self):
        extracted_tables = self.process_pdfs_in_parallel()
        extracted_tables = [file for file in extracted_tables if self.validate_table_output(file)]
        self.sample_and_inspect(extracted_tables)
        json_files = [self.csv_to_json(file) for file in extracted_tables if file]
        logging.info(f"Successfully processed {len(json_files)} tables into JSON.")

if __name__ == "__main__":
    # pdf_folder = input("Enter the PDF folder path: ")  # Allow user input
    pdf_folder = "./Docs/"
    extractor = PDFTableExtractor(pdf_folder)
    extractor.run()
