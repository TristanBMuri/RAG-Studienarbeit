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
        logging.basicConfig(filename=log_file, level=logging.INFO)

    def detect_table(self, pdf_path):
        try:
            tables = camelot.read_pdf(pdf_path, pages="1")
            return len(tables) > 0
        except:
            return False

    def detect_table_plumber(self, pdf_path):
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                if tables:
                    return True
        return False

    def extract_tables_from_pdf(self, pdf_path):
        tables_data = []
        output_csv = pdf_path.replace(".pdf", "_tables.csv")

        if self.detect_table(pdf_path):  # Try Camelot first
            tables = camelot.read_pdf(pdf_path, pages="all")
            for i, table in enumerate(tables):
                table.df.to_csv(output_csv, mode='a', header=False, index=False)
        else:  # Use PDFPlumber as a fallback
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables()
                    for table in tables:
                        tables_data.extend(table)
            df = pd.DataFrame(tables_data)
            df.to_csv(output_csv, index=False)

        if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
            logging.info(f"Extracted tables successfully: {pdf_path}")
            return output_csv
        else:
            logging.warning(f"No tables detected or extracted: {pdf_path}")
            return None

    def process_pdfs_in_parallel(self):
        pdf_files = [os.path.join(self.pdf_folder, f) for f in os.listdir(self.pdf_folder) if f.endswith(".pdf")]
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
            print(pd.read_csv(file).head())  # Print first few rows for quick inspection

    def csv_to_json(self, csv_file):
        df = pd.read_csv(csv_file)
        json_output = df.to_json(orient="records")
        json_file = csv_file.replace(".csv", ".json")
        with open(json_file, "w") as f:
            f.write(json_output)
        return json_file

    def run(self):
        extracted_tables = self.process_pdfs_in_parallel()
        extracted_tables = [file for file in extracted_tables if self.validate_table_output(file)]
        self.sample_and_inspect(extracted_tables)
        json_files = [self.csv_to_json(file) for file in extracted_tables]
        logging.info(f"Successfully processed {len(json_files)} tables into JSON.")

if __name__ == "__main__":
    pdf_folder = "Docs/"  # Define the folder where PDFs are stored
    extractor = PDFTableExtractor(pdf_folder)
    extractor.run()
