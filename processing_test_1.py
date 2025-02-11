import os
import pathlib
from typing import List, Dict
import pandas as pd
import PyPDF2
from datetime import datetime


def get_pdf_content(file_path: str) -> Dict:
    """Extract content and metadata from PDF file"""
    try:
        with open(file_path, 'rb') as file:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)

            # Extract text from all pages
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"

            # Get metadata
            metadata = {
                'num_pages': len(pdf_reader.pages),
                'metadata': pdf_reader.metadata or {}
            }

            return {
                'content': text.strip(),
                'metadata': metadata
            }
    except Exception as e:
        print(f"Error processing PDF {file_path}: {str(e)}")
        return {'content': "", 'metadata': {}}


def process_docs_folder(docs_path: str = "docs") -> List[Dict]:
    """Process all PDF files in the docs folder"""
    processed_docs = []

    # Create docs folder if it doesn't exist
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)
        print(f"Created {docs_path} directory")
        return processed_docs

    # Walk through all files in docs folder
    for root, _, files in os.walk(docs_path):
        for file in files:
            if not file.lower().endswith('.pdf'):
                continue

            file_path = os.path.join(root, file)

            # Get file stats
            file_stats = os.stat(file_path)

            # Process PDF
            pdf_data = get_pdf_content(file_path)

            if pdf_data['content']:
                doc_info = {
                    'file_name': file,
                    'file_path': file_path,
                    'content': pdf_data['content'],
                    'num_pages': pdf_data['metadata']['num_pages'],
                    'file_size': file_stats.st_size,
                    'created_date': datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                    'modified_date': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                    'pdf_metadata': str(pdf_data['metadata']['metadata'])
                }
                processed_docs.append(doc_info)
                print(f"Processed: {file} ({pdf_data['metadata']['num_pages']} pages)")

    print(f"\nProcessed {len(processed_docs)} PDF documents")
    return processed_docs


def save_processed_docs(processed_docs: List[Dict], output_file: str = "src/processed_pdfs.csv"):
    """Save processed documents to a CSV file"""
    if processed_docs:
        df = pd.DataFrame(processed_docs)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Saved processed documents to {output_file}")

        # Save a summary
        summary = pd.DataFrame({
            'file_name': df['file_name'],
            'num_pages': df['num_pages'],
            'file_size': df['file_size'],
            'content_length': df['content'].str.len()
        })
        summary.to_csv('src/pdf_processing_summary.csv', index=False)
        print("Saved processing summary to pdf_processing_summary.csv")


if __name__ == "__main__":
    # Process all documents
    docs = process_docs_folder()

    # Save processed documents
    save_processed_docs(docs)