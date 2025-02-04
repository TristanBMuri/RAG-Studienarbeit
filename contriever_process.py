"""
PDF Processor for Contriever Preparation
- Now with sentence-aware chunking and error handling
"""

import glob
import logging
import re
import fitz  # PyMuPDF
import json
import spacy
from tqdm import tqdm
from collections import defaultdict
from pdf_table_extraction import PDFTableExtractor

# Set up logging
logging.basicConfig(filename="contriever_process.log", level=logging.INFO)

# Configuration
DEFAULT_MAX_WORDS = 300
MIN_CHUNK_SIZE = 100
HEADER_FOOTER_THRESHOLD = 0.75  # % of pages needing pattern to qualify as header/footer

# Load spaCy model for sentence splitting
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError("Please install the spaCy English model first:\n"
                       "python -m spacy download en_core_web_sm")

def extract_text(pdf_path):
    """Extract text with page numbers using PyMuPDF"""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening {pdf_path}: {str(e)}")
        return []

    text_pages = []
    for page_num, page in enumerate(doc, start=1):
        try:
            text = page.get_text()
            text_pages.append({
                "text": text,
                "page": page_num,
                "source": pdf_path.split("/")[-1]
            })
        except Exception as e:
            print(f"Error processing page {page_num} in {pdf_path}: {str(e)}")

    return text_pages

def detect_repeating_patterns(text_pages, num_lines=2):
    """Identify potential headers/footers by analyzing repeating top/bottom lines"""
    header_counts = defaultdict(int)
    footer_counts = defaultdict(int)

    for page in text_pages:
        lines = page["text"].split('\n')
        header = '\n'.join(lines[:num_lines])
        header_counts[header] += 1
        footer = '\n'.join(lines[-num_lines:]) if len(lines) > num_lines else ''
        footer_counts[footer] += 1

    common_headers = [patt for patt, count in header_counts.items()
                     if count > len(text_pages) * HEADER_FOOTER_THRESHOLD]
    common_footers = [patt for patt, count in footer_counts.items()
                     if count > len(text_pages) * HEADER_FOOTER_THRESHOLD]

    return common_headers, common_footers

def clean_text(text, page_num, header_patterns=None, footer_patterns=None):
    """Clean text with optional header/footer removal"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'-\n', '', text)
    text = text.strip()

    if header_patterns or footer_patterns:
        lines = text.split('\n')

        if header_patterns and len(lines) > 1:
            while any(re.search(patt, lines[0], re.IGNORECASE) for patt in header_patterns):
                lines.pop(0)
                if not lines: break

        if footer_patterns and len(lines) > 1:
            while any(re.search(patt, lines[-1], re.IGNORECASE) for patt in footer_patterns):
                lines.pop()
                if not lines: break

        text = '\n'.join(lines)

    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    return text.strip()

def create_chunks(text, max_words=DEFAULT_MAX_WORDS, min_chunk_size=MIN_CHUNK_SIZE):
    """Split text into sentence-aware chunks"""
    doc = nlp(text)
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sent in doc.sents:
        sent_word_count = len(sent.text.split())

        if current_word_count + sent_word_count > max_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0

        current_chunk.append(sent.text)
        current_word_count += sent_word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Post-processing to handle small chunks
    if len(chunks) > 1 and len(chunks[-1].split()) < min_chunk_size:
        chunks[-2] += " " + chunks[-1]
        chunks.pop()

    return chunks

def process_pdf(pdf_path, remove_headers=True):
    """Full processing pipeline for a single PDF"""
    extracted = extract_text(pdf_path)
    if not extracted:
        return []

    all_chunks = []
    header_patts, footer_patts = [], []

    try:
        if remove_headers and len(extracted) > 2:
            header_patts, footer_patts = detect_repeating_patterns(extracted)
    except Exception as e:
        print(f"Error detecting patterns in {pdf_path}: {str(e)}")

    for page in extracted:
        try:
            cleaned = clean_text(
                text=page["text"],
                page_num=page["page"],
                header_patterns=header_patts if remove_headers else None,
                footer_patterns=footer_patts if remove_headers else None
            )

            if not cleaned:
                continue

            chunks = create_chunks(cleaned)
            for chunk_num, chunk_text in enumerate(chunks, 1):
                chunk_metadata = {
                    "source": page["source"],
                    "pages": page["page"],
                    "chunk_id": f"{page['source']}_p{page['page']}_c{chunk_num}",
                    "text": chunk_text
                }
                all_chunks.append(chunk_metadata)
        except Exception as e:
            print(f"Error processing page {page['page']} in {pdf_path}: {str(e)}")

    return all_chunks

if __name__ == "__main__":
    pdf_files = glob.glob("./Docs/*.pdf")
    all_documents = []

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            all_documents.extend(process_pdf(pdf_file, remove_headers=True))
        except Exception as e:
            print(f"Critical error processing {pdf_file}: {str(e)}")
            continue

    with open("src/processed_chunks.json", "w") as f:
        json.dump(all_documents, f, indent=2)

    print(f"Successfully processed {len(all_documents)} chunks from {len(pdf_files)} PDFs")
