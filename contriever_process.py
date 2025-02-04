"""
PDF Processor for Contriever Preparation
- Handles text extraction, cleaning, chunking, and metadata
- Configurable header/footer removal
- Table-aware processing
"""

import glob
import re
import fitz  # PyMuPDF
import json
from tqdm import tqdm
from collections import defaultdict

# Configuration
DEFAULT_MAX_WORDS = 300
MIN_CHUNK_SIZE = 100
HEADER_FOOTER_THRESHOLD = 0.75 #% of pages needing pattern to qualify as header / footer

def extract_text(pdf_path):
    """Extract text with page numbers using PyMuPDF"""
    doc = fitz.open(pdf_path)
    text_pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        text_pages.append({
            "text": text,
            "page": page_num,
            "source": pdf_path.split("/")[-1]
        })
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
    # Basic normalization
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'-\n', '', text)
    text = text.strip()

    if header_patterns or footer_patterns:
        lines = text.split('\n')

        # Header removal
        if header_patterns and len(lines) > 1:
            while any(re.search(patt, lines[0], re.IGNORECASE) for patt in header_patterns):
                lines.pop(0)
                if not lines: break

        # Footer removal
        if footer_patterns and len(lines) > 1:
            while any(re.search(patt, lines[-1], re.IGNORECASE) for patt in footer_patterns):
                lines.pop()
                if not lines: break

        text = '\n'.join(lines)

    # Table preservation
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    return text.strip()


def create_chunks(text, max_words=DEFAULT_MAX_WORDS, min_chunk_size=MIN_CHUNK_SIZE):
    """Split text into word-count based chunks"""
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    if len(chunks) > 1 and len(chunks[-1].split()) < min_chunk_size:
        chunks[-2] += " " + chunks[-1]
        chunks.pop()

    return chunks


def process_pdf(pdf_path, remove_headers=True):
    """Full processing pipeline for a single PDF"""
    extracted = extract_text(pdf_path)
    all_chunks = []

    header_patts, footer_patts = [], []
    if remove_headers and len(extracted) > 2:
        header_patts, footer_patts = detect_repeating_patterns(extracted)

    for page in extracted:
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

    return all_chunks


if __name__ == "__main__":
    pdf_files = glob.glob("./Docs/*.pdf")
    all_documents = []

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        all_documents.extend(process_pdf(pdf_file, remove_headers=True))

    with open("processed_chunks.json", "w") as f:
        json.dump(all_documents, f, indent=2)

    print(f"Processed {len(all_documents)} chunks from {len(pdf_files)} PDFs")