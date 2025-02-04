# Contriever processing of PDFs
import glob
import re
import fitz  # PyMuPDF
import json
from tqdm import tqdm
from collections import defaultdict

src = "./Docs/*.pdf"  # Explicit PDF pattern


# ------------------
# Text Extraction
# ------------------
def extract_text(pdf_path):
    """Extract text with page numbers using PyMuPDF"""
    doc = fitz.open(pdf_path)
    text_pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("blocks")  # Get text blocks with coordinates
        table_blocks = [b for b in text if "table" in b[4].lower()]
        for block in table_blocks:
            # Add table marker
            text = text.replace(block[4], f"\n[TABLE_START]{block[4]}[TABLE_END]\n")
        text_pages.append({"text": text, "page": page_num, "source": pdf_path.split("/")[-1]})
    return text_pages


# ------------------
# Header/Footer Detection
# ------------------
def detect_repeating_patterns(text_pages, num_lines=2):
    """Identify potential headers/footers by analyzing repeating top/bottom lines"""
    header_counts = defaultdict(int)
    footer_counts = defaultdict(int)

    for page in text_pages:
        lines = page["text"].split('\n')
        # Check top N lines
        header = '\n'.join(lines[:num_lines])
        header_counts[header] += 1
        # Check bottom N lines
        footer = '\n'.join(lines[-num_lines:]) if len(lines) > num_lines else ''
        footer_counts[footer] += 1

    # Find patterns that appear on most pages
    common_headers = [patt for patt, count in header_counts.items()
                      if count > len(text_pages) * 0.75]  # Appears on >75% pages
    common_footers = [patt for patt, count in footer_counts.items()
                      if count > len(text_pages) * 0.75]

    return common_headers, common_footers


# ------------------
# Cleaning & Normalization
# ------------------
def clean_text(text, page_num, header_patterns=None, footer_patterns=None):
    # Basic cleaning
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'-\n', '', text)
    text = text.strip()

    # Header/footer removal if patterns are provided
    if header_patterns or footer_patterns:
        lines = text.split('\n')

        # Remove header patterns from first 2 lines
        if header_patterns and len(lines) > 1:
            for pattern in header_patterns:
                if re.search(pattern, lines[0], re.IGNORECASE):
                    lines = lines[1:]
                if re.search(pattern, lines[0], re.IGNORECASE):
                    lines = lines[1:]

        # Remove footer patterns from last 2 lines
        if footer_patterns and len(lines) > 1:
            for pattern in footer_patterns:
                if re.search(pattern, lines[-1], re.IGNORECASE):
                    lines = lines[:-1]
                if re.search(pattern, lines[-1], re.IGNORECASE):
                    lines = lines[:-1]

        text = '\n'.join(lines)

    # Additional table preservation cleanups
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)  # Fix space between table numbers
    return text.strip()

# ------------------
# Chunking/Passage Splitting
# ------------------
def create_chunks(text, max_words=300, min_chunk_size=100):
    """Split text into word-count based chunks with context preservation"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_word_count = 0

    for word in words:
        current_chunk.append(word)
        current_word_count += 1
        if current_word_count >= max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_word_count = 0

    # Handle remaining text
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Merge small last chunk if needed
    if len(chunks) > 1 and len(chunks[-1].split()) < min_chunk_size:
        chunks[-2] += " " + chunks[-1]
        chunks.pop()

    return chunks


# ------------------
# Metadata & Storage
# ------------------
def process_pdf(pdf_path, remove_headers=True):
    # Full processing pipeline
    extracted = extract_text(pdf_path)
    all_chunks = []

    # Auto-detect headers/footers if enabled
    header_patts, footer_patts = [], []
    if remove_headers and len(extracted) > 2:
        header_patts, footer_patts = detect_repeating_patterns(extracted)
        print(f"Detected {len(header_patts)} header patterns, {len(footer_patts)} footer patterns")

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
                "text": chunk_text,
                "has_table": "[TABLE]" in chunk_text  # Simple table detection
            }
            all_chunks.append(chunk_metadata)

    return all_chunks


# ------------------
# Main Processing
# ------------------
if __name__ == "__main__":
    pdf_files = glob.glob(src)
    all_documents = []

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        all_documents.extend(process_pdf(pdf_file))

    # Save results for indexing
    with open("processed_chunks.json", "w") as f:
        json.dump(all_documents, f, indent=2)

    print(f"Processed {len(all_documents)} chunks from {len(pdf_files)} PDFs")