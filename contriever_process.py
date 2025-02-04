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
        text = page.get_text()
        text_pages.append({
            "text": text,
            "page": page_num,
            "source": pdf_path.split("/")[-1]
        })
    return text_pages


# ------------------
# Cleaning & Normalization
# ------------------
def clean_text(text):
    # Basic cleaning pipeline
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace
    text = re.sub(r'-\n', '', text)  # Remove hyphenated line breaks
    text = text.strip()
    return text


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
def process_pdf(pdf_path):
    # Full processing pipeline
    extracted = extract_text(pdf_path)
    all_chunks = []

    for page in extracted:
        cleaned = clean_text(page["text"])
        if not cleaned:
            continue

        chunks = create_chunks(cleaned)
        for chunk_num, chunk_text in enumerate(chunks, 1):
            chunk_metadata = {
                "source": page["source"],
                "pages": (page["page"], page["page"]),  # Could expand for multi-page chunks
                "chunk_id": f"{page['source']}_p{page['page']}_c{chunk_num}",
                "text": chunk_text
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