import os
import sys
import json
import fitz  # PyMuPDF
from PIL import Image
import google.generativeai as genai
import io


def pdf_to_images(pdf_path):
    """Convert each page of the PDF to an image."""
    images = []
    pdf_document = fitz.open(pdf_path)

    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        pix = page.get_pixmap()
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        images.append(img)

    pdf_document.close()
    return images


def extract_table_from_image(image):
    """
    Use the multimodal Gemini model to extract table data.
    The prompt asks the model to return a JSON object with one key "table"
    mapping to a list of rows (each a list of cell strings).
    If no table is found, the model should return {}.
    """
    prompt = (
        "Extract any table present in the image as a JSON object. "
        "The JSON object must have a single key 'table' that maps to a list of rows, "
        "where each row is a list of strings representing the cell values. "
        "Do not include any additional text. If no table is present, output {}."
    )
    # Create a chat session that includes both the prompt and the image.
    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
    chat = model.start_chat(history=[
        {"role": "user", "parts": [prompt, image]}
    ])
    # Send an empty message to complete the turn.
    response = chat.send_message("")
    return response.text


def parse_table_json(text):
    """Attempt to parse the response text as JSON and extract the table."""
    try:
        data = json.loads(text)
        table = data.get("table", [])
        if isinstance(table, list) and all(isinstance(row, list) for row in table):
            return table
    except Exception as e:
        print("Error parsing JSON:", e)
    return []


def merge_tables(tables):
    """
    Merge tables from multiple images.
    Assumes the first non-empty table contains the header.
    For subsequent tables, if the header row is identical, skip it.
    """
    merged = []
    header = None
    for table in tables:
        if not table:
            continue
        if header is None:
            header = table[0]
            merged.append(header)
            merged.extend(table[1:])
        else:
            if table[0] == header:
                merged.extend(table[1:])
            else:
                merged.extend(table)
    return merged


def table_to_markdown(table):
    """Convert a 2D table (list of rows) to a Markdown table string."""
    if not table:
        return "No table data extracted."
    header = table[0]
    md_lines = ["| " + " | ".join(header) + " |",
                "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in table[1:]:
        md_lines.append("| " + " | ".join(row) + " |")
    return "\n".join(md_lines)


def main(pdf_path):
    images = pdf_to_images(pdf_path)
    print(f"Converted PDF to {len(images)} images.")
    all_tables = []
    for idx, img in enumerate(images):
        print(f"Processing image {idx + 1} of {len(images)}...")
        response_text = extract_table_from_image(img)
        print(f"Response for image {idx + 1}:\n{response_text}\n")
        table = parse_table_json(response_text)
        if table:
            all_tables.append(table)
    merged = merge_tables(all_tables)
    md_table = table_to_markdown(merged)

    output_dir = os.path.join("src", "md_versions2")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "result.md")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(md_table)
    print(f"Markdown table saved to {output_file}")


if __name__ == "__main__":
    pdf_path = "C:/Users/trist/PycharmProjects/RAG-Studienarbeit/Docs/A.01_Leistungsuebersicht_en.pdf"
    # Ensure the API key is set
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Please set the GEMINI_API_KEY environment variable.")
        sys.exit(1)
    genai.configure(api_key=api_key)
    main(pdf_path)