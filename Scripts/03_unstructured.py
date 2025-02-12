import os
import logging
from bs4 import BeautifulSoup

# Hardcoded directories – update these paths to your actual folders
INPUT_FOLDER = "../Docs"
OUTPUT_FOLDER = "../src/Unstructured"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Create output folder if it does not exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def html_table_to_markdown(html):
    """
    Convert an HTML table to Markdown format.
    Uses BeautifulSoup to extract rows and cells,
    then formats the header and subsequent rows according to Markdown table syntax.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return html  # Fallback to original HTML if no table is found
    rows = table.find_all("tr")
    markdown_lines = []
    for i, row in enumerate(rows):
        cells = row.find_all(["th", "td"])
        cell_texts = [cell.get_text(strip=True) for cell in cells if cell.get_text(strip=True)]
        if not cell_texts:
            continue
        # Build a Markdown table row
        row_md = "| " + " | ".join(cell_texts) + " |"
        markdown_lines.append(row_md)
        # After the header row, add a divider line
        if i == 0:
            divider = "| " + " | ".join(["---"] * len(cell_texts)) + " |"
            markdown_lines.append(divider)
    return "\n".join(markdown_lines)

def convert_file_to_markdown(input_path, output_path):
    try:
        logging.info(f"Processing file: {input_path}")
        extension = os.path.splitext(input_path)[1].lower()
        # Choose the partition function based on file extension
        if extension == ".pdf":
            from unstructured.partition.pdf import partition_pdf
            elements = partition_pdf(
                filename=input_path,
                infer_table_structure=True,
                strategy="hi_res"
            )
        elif extension == ".docx":
            from unstructured.partition.docx import partition_docx
            elements = partition_docx(filename=input_path)
        elif extension in [".html", ".htm"]:
            from unstructured.partition.html import partition_html
            elements = partition_html(filename=input_path)
        elif extension == ".txt":
            from unstructured.partition.text import partition_text
            elements = partition_text(filename=input_path)
        elif extension == ".pptx":
            from unstructured.partition.pptx import partition_pptx
            elements = partition_pptx(filename=input_path)
        else:
            # Fall back to auto partition if file type is unrecognized
            from unstructured.partition.auto import partition
            elements = partition(filename=input_path)

        # Process elements into Markdown
        markdown_content = []
        for elem in elements:
            if elem.category == "Table":
                # If table metadata contains an HTML representation, convert it to Markdown
                if hasattr(elem, "metadata") and hasattr(elem.metadata, "text_as_html") and elem.metadata.text_as_html:
                    md_table = html_table_to_markdown(elem.metadata.text_as_html)
                    markdown_content.append(md_table)
                else:
                    markdown_content.append(elem.text)
            else:
                # Use built-in conversion if available, otherwise use plain text
                if hasattr(elem, "to_markdown"):
                    markdown_content.append(elem.to_markdown())
                else:
                    markdown_content.append(elem.text)
        final_markdown = "\n\n".join(markdown_content)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_markdown)
        logging.info(f"Saved converted Markdown to: {output_path}")
    except Exception as e:
        logging.error(f"Error processing {input_path}: {e}")

def main():
    # Define supported file extensions
    supported_extensions = (".pdf", ".docx", ".html", ".htm", ".txt", ".pptx")
    for file_name in os.listdir(INPUT_FOLDER):
        if file_name.lower().endswith(supported_extensions):
            input_file_path = os.path.join(INPUT_FOLDER, file_name)
            output_file_name = os.path.splitext(file_name)[0] + ".md"
            output_file_path = os.path.join(OUTPUT_FOLDER, output_file_name)
            convert_file_to_markdown(input_file_path, output_file_path)

if __name__ == "__main__":
    main()
