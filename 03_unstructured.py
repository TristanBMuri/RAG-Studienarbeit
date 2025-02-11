import os
import logging
from unstructured.partition.pdf import partition_pdf
from bs4 import BeautifulSoup

# Hardcoded directories – update these paths to your actual folders
INPUT_FOLDER = "./Docs"
OUTPUT_FOLDER = "./src/Unstructured"

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
    This function uses BeautifulSoup to extract rows and cells,
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

def convert_pdf_to_markdown(input_path, output_path):
    try:
        logging.info(f"Processing file: {input_path}")
        # Extract elements with table inference enabled
        elements = partition_pdf(
            filename=input_path,
            infer_table_structure=True,
            strategy="hi_res"
        )
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
                # For non-table elements, use a built-in conversion if available
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
    for file_name in os.listdir(INPUT_FOLDER):
        if file_name.lower().endswith(".pdf"):
            input_file_path = os.path.join(INPUT_FOLDER, file_name)
            output_file_name = os.path.splitext(file_name)[0] + ".md"
            output_file_path = os.path.join(OUTPUT_FOLDER, output_file_name)
            convert_pdf_to_markdown(input_file_path, output_file_path)

if __name__ == "__main__":
    main()
