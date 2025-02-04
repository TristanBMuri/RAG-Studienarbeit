import os
from pathlib import Path
from markitdown import MarkItDown


def convert_pdfs_to_markdown():
    # Define source and destination directories
    md = MarkItDown()
    docs_dir = Path("Docs")
    md_dir = Path("src/md_versions")

    # Create destination directory if it doesn't exist
    md_dir.mkdir(parents=True, exist_ok=True)

    # Get all PDF files from Docs directory
    pdf_files = list(docs_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in the Docs directory.")
        return

    # Convert each PDF file to markdown
    for pdf_path in pdf_files:
        # Create output markdown filename
        md_filename = pdf_path.stem + ".md"
        md_path = md_dir / md_filename

        try:
            # Convert PDF to markdown
            temp = md.convert(str(pdf_path))
            # create a md file from md_filename at md_path from temp
            write_markdown_file(md_path, temp.text_content)
            print(temp)
            print(f"Successfully converted: {pdf_path.name} -> {md_filename}")
        except Exception as e:
            print(f"Error converting {pdf_path.name}: {str(e)}")

def write_markdown_file(md_path, temp_content):
    try:
        with open(md_path, 'w', encoding='utf-8') as md_file:
            md_file.write(temp_content)
        return True
    except Exception as e:
        print(f"Error writing markdown file {md_path}: {str(e)}")
        return False


if __name__ == "__main__":
    convert_pdfs_to_markdown()