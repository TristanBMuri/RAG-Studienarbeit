from markitdown import MarkItDown

from preprocessing_scripts.pdf_to_md_converter import write_markdown_file


def mark_any_down(file_path, output_dir):
    md = MarkItDown()
    file_dir = file_path
    output_dir = output_dir

    md_file_name = file_path.stem + ".md"
    md_path = output_dir / md_file_name

    try:
        temp = md.convert(file_dir)
        write_markdown_file(md_path, temp.text_content)
        print(f"Successfully converted: {file_path.name} -> {md_file_name}")
    except Exception as e:
        print(f"Error converting {file_path.name}: {str(e)}")


if __name__ == '__main__':
    md = MarkItDown()
    result = md.convert("../Docs/sample-1.pptx")
    print(result.text_content)