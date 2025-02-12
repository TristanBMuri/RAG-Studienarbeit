from markitdown import MarkItDown


if __name__ == '__main__':
    md = MarkItDown()
    result = md.convert("Docs/sample-1.pptx")
    print(result.text_content)