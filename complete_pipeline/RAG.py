from pathlib import Path

from preprocessing_scripts.mark_any_down import mark_any_down
from preprocessing_scripts.pdf_to_md_converter import convert_pdfs_to_markdown


def _preprocess(input_dir, output_dir):
    MD_FOLDER = Path(output_dir)
    MD_FOLDER.mkdir(parents=True, exist_ok=True)

    # convertion of pdfs
    convert_pdfs_to_markdown(input_dir, output_dir)

    # convertion of non pdfs
    for each_file in input_dir:
        if each_file.endswith(".pdf"):
            pass
        else:
            mark_any_down(each_file, output_dir)


class RAG:
    def __init__(self):
        self.model = None
        MD_FOLDER = Path("../src/md_versions")

    def preprocess(self):
        # input for input output folders
        input_dir = input("Enter the document folder directory (or type N for default): ")
        output_dir = input("Enter the output folder directory (or type N for default): ")

        if input_dir == "N" or input_dir == "n":
            input_dir = Path("../Docs")

        if output_dir == "N" or output_dir == "n":
            output_dir = Path("../src/md_versions")

        _preprocess(input_dir, output_dir)

    def embbed(self):
        # embbed all files from the output MD_FOLDER w/ huggingface embeddings or Google Text embedding 004 via api
        # use chroma.py to create the vector db
        pass

    def query(self):
        # take input and do a vector search return n most relevant vectors
        pass

    def prompt(self):
        # take input from query, send to llm, return response to user
        pass

