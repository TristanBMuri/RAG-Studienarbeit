import getpass
from pathlib import Path
from complete_pipeline.prompt.SimpleLLMConversation import SimpleLLMConversation
from complete_pipeline.prompt.langfuse_prompter import Langfuse_Prompter
from preprocessing_scripts.mark_any_down import mark_any_down
from preprocessing_scripts.pdf_to_md_converter import convert_pdfs_to_markdown
from preprocessing_scripts.any_unstructured import *
from vector_db.chroma import *
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from langfuse import Langfuse

def _md_preprocess(input_dir, output_dir):
    md_folder = Path(output_dir)
    md_folder.mkdir(parents=True, exist_ok=True)

    # Conversion of PDFs
    convert_pdfs_to_markdown(input_dir, output_dir)

    # Conversion of non-PDFs
    for each_file in input_dir.iterdir():
        if each_file.suffix.lower() == ".pdf":
            continue
        else:
            mark_any_down(each_file, output_dir)


class RAG_Fuse:
    def __init__(self):
        self.model = None  # Placeholder for an LLM model
        self.md_folder = Path("../src/md_versions")
        self.vectorstore = None
        self.chat = Langfuse_Prompter()

    def preprocess(self, settings = "d"):
        # Prompt user for input and output directories
        if settings != "d":
            settings = input("Set the preprocessing settings (c for custom, d for default and skip to skip): ")

        if settings == "skip":
            self.md_folder = Path("../src/md_versions")
            return None

        if settings == "d":
            input_dir = Path("../Docs")
            output_dir = Path("../src/md_versions")
            _md_preprocess(input_dir, output_dir)
            self.md_folder = output_dir
            return None

        if settings == "c":
            preprocessor_choice = input("Type md for markitdown preprocessing or leave empty for default: ")
            input_dir = input("Enter the document folder directory (or type N for default): ")
            output_dir = input("Enter the output folder directory (or type N for default): ")

            if preprocessor_choice == "u":
                if input_dir.lower() == "n":
                    input_dir = Path("../Docs")
                else:
                    input_dir = Path(input_dir)

                if output_dir.lower() == "n":
                    output_dir = Path("../src/Unstructured")
                else:
                    output_dir = Path(output_dir)

                self.md_folder = output_dir
                convert_folder(input_dir, output_dir)


            if preprocessor_choice == "md":
                if input_dir.lower() == "n":
                    input_dir = Path("../Docs")
                else:
                    input_dir = Path(input_dir)

                if output_dir.lower() == "n":
                    output_dir = Path("../src/md_versions")
                else:
                    output_dir = Path(output_dir)

                _md_preprocess(input_dir, output_dir)
                self.md_folder = output_dir

    def embbed(self):
        # Gather all markdown files from the markdown folder.
        markdown_files = list(self.md_folder.glob("*.md"))
        if not markdown_files:
            print(f"No markdown files found in {self.md_folder}.")
            return

        documents = []
        for file in markdown_files:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.append(content)
            except Exception as e:
                print(f"Error reading {file}: {e}")

        # Filter out invalid documents
        documents = [doc for doc in documents if doc]

        print(f"Embedding {len(documents)} documents...")
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        embeddings = embedding_function.embed_documents(documents)
        embeddings_array = np.array(embeddings)

        # Hier übergeben wir sowohl embeddings_array als auch documents
        self.vectorstore = initialize_vector_db_langchain(embeddings_array, documents)
        print("Vector store initialized.")

    def query(self):
        """
        Prompts for a search query and returns the n most relevant documents from the vector DB.
        """
        # TODO: return reasoning and ranking for prompt
        if self.vectorstore is None:
            print("Vector store not initialized. Please run embbed() first.")
            return []
        query_input = input("Enter your search query: ")
        results = query_vector_db_langchain(self.vectorstore, query_input, k=3)
        return results

    def prompt(self, prompt= None):
        """
        Takes a user question, retrieves context from the vector DB,
        and simulates sending it to an LLM to generate a response.
        """
        if prompt is None:
            query_input = input("Enter your question: ")
        else:
            query_input = prompt

        if self.vectorstore is None:
            print("Vector store not initialized. Please run embbed() first.")
            return

        results = query_vector_db_langchain(self.vectorstore, query_input, k=3)
        context = "\n".join([getattr(res, "page_content", str(res)) for res in results])
        response = self.chat.prompt(query_input, context)
        return response


if __name__ == "__main__":
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

    rag = RAG_Fuse()
    # Run preprocessing to generate markdown versions of documents
    rag.preprocess()

    # Compute embeddings and initialize the vector DB
    rag.embbed()

    test_question = " Was is der unterschied zwischen NATURA und NATURA plus? "
    #re = rag.prompt(test_question)
    # Interactive loop for querying and prompting
    while True:
        action = input("Enter action (query, prompt, or exit): ").lower()
        if action == "query":
            results = rag.query()
            print("Query results:", results)
        elif action == "prompt":
            print(rag.prompt())
        elif action == "exit":
            break
        else:
            print("Invalid action. Please choose 'query', 'prompt', or 'exit'.")
